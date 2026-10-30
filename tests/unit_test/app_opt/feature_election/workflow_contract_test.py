# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from nvflare.apis.fl_constant import ReturnCode
from nvflare.apis.shareable import Shareable, make_reply
from nvflare.apis.signal import Signal
from nvflare.app_opt.feature_election.controller import FeatureElectionController
from nvflare.app_opt.feature_election.executor import FeatureElectionExecutor


def _vote(mask, scores, samples):
    return Shareable({"selected_features": mask, "feature_scores": scores, "num_samples": samples})


def _selections():
    return {"a": _vote([True, False, True], [0.9, 0, 0.4], 4), "b": _vote([False, True, True], [0, 0.9, 0.4], 8)}


def test_three_phase_workflow_selects_best_mask_and_aggregates_weighted_rounds(tmp_path):
    controller = FeatureElectionController(min_clients=2, num_rounds=2, auto_tune=True, tuning_rounds=5)
    requests = []

    def gather(task, signal, ctx, timeout=0):
        requests.append(dict(task))
        kind = task["request_type"]
        if kind == "feature_selection":
            return _selections()
        if kind == "tuning_eval":
            return {
                "a": Shareable({"tuning_score": 0.5, "num_samples": 4}),
                "b": Shareable({"tuning_score": 0.5, "num_samples": 8}),
            }
        if kind == "apply_mask":
            return {"a": make_reply(ReturnCode.OK), "b": make_reply(ReturnCode.OK)}
        assert kind == "train" and timeout == 300
        return {
            "a": Shareable({"params": {"weight": [1.0, 2.0]}, "num_samples": 4}),
            "b": Shareable({"params": {"weight": [3.0, 4.0]}, "num_samples": 8}),
        }

    controller._broadcast_and_gather = gather
    signal, ctx = Signal(), MagicMock()
    controller.start_controller(ctx)
    controller.control_flow(signal, ctx)
    assert not signal.triggered
    assert [t["request_type"] for t in requests] == [
        "feature_selection",
        "tuning_eval",
        "tuning_eval",
        "tuning_eval",
        "apply_mask",
        "train",
        "train",
    ]
    assert len(controller.tuning_history) == 3
    assert controller.freedom_degree == controller.tuning_history[0][0]
    np.testing.assert_allclose(controller.global_weights["weight"], [7 / 3, 10 / 3])
    np.testing.assert_allclose(requests[-1]["params"]["weight"], [7 / 3, 10 / 3])
    ctx.get_engine.return_value.get_workspace.return_value.get_run_dir.return_value = str(tmp_path)
    controller.stop_controller(ctx)
    result = json.loads((tmp_path / "feature_election_results.json").read_text())
    assert result["global_mask"] == controller.global_feature_mask.tolist()
    assert result["num_features_selected"] == int(controller.global_feature_mask.sum())


@pytest.mark.parametrize("responses", [{}, {"empty": _vote([False, False], [0, 0], 2)}])
def test_election_does_not_continue_without_meaningful_votes(responses):
    controller = FeatureElectionController()
    controller._broadcast_and_gather = MagicMock(return_value=responses)
    controller._phase_two_tuning_and_masking = MagicMock()
    controller.control_flow(Signal(), MagicMock())
    controller._phase_two_tuning_and_masking.assert_not_called()


def test_mask_requires_quorum_and_empty_tuning_scores_do_not_change_search_state():
    controller = FeatureElectionController(min_clients=2, auto_tune=True, tuning_rounds=2)
    controller.cached_client_selections = controller._extract_client_data(_selections())
    controller._broadcast_and_gather = MagicMock(side_effect=[{"a": Shareable()}, {}, {"a": Shareable()}])
    assert controller._phase_two_tuning_and_masking(Signal(), MagicMock()) is False
    assert controller.tuning_history == []
    assert controller.freedom_degree == 0.5
    assert controller.global_feature_mask is not None


def test_training_phase_respects_early_cancellation():
    controller = FeatureElectionController(num_rounds=3)
    signal = Signal()
    signal.trigger(True)
    controller._broadcast_and_gather = MagicMock()
    controller._phase_three_aggregation(signal, MagicMock())
    controller._broadcast_and_gather.assert_not_called()


def test_broadcast_collects_callbacks_and_backup_results_without_duplicates():
    controller = FeatureElectionController(min_clients=1, wait_time_after_min_received=7)
    good = make_reply(ReturnCode.OK)
    bad = make_reply(ReturnCode.EXECUTION_EXCEPTION)
    ctx = MagicMock()

    def broadcast(**kwargs):
        task = kwargs["task"]
        assert kwargs["min_responses"] == 1 and kwargs["wait_time_after_min_received"] == 7
        controller._result_received_cb(SimpleNamespace(client=SimpleNamespace(name="a"), result=good), ctx)
        controller._result_received_cb(SimpleNamespace(client=SimpleNamespace(name="failed"), result=bad), ctx)
        controller._result_received_cb(SimpleNamespace(client=SimpleNamespace(name="missing"), result=None), ctx)
        task.client_tasks.extend(
            [
                SimpleNamespace(client=SimpleNamespace(name="a"), result=bad),
                SimpleNamespace(client=SimpleNamespace(name="b"), result=good),
                SimpleNamespace(client=SimpleNamespace(name="c"), result=bad),
            ]
        )

    controller.broadcast_and_wait = broadcast
    assert controller._broadcast_and_gather(Shareable(), Signal(), ctx) == {"a": good, "b": good}


@pytest.mark.parametrize("bad_weights", [{"extra": [1.0]}, {"weight": [1.0, 2.0, 3.0]}, {}])
def test_weight_aggregation_rejects_incompatible_client_without_partial_sum(bad_weights):
    controller = FeatureElectionController()
    controller._aggregate_weights(
        {
            "good": Shareable({"params": {"weight": [2.0, 4.0]}, "num_samples": 3}),
            "bad": Shareable({"params": bad_weights, "num_samples": 100}),
        }
    )
    np.testing.assert_allclose(controller.global_weights["weight"], [2.0, 4.0])
    controller._aggregate_weights({"missing": Shareable()})
    np.testing.assert_allclose(controller.global_weights["weight"], [2.0, 4.0])


@pytest.fixture
def data():
    x = np.linspace(-3, 3, 40)
    rng = np.random.default_rng(42)
    return np.column_stack([x, np.sin(x * 17), np.zeros(40), rng.normal(size=40)]), (x > 0).astype(int)


@pytest.mark.parametrize(
    "method,params",
    [
        ("lasso", {"alpha": 0.01}),
        ("elastic_net", {"alpha": 0.01}),
        ("mutual_info", {"k": 2, "random_state": 5}),
        ("random_forest", {"k": 2, "n_estimators": 10, "max_depth": 2}),
    ],
)
def test_feature_selection_recovers_informative_feature_and_preserves_method_options(data, method, params):
    x, y = data
    executor = FeatureElectionExecutor(method, params)
    executor.set_data(x, y)
    response = executor.execute(
        "feature_election", Shareable({"request_type": "feature_selection"}), MagicMock(), Signal()
    )
    assert response.get_return_code() == ReturnCode.OK
    assert response["selected_features"][0]
    assert len(response["feature_scores"]) == 4 and np.isfinite(response["feature_scores"]).all()
    assert response["num_samples"] == 40
    if "k" in params:
        assert sum(response["selected_features"]) == 2
        assert executor.fs_params["k"] == 2


def test_executor_mask_tuning_training_and_retransmission_keep_consistent_feature_space(data):
    x, y = data
    executor = FeatureElectionExecutor(eval_metric="accuracy")
    executor.set_data(
        pd.DataFrame(x), pd.Series(y), pd.DataFrame(x), pd.Series(y), ["signal", "noise", "constant", "other"]
    )
    assert isinstance(executor.X_train, np.ndarray) and isinstance(executor.y_train, np.ndarray)
    mask = [True, False, False, True]
    tuning = executor.execute(
        "feature_election", Shareable({"request_type": "tuning_eval", "tuning_mask": mask}), MagicMock(), Signal()
    )
    assert tuning.get_return_code() == ReturnCode.OK and tuning["tuning_score"] >= 0.95
    apply = Shareable({"request_type": "apply_mask", "global_feature_mask": mask})
    assert executor.execute("feature_election", apply, MagicMock(), Signal()).get_return_code() == ReturnCode.OK
    assert executor.feature_names == ["signal", "other"]
    snapshot = executor.X_train.copy()
    assert executor._handle_apply_mask(apply).get_return_code() == ReturnCode.OK
    np.testing.assert_array_equal(executor.X_train, snapshot)
    assert (
        executor._handle_apply_mask(Shareable({"global_feature_mask": [True, True]})).get_return_code()
        == ReturnCode.EXECUTION_EXCEPTION
    )
    first = executor._handle_train(Shareable())
    assert first.get_return_code() == ReturnCode.OK
    assert np.array(first["params"]["weight_0"]).shape == (1, 2) and first["num_samples"] == 40
    scaler = executor.scaler
    second = executor._handle_train(Shareable({"params": first["params"]}))
    assert second.get_return_code() == ReturnCode.OK
    assert executor.scaler is scaler
    assert executor.evaluate_model(executor.X_train, y, executor.X_val, y, scaler=scaler) >= 0.95


def test_executor_rejects_missing_data_masks_and_unknown_requests(data):
    executor = FeatureElectionExecutor()
    ctx, signal = MagicMock(), Signal()
    assert executor.execute("wrong", Shareable(), ctx, signal).get_return_code() == ReturnCode.TASK_UNKNOWN
    assert (
        executor.execute("feature_election", Shareable(), ctx, signal).get_return_code()
        == ReturnCode.EXECUTION_EXCEPTION
    )
    assert executor._handle_feature_selection().get_return_code() == ReturnCode.EXECUTION_EXCEPTION
    assert executor._handle_apply_mask(Shareable()).get_return_code() == ReturnCode.EXECUTION_EXCEPTION
    assert executor._handle_train(Shareable()).get_return_code() == ReturnCode.EXECUTION_EXCEPTION
    x, y = data
    with pytest.raises(ValueError, match="feature_names"):
        executor.set_data(x, y, feature_names=["one"])
    with pytest.raises(ValueError, match="feature count"):
        executor.set_data(x, y, X_val=x[:, :2], y_val=y)
    executor.set_data(x, y)
    for raw in (None, [False] * 4, [True]):
        assert (
            executor._handle_tuning_eval(Shareable({"tuning_mask": raw})).get_return_code()
            == ReturnCode.EXECUTION_EXCEPTION
        )
    executor.fs_method = "unknown"
    assert executor._handle_feature_selection().get_return_code() == ReturnCode.EXECUTION_EXCEPTION
    assert executor.evaluate_model(x[:0], y[:0], x, y) == 0.0
    assert executor.evaluate_model(x, np.zeros_like(y), x, y) == 0.0
