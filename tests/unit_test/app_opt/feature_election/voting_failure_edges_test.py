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

import numpy as np
import pytest

pytest.importorskip("sklearn")

from nvflare.apis.fl_context import FLContext  # noqa: E402
from nvflare.apis.shareable import ReturnCode, Shareable  # noqa: E402
from nvflare.apis.signal import Signal  # noqa: E402
from nvflare.app_opt.feature_election import executor as executor_module  # noqa: E402
from nvflare.app_opt.feature_election.controller import FeatureElectionController  # noqa: E402
from nvflare.app_opt.feature_election.executor import FeatureElectionExecutor  # noqa: E402


def test_empty_election_requires_known_feature_dimension_and_identical_masks_need_no_voting():
    controller = FeatureElectionController()
    with pytest.raises(ValueError, match="n_features"):
        controller.aggregate_selections({})
    controller.n_features = 3
    np.testing.assert_array_equal(controller.aggregate_selections({}), [False, False, False])
    selections = {
        "a": {"selected_features": [True, False, True], "feature_scores": [1, 0, 1], "num_samples": 2},
        "b": {"selected_features": [True, False, True], "feature_scores": [2, 0, 2], "num_samples": 3},
    }
    np.testing.assert_array_equal(controller.aggregate_selections(selections), [True, False, True])


def test_weighted_election_skips_empty_client_and_normalizes_constant_scores_without_nan():
    controller = FeatureElectionController(freedom_degree=0.5)
    masks = np.array([[False, False, False], [True, True, False], [False, True, True]])
    scores = np.array([[0.0, 0.0, 0.0], [2.0, 2.0, 0.0], [0.0, 1.0, 3.0]])
    result = controller._weighted_election(
        masks, scores, np.array([0.0, 0.5, 0.5]), np.all(masks, axis=0), np.any(masks, axis=0)
    )
    np.testing.assert_array_equal(result, [False, True, True])
    controller.freedom_degree = 0
    np.testing.assert_array_equal(
        controller._weighted_election(
            masks, scores, np.array([0.0, 0.5, 0.5]), np.all(masks, axis=0), np.any(masks, axis=0)
        ),
        [False, False, False],
    )


def test_freedom_degree_improvement_keeps_direction_and_first_move_turns_away_from_upper_bound():
    controller = FeatureElectionController(freedom_degree=0.99)
    controller.search_step = 0.1
    assert controller._calculate_next_fd(first_step=True) == pytest.approx(0.89)
    assert controller.current_direction == -1
    controller.tuning_history = [(0.8, 0.5), (0.7, 0.8)]
    assert controller._calculate_next_fd(first_step=False) == pytest.approx(0.6)
    controller.tuning_history = []
    assert controller._calculate_next_fd(first_step=False) == 0.99


@pytest.fixture
def executor():
    item = FeatureElectionExecutor(fs_method="mutual_info", fs_params={"k": 1})
    features = np.column_stack((np.arange(20, dtype=float), np.arange(20) % 3))
    labels = np.arange(20) % 2
    item.set_data(features, labels)
    return item


def test_invalid_mask_shapes_fail_before_permanent_feature_reduction(executor):
    original = executor.X_train.copy()
    for mask in ([True], 3):
        result = executor._handle_apply_mask(Shareable({"global_feature_mask": mask}))
        assert result.get_return_code() == ReturnCode.EXECUTION_EXCEPTION
        np.testing.assert_array_equal(executor.X_train, original)
        assert executor.global_feature_mask is None
    result = executor._handle_tuning_eval(Shareable({"tuning_mask": [[True, False], [False, True]]}))
    assert result.get_return_code() == ReturnCode.EXECUTION_EXCEPTION
    np.testing.assert_array_equal(executor.X_train, original)


def test_train_request_accepts_one_dimensional_binary_global_coefficients(executor):
    request = Shareable({"request_type": "train", "params": {"weight_0": [0.0, 0.0], "weight_1": [0.0]}})
    result = executor.execute(executor.task_name, request, FLContext(), Signal())
    assert result.get_return_code() == ReturnCode.OK
    assert np.asarray(result["params"]["weight_0"]).shape == (1, 2)
    assert result["num_samples"] == 20
    assert np.isfinite(result["params"]["weight_0"]).all()


def test_missing_pyimpetus_falls_back_to_real_deterministic_mutual_information(executor, monkeypatch, caplog):
    monkeypatch.setattr(executor_module, "PYIMPETUS_AVAILABLE", False)
    executor.fs_method = "pyimpetus"
    mask, scores = executor.perform_feature_selection()
    assert mask.shape == scores.shape == (2,)
    assert mask.sum() == 1 and np.isfinite(scores).all()
    np.testing.assert_array_equal(np.flatnonzero(mask), np.argsort(scores)[-1:])
    assert "falling back to mutual_info" in caplog.text
