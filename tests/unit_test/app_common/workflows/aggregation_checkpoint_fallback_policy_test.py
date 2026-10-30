# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

from unittest.mock import Mock

import pytest

from nvflare.apis.fl_context import FLContext
from nvflare.app_common.abstract.fl_model import FLModel
from nvflare.app_common.app_constant import AppConstants
from nvflare.app_common.workflows import base_fedavg as base
from nvflare.app_common.workflows.fedavg import FedAvg


def test_fedavg_simple_checkpoint_round_trip_and_missing_file_fallback(tmp_path, monkeypatch):
    controller = FedAvg(persistor_id="", save_filename="checkpoint.bin")
    controller.persistor = None
    monkeypatch.setattr(controller, "get_run_dir", lambda: str(tmp_path))
    empty = controller.load_model()
    assert isinstance(empty, FLModel) and empty.params == {}
    model = FLModel(params={"weight": [1, 2]}, metrics={"accuracy": 0.8}, meta={"round": 3})
    controller.save_model(model)
    assert (tmp_path / "checkpoint.bin").is_file()
    restored = controller.load_model()
    assert restored.params == model.params
    assert restored.metrics == model.metrics and restored.meta == model.meta


def test_fedavg_disabled_persistence_warns_without_writing(tmp_path, monkeypatch):
    controller = FedAvg(persistor_id="", save_filename=None)
    controller.persistor = None
    warning = Mock()
    monkeypatch.setattr(controller, "warning", warning)
    controller.save_model(FLModel(params={"weight": 1}))
    warning.assert_called_once_with("No persistor or save_filename configured, model not saved")
    assert list(tmp_path.iterdir()) == []


def test_aggregation_callback_failure_panics_without_reusing_prior_round_stats(monkeypatch):
    controller = FedAvg(persistor_id="")
    controller.fl_ctx = FLContext()
    controller.fl_ctx.set_prop(AppConstants.AGGREGATION_STATS, {"old": True})
    monkeypatch.setattr(controller, "event", Mock())
    panic = Mock()
    monkeypatch.setattr(controller, "panic", panic)
    callback = Mock(side_effect=ValueError("shape mismatch"))
    result = controller.aggregate([FLModel(params={"weight": 1})], callback)
    assert isinstance(result, FLModel) and result.params is None
    assert "shape mismatch" in panic.call_args.args[0]
    assert controller.fl_ctx.get_prop(AppConstants.AGGREGATION_STATS) is None


def test_stats_failure_cannot_discard_valid_aggregation_result(monkeypatch):
    controller = FedAvg(persistor_id="")
    controller.fl_ctx = FLContext()
    monkeypatch.setattr(controller, "event", Mock())
    event = Mock()
    monkeypatch.setattr(controller, "fire_event_with_data", event)
    monkeypatch.setattr(base, "compute_key_match_stats", Mock(side_effect=ValueError("stats unavailable")))
    model = FLModel(params={"weight": 2})
    result = controller.aggregate([FLModel(params={"weight": 1})], lambda values: model)
    assert result is model
    assert controller.fl_ctx.get_prop(AppConstants.AGGREGATION_STATS) is None
    assert event.call_args.args[-1] is model


@pytest.mark.parametrize("results", [[], [FLModel(params={})]])
def test_aggregation_rejects_empty_results_or_empty_client_parameters(results):
    with pytest.raises(ValueError, match="empty"):
        if results:
            base.BaseFedAvg._check_results(results)
        else:
            base.BaseFedAvg.aggregate_fn(results)


@pytest.mark.parametrize(
    "condition,expected", [("loss < 2", "min"), ("accuracy >= 3", "max"), ("bad", None), ("metric == 1", None)]
)
def test_metrics_reporting_derives_direction_only_from_supported_stop_condition(condition, expected):
    info = base.make_key_metric_info_from_stop_condition(condition, ("metric", "op", 1))
    assert (info["mode"] if info else None) == expected
    aggregation = base.make_fedavg_metrics_aggregation_info("accuracy", "max", "operator")
    assert aggregation["key_metric"] == {"name": "accuracy", "mode": "max", "mode_source": "operator"}
