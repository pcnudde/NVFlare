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

from unittest.mock import MagicMock

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")

from nvflare.apis.dxo import DXO, DataKind, from_shareable
from nvflare.apis.fl_constant import ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.apis.signal import Signal
from nvflare.app_common.app_constant import AppConstants
from nvflare.app_opt.tf.tf_validator import TFValidator


def _model():
    return tf.keras.Sequential([tf.keras.Input((2,)), tf.keras.layers.Dense(1, use_bias=False, name="dense")])


def _request():
    result = DXO(DataKind.WEIGHTS, {"dense_nvf_0": np.array([[2.0], [3.0]], dtype=np.float32)}).to_shareable()
    result.set_header(AppConstants.MODEL_OWNER, "server")
    return result


def test_validator_loads_flat_weights_before_custom_metric_and_returns_metrics_dxo():
    model = _model()

    def metric(subject, loader):
        assert subject is model
        assert loader == "data"
        return {"prediction": float(subject(tf.constant([[1.0, 1.0]])).numpy()[0, 0])}

    validator = TFValidator(model, data_loader="data", metric_fn=metric)
    result = validator.execute(AppConstants.TASK_VALIDATION, _request(), FLContext(), Signal())
    dxo = from_shareable(result)
    assert dxo.data_kind == DataKind.METRICS
    assert dxo.data == {"prediction": 5.0}


@pytest.mark.parametrize(
    "mode", ["unknown", "invalid_dxo", "empty_weights", "wrong_kind", "weight_error", "no_evaluator", "metric_error"]
)
def test_validator_maps_task_and_payload_failures_to_return_codes(mode):
    model = _model()
    validator = TFValidator(model)
    request = _request()
    task = AppConstants.TASK_VALIDATION
    expected = ReturnCode.EXECUTION_EXCEPTION
    if mode == "unknown":
        task = "unknown"
        expected = ReturnCode.TASK_UNKNOWN
    elif mode == "invalid_dxo":
        request = Shareable()
        expected = ReturnCode.BAD_TASK_DATA
    elif mode == "empty_weights":
        request = DXO(DataKind.WEIGHTS, {}).to_shareable()
        expected = ReturnCode.BAD_TASK_DATA
    elif mode == "wrong_kind":
        request = DXO(DataKind.METRICS, {"loss": 1}).to_shareable()
        expected = ReturnCode.BAD_TASK_DATA
    elif mode == "weight_error":
        request = DXO(DataKind.WEIGHTS, {"missing_nvf_0": np.ones(1)}).to_shareable()
    elif mode == "metric_error":
        validator.metric_fn = MagicMock(side_effect=ValueError("metric failure"))
    result = validator.execute(task, request, FLContext(), Signal())
    assert result.get_return_code() == expected


@pytest.mark.parametrize("stage", ["before_load", "after_load", "after_metric"])
def test_validator_observes_abort_between_load_and_metric_boundaries(monkeypatch, stage):
    model = _model()
    abort = Signal()
    metric = MagicMock(return_value={"accuracy": 1.0})
    validator = TFValidator(model, metric_fn=metric)
    load = model.layers[0].set_weights

    def aborting_load(weights):
        load(weights)
        abort.trigger(True)

    if stage == "before_load":
        abort.trigger(True)
    elif stage == "after_load":
        monkeypatch.setattr(model.layers[0], "set_weights", aborting_load)
    else:

        def metric_and_abort(*args):
            abort.trigger(True)
            return {"accuracy": 1.0}

        metric.side_effect = metric_and_abort
    result = validator.execute(AppConstants.TASK_VALIDATION, _request(), FLContext(), abort)
    assert result.get_return_code() == ReturnCode.TASK_ABORTED
    assert metric.call_count == int(stage == "after_metric")


def test_default_evaluation_delegates_with_quiet_flag_and_serializes_numeric_metrics(monkeypatch):
    model = _model()
    evaluate = MagicMock(return_value=(np.float32(0.25), np.float32(0.75)))
    monkeypatch.setattr(model, "evaluate", evaluate)
    validator = TFValidator(model, data_loader="dataset")
    result = from_shareable(validator.execute(AppConstants.TASK_VALIDATION, _request(), FLContext(), Signal()))
    assert result.data == {"loss": 0.25, "accuracy": 0.75}
    evaluate.assert_called_once_with("dataset", verbose=0)


def test_validator_rejects_non_mapping_metric_payload():
    context = FLContext()
    validator = TFValidator(_model(), metric_fn=lambda model, data: [0.1, 0.2])
    assert (
        validator.execute(AppConstants.TASK_VALIDATION, _request(), context, Signal()).get_return_code()
        == ReturnCode.EXECUTION_EXCEPTION
    )
