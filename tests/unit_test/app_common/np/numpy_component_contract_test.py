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

import ast
from unittest.mock import Mock

import numpy as np
import pytest

from nvflare.apis.dxo import DXO, DataKind, MetaKey, from_shareable
from nvflare.apis.fl_constant import FLContextKey, ReservedKey
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import ReturnCode, Shareable
from nvflare.apis.signal import Signal
from nvflare.app_common.app_constant import AppConstants
from nvflare.app_common.ccwf.comps import np_trainer as ccwf_trainer
from nvflare.app_common.np import np_trainer, np_validator
from nvflare.app_common.np.constants import NPConstants
from nvflare.app_common.np.np_formatter import NPFormatter
from nvflare.app_common.np.np_model_locator import NPModelLocator
from tests.timing_utils import isolate_time


def context(root):
    ctx = FLContext()
    engine = Mock()
    engine.get_workspace.return_value.get_run_dir.return_value = str(root)
    engine.get_workspace.return_value.get_result_root.return_value = str(root)
    ctx.set_prop(ReservedKey.ENGINE, engine, private=True, sticky=True)
    ctx.set_prop(FLContextKey.CURRENT_RUN, "job", private=True, sticky=True)
    return ctx


@pytest.fixture(params=[np_trainer, ccwf_trainer], ids=["full-weight-trainer", "ccwf-delta-trainer"])
def trainer_module(request):
    return request.param


def test_numpy_training_persists_full_model_and_returns_expected_weight_representation(trainer_module, tmp_path):
    trainer = trainer_module.NPTrainer(delta=0.5)
    ctx = context(tmp_path)
    initial = np.ones((3, 3), dtype=np.float32)
    request = DXO(DataKind.WEIGHTS, {NPConstants.NUMPY_KEY: initial}).to_shareable()
    request.set_header(AppConstants.CURRENT_ROUND, 2)
    request.set_header(AppConstants.NUM_ROUNDS, 4)
    reply = trainer.execute(AppConstants.TASK_TRAIN, request, ctx, Signal())
    result = from_shareable(reply)
    assert result.meta[MetaKey.NUM_STEPS_CURRENT_ROUND] == 1
    if trainer_module is ccwf_trainer:
        assert result.data_kind == DataKind.WEIGHT_DIFF
        np.testing.assert_array_equal(result.data[NPConstants.NUMPY_KEY], np.full((3, 3), 0.5))
        np.testing.assert_array_equal(initial, np.ones((3, 3)))
    else:
        assert result.data_kind == DataKind.WEIGHTS
        np.testing.assert_array_equal(result.data[NPConstants.NUMPY_KEY], np.full((3, 3), 1.5))
    submitted = from_shareable(trainer.execute(AppConstants.TASK_SUBMIT_MODEL, Shareable(), ctx, Signal()))
    assert submitted.data_kind == DataKind.WEIGHTS
    np.testing.assert_array_equal(submitted.data[NPConstants.NUMPY_KEY], np.full((3, 3), 1.5))
    assert trainer.execute("unknown", Shareable(), ctx, Signal()).get_return_code() == ReturnCode.TASK_UNKNOWN


@pytest.mark.parametrize("bad", ["malformed", "kind", "empty", "missing-key"])
def test_numpy_trainers_reject_invalid_model_data_before_saving(trainer_module, tmp_path, bad):
    trainer = trainer_module.NPTrainer()
    trainer.system_panic = Mock()
    trainer._save_local_model = Mock()
    request = {
        "malformed": Shareable(),
        "kind": DXO(DataKind.METRICS, {"metric": 1}).to_shareable(),
        "empty": DXO(DataKind.WEIGHTS, {}).to_shareable(),
        "missing-key": DXO(DataKind.WEIGHTS, {"other": [1]}).to_shareable(),
    }[bad]
    assert (
        trainer.execute(AppConstants.TASK_TRAIN, request, context(tmp_path), Signal()).get_return_code()
        == ReturnCode.BAD_TASK_DATA
    )
    trainer._save_local_model.assert_not_called()


@pytest.mark.parametrize("mode", ["before", "after-save", "save-failure"])
def test_training_checks_cancellation_and_tolerates_checkpoint_write_failure(trainer_module, tmp_path, mode):
    trainer = trainer_module.NPTrainer()
    signal = Signal()
    if mode == "before":
        signal.trigger(True)
    elif mode == "after-save":
        trainer._save_local_model = Mock(side_effect=lambda *_: signal.trigger(True))
    else:
        trainer._save_local_model = Mock(side_effect=OSError("disk full"))
    request = DXO(DataKind.WEIGHTS, {NPConstants.NUMPY_KEY: np.ones((3, 3))}).to_shareable()
    result = trainer.execute(AppConstants.TASK_TRAIN, request, context(tmp_path), signal)
    assert result.get_return_code() == (ReturnCode.OK if mode == "save-failure" else ReturnCode.TASK_ABORTED)


def test_numpy_trainers_return_model_load_errors_and_abort_sleep_without_real_delay(
    trainer_module, tmp_path, monkeypatch
):
    trainer = trainer_module.NPTrainer(sleep_time=1)
    ctx = context(tmp_path)
    signal = Signal()
    clock = isolate_time(monkeypatch, trainer_module)
    clock.sleep = Mock(side_effect=lambda _: signal.trigger(True))
    assert (
        trainer.execute(AppConstants.TASK_TRAIN, Shareable(), ctx, signal).get_return_code() == ReturnCode.TASK_ABORTED
    )
    clock.sleep.assert_called_once_with(0.5)
    trainer._sleep_time = 0
    assert trainer._submit_model(ctx, Signal()).get_return_code() == ReturnCode.EXECUTION_RESULT_ERROR
    trainer._load_local_model = Mock(side_effect=OSError("load failure"))
    assert trainer._submit_model(ctx, Signal()).get_return_code() == ReturnCode.EXECUTION_RESULT_ERROR
    assert trainer._submit_model(ctx, signal).get_return_code() == ReturnCode.TASK_ABORTED
    trainer._train = Mock(side_effect=ValueError("bad training input"))
    assert (
        trainer.execute(AppConstants.TASK_TRAIN, Shareable(), ctx, Signal()).get_return_code()
        == ReturnCode.EXECUTION_EXCEPTION
    )


def test_numpy_validator_computes_normalized_metric_and_returns_metrics_dxo(monkeypatch):
    validator = np_validator.NPValidator()
    monkeypatch.setattr(np_validator.np.random, "random", lambda: 0.25)
    request = DXO(DataKind.WEIGHTS, {NPConstants.NUMPY_KEY: np.array([1.0, 2.0, 4.0])}).to_shareable()
    request.set_header(AppConstants.MODEL_OWNER, "site-1")
    result = from_shareable(validator.execute(AppConstants.TASK_VALIDATION, request, FLContext(), Signal()))
    assert result.data_kind == DataKind.METRICS
    assert result.data["accuracy"] == pytest.approx(2.0)


@pytest.mark.parametrize("bad", ["malformed", "kind", "empty", "missing", "array", "abort"])
def test_numpy_validator_rejects_bad_inputs_or_honors_abort(bad):
    validator = np_validator.NPValidator()
    data = {NPConstants.NUMPY_KEY: "not-numeric" if bad == "array" else np.array([1.0])}
    if bad == "empty":
        data = {}
    elif bad == "missing":
        data = {"unknown": 1}
    request = (
        Shareable()
        if bad == "malformed"
        else DXO(DataKind.METRICS if bad == "kind" else DataKind.WEIGHTS, data).to_shareable()
    )
    signal = Signal()
    if bad == "abort":
        signal.trigger(True)
    expected = (
        ReturnCode.TASK_ABORTED
        if bad == "abort"
        else ReturnCode.EXECUTION_EXCEPTION if bad == "array" else ReturnCode.BAD_TASK_DATA
    )
    assert validator.execute(AppConstants.TASK_VALIDATION, request, FLContext(), signal).get_return_code() == expected
    assert (
        validator.execute(AppConstants.TASK_SUBMIT_MODEL, Shareable(), FLContext(), Signal()).get_return_code()
        == ReturnCode.OK
    )
    assert validator.execute("unknown", Shareable(), FLContext(), Signal()).get_return_code() == ReturnCode.TASK_UNKNOWN


def test_numpy_formatter_loads_only_metrics_and_contains_file_errors(tmp_path):
    metrics, weights = tmp_path / "metrics", tmp_path / "weights"
    DXO(DataKind.METRICS, {"accuracy": 0.8}).to_file(str(metrics))
    DXO(DataKind.WEIGHTS, {"weight": [1]}).to_file(str(weights))
    ctx = FLContext()
    ctx.set_prop(
        AppConstants.VALIDATION_RESULT,
        {"data-site": {"good": str(metrics), "ignored": str(weights)}, "empty": {}},
        private=True,
        sticky=False,
    )
    assert ast.literal_eval(NPFormatter().format(ctx)) == {"data-site": {"good": {"accuracy": 0.8}}}
    ctx.set_prop(
        AppConstants.VALIDATION_RESULT, {"site": {"bad": str(tmp_path / "missing")}}, private=True, sticky=False
    )
    assert ast.literal_eval(NPFormatter().format(ctx)) == {"site": {}}


@pytest.mark.parametrize("absolute", [False, True])
def test_numpy_model_locator_loads_declared_names_and_returns_none_for_absent_models(tmp_path, absolute):
    directory = tmp_path / "models"
    directory.mkdir()
    path = directory / "model.npy"
    np.save(path, np.array([2, 3], dtype=np.float32))
    locator = NPModelLocator(model_name={"trained": str(path) if absolute else "model.npy", "missing": "missing.npy"})
    assert locator.get_model_names(context(tmp_path)) == ["trained", "missing"]
    result = locator.locate_model("trained", context(tmp_path))
    assert result.data_kind == DataKind.WEIGHTS
    np.testing.assert_array_equal(result.data[NPConstants.NUMPY_KEY], [2, 3])
    assert locator.locate_model("missing", context(tmp_path)) is None
    assert locator.locate_model("unknown", context(tmp_path)) is None
    assert NPModelLocator(model_name=None).get_model_names(None) == ["server"]
    with pytest.raises(ValueError, match="model_name"):
        NPModelLocator(model_name=[])
