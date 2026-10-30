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

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from nvflare.apis.dxo import DXO, DataKind, from_file
from nvflare.apis.fl_constant import ReservedKey
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import ReturnCode, make_reply
from nvflare.apis.signal import Signal
from nvflare.app_common.abstract.formatter import Formatter
from nvflare.app_common.abstract.model_locator import ModelLocator
from nvflare.app_common.app_constant import AppConstants
from nvflare.app_common.workflows import cross_site_model_eval as subject
from nvflare.widgets.info_collector import GroupInfoCollector, InfoCollector
from tests.timing_utils import isolate_time


@pytest.fixture
def runtime(tmp_path):
    ctl = subject.CrossSiteModelEval(cleanup_models=True)
    engine = Mock()
    engine.get_clients.return_value = [SimpleNamespace(name="site-1"), SimpleNamespace(name="site-2")]
    engine.get_workspace.return_value.get_run_dir.return_value = str(tmp_path)
    ctl._engine = engine
    ctl.fire_event = Mock()
    ctl.system_panic = Mock()
    ctl.broadcast = Mock()
    ctl.cancel_all_tasks = Mock()
    ctx = FLContext()
    ctx.set_prop(ReservedKey.ENGINE, engine, private=True, sticky=True)
    ctl.start_controller(ctx)
    return ctl, ctx, engine


@pytest.mark.parametrize(
    "field",
    [
        "cross_val_dir",
        "submit_model_timeout",
        "validation_timeout",
        "model_locator_id",
        "formatter_id",
        "submit_model_task_name",
        "validation_task_name",
        "cleanup_models",
        "participating_clients",
    ],
)
def test_model_evaluation_rejects_wrong_configuration_types(field):
    bad_value = "not-a-list" if field == "participating_clients" else []
    with pytest.raises(TypeError):
        subject.CrossSiteModelEval(**{field: bad_value})


@pytest.mark.parametrize("field", ["submit_model_timeout", "validation_timeout", "wait_for_clients_timeout"])
def test_model_evaluation_rejects_negative_deadlines(field):
    with pytest.raises(ValueError):
        subject.CrossSiteModelEval(**{field: -1})


def test_start_recreates_model_and_result_directories(runtime):
    ctl, ctx, _ = runtime
    models = Path(ctl._cross_val_models_dir)
    results = Path(ctl._cross_val_results_dir)
    (models / "old").write_text("stale")
    (results / "old").write_text("stale")
    ctl.start_controller(ctx)
    assert list(models.iterdir()) == [] and list(results.iterdir()) == []
    assert ctl._participating_clients == ["site-1", "site-2"]
    assert ctx.get_prop(AppConstants.CROSS_VAL_MODEL_PATH) == str(models)
    assert ctx.get_prop(AppConstants.CROSS_VAL_RESULTS_PATH) == str(results)


def test_client_model_and_metric_callbacks_persist_content_and_release_payloads(runtime):
    ctl, ctx, _ = runtime
    model = DXO(DataKind.WEIGHTS, {"w": [1, 2]}).to_shareable()
    client = SimpleNamespace(name="site-1")
    submitted = SimpleNamespace(client=client, result=model)
    ctl._receive_local_model_cb(submitted, ctx)
    assert submitted.result is None
    assert from_file(ctl._client_models["site-1"]).data == {"w": [1, 2]}
    task = ctl.broadcast.call_args.kwargs["task"]
    assert ctl.broadcast.call_args.kwargs["targets"] == ["site-1", "site-2"]
    assert task.props[AppConstants.MODEL_OWNER] == "site-1"
    client_task = SimpleNamespace(task=task, client=client)
    ctl._before_send_validate_task_cb(client_task, ctx)
    assert task.data.get_header(AppConstants.MODEL_OWNER) == "site-1"
    assert task.data.get_cookie(AppConstants.MODEL_OWNER) == "site-1"
    assert ctx.get_prop(AppConstants.MODEL_TO_VALIDATE) is task.data
    ctl._after_send_validate_task_cb(client_task, ctx)
    assert task.data is None
    metrics = DXO(DataKind.METRICS, {"accuracy": 0.75}).to_shareable()
    metrics.add_cookie(AppConstants.MODEL_OWNER, "site-1")
    client_task.result = metrics
    ctl._receive_val_result_cb(client_task, ctx)
    assert client_task.result is None
    assert from_file(ctl._val_results["site-1"]["site-1"]).data == {"accuracy": 0.75}


@pytest.mark.parametrize(
    "rc",
    [
        ReturnCode.MISSING_PEER_CONTEXT,
        ReturnCode.EXECUTION_EXCEPTION,
        ReturnCode.EXECUTION_RESULT_ERROR,
        ReturnCode.ERROR,
    ],
)
def test_bad_model_and_validation_results_do_not_save_payloads(runtime, rc):
    ctl, ctx, _ = runtime
    result = make_reply(rc)
    result.add_cookie(AppConstants.MODEL_OWNER, "bad-model")
    ctl._accept_local_model("site-1", result, ctx)
    ctl._accept_val_result("site-2", result, ctx)
    assert ctl._client_models["site-1"] is None
    assert ctl._val_results["site-2"]["bad-model"] == {}
    ctl.broadcast.assert_not_called()


@pytest.mark.parametrize("mode", ["valid", "wrong-type", "save-failure"])
def test_server_model_locator_results_are_validated_before_becoming_evaluation_tasks(runtime, monkeypatch, mode):
    ctl, ctx, _ = runtime
    ctl._model_locator = Mock(spec=ModelLocator)
    ctl._model_locator.get_model_names.return_value = ["global"]
    ctl._model_locator.locate_model.return_value = (
        object() if mode == "wrong-type" else DXO(DataKind.WEIGHTS, {"w": [3]})
    )
    if mode == "save-failure":
        monkeypatch.setattr(ctl, "_save_dxo_content", Mock(side_effect=OSError("disk full")))
    assert ctl._locate_server_models(ctx) is (mode == "valid")
    assert ctl.system_panic.called is (mode != "valid")
    if mode == "valid":
        assert from_file(ctl._server_models["SRV_global"]).data == {"w": [3]}
        assert ctl._val_results["SRV_global"] == {}


def test_control_flow_sends_submit_and_server_validation_tasks_with_bounded_wait(runtime, monkeypatch):
    ctl, ctx, _ = runtime
    ctl._model_locator = Mock(spec=ModelLocator)
    ctl._model_locator.get_model_names.return_value = ["global"]
    ctl._model_locator.locate_model.return_value = DXO(DataKind.WEIGHTS, {"w": [3]})
    ctl.get_num_standing_tasks = Mock(side_effect=[1, 0])
    clock = isolate_time(monkeypatch, subject)
    clock.sleep = Mock()
    ctl.control_flow(Signal(), ctx)
    assert [call.kwargs["task"].name for call in ctl.broadcast.call_args_list] == [
        AppConstants.TASK_SUBMIT_MODEL,
        AppConstants.TASK_VALIDATION,
    ]
    clock.sleep.assert_called_once_with(ctl._task_check_period)
    ctl.system_panic.assert_not_called()


@pytest.mark.parametrize("mode", ["timeout", "abort", "exception"])
def test_control_flow_terminates_if_no_clients_abort_or_scheduling_failure(runtime, monkeypatch, mode):
    ctl, ctx, engine = runtime
    signal = Signal()
    clock = isolate_time(monkeypatch, subject)
    clock.sleep = Mock()
    if mode == "timeout":
        ctl._participating_clients = []
        engine.get_clients.return_value = []
        clock.time = Mock(side_effect=[0, 500])
    elif mode == "abort":
        signal.trigger(True)
    else:
        ctl.broadcast.side_effect = RuntimeError("cannot broadcast")
    ctl.control_flow(signal, ctx)
    assert ctl.system_panic.called is (mode == "exception")
    if mode == "timeout":
        ctl.broadcast.assert_not_called()


def test_stop_removes_saved_models_but_preserves_validation_results(runtime):
    ctl, ctx, _ = runtime
    model = DXO(DataKind.WEIGHTS, {"w": [1]})
    path = ctl._save_dxo_content("model", ctl._cross_val_models_dir, model, ctx)
    metrics = ctl._save_dxo_content("metrics", ctl._cross_val_results_dir, DXO(DataKind.METRICS, {"accuracy": 1}), ctx)
    ctl._server_models["global"] = path
    ctl._client_models["site-1"] = path
    ctl.stop_controller(ctx)
    assert not Path(path).exists() and Path(metrics).exists()
    ctl.cancel_all_tasks.assert_called_once_with(fl_ctx=ctx)


@pytest.mark.parametrize("mode", ["missing-file", "empty"])
def test_before_send_rejects_missing_validation_model_without_setting_task_data(runtime, monkeypatch, mode):
    ctl, ctx, _ = runtime
    task = SimpleNamespace(props={AppConstants.MODEL_OWNER: "missing"}, data=None)
    if mode == "empty":
        monkeypatch.setattr(ctl, "_load_validation_content", Mock(return_value=None))
    ctl._before_send_validate_task_cb(SimpleNamespace(task=task), ctx)
    assert task.data is None
    ctl.system_panic.assert_called_once()


def test_stats_export_requires_collector_contract_and_uses_formatter(runtime):
    ctl, ctx, _ = runtime
    ctl._formatter = Mock(spec=Formatter)
    ctl._formatter.format.return_value = "accuracy table"
    collector = Mock(spec=GroupInfoCollector)
    ctx.set_prop(InfoCollector.CTX_KEY_STATS_COLLECTOR, collector, private=True, sticky=False)
    ctl.handle_event(InfoCollector.EVENT_TYPE_GET_STATS, ctx)
    assert collector.add_info.call_args.kwargs["info"] == {"val_results": "accuracy table"}
    assert ctx.get_prop(AppConstants.VALIDATION_RESULT) is ctl._val_results
    ctx.set_prop(InfoCollector.CTX_KEY_STATS_COLLECTOR, object(), private=True, sticky=False)
    with pytest.raises(TypeError, match="GroupInfoCollector"):
        ctl.handle_event(InfoCollector.EVENT_TYPE_GET_STATS, ctx)
