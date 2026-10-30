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

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

pytest.importorskip("xgboost")

from nvflare.apis.fl_constant import FLContextKey, ReservedKey, ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.apis.signal import Signal
from nvflare.app_opt.xgboost.data_loader import XGBDataLoader
from nvflare.app_opt.xgboost.histogram_based import controller as ctrl_module
from nvflare.app_opt.xgboost.histogram_based import executor as exec_module
from nvflare.app_opt.xgboost.histogram_based.constants import XGB_TRAIN_TASK, XGBShareableHeader


@pytest.fixture
def context(tmp_path):
    ctx = FLContext()
    engine = MagicMock()
    engine.get_clients.return_value = [SimpleNamespace(name="site-b"), SimpleNamespace(name="site-a")]
    workspace = MagicMock()
    workspace.get_startup_kit_dir.return_value = str(tmp_path)
    workspace.get_run_dir.return_value = str(tmp_path)
    ctx.set_prop(ReservedKey.ENGINE, engine)
    ctx.set_prop(ReservedKey.IDENTITY_NAME, "site-a")
    ctx.set_prop(FLContextKey.WORKSPACE_OBJECT, workspace)
    ctx.set_prop(FLContextKey.CURRENT_RUN, "job")
    return ctx


@pytest.mark.parametrize("secure", [False, True])
def test_legacy_controller_starts_correct_process_and_broadcasts_rank_configuration(
    monkeypatch, tmp_path, context, secure
):
    process = MagicMock()
    factory = MagicMock(return_value=process)
    monkeypatch.setattr(ctrl_module, "multiprocessing", SimpleNamespace(Process=factory))
    federated = SimpleNamespace(run_federated_server=MagicMock())
    monkeypatch.setattr(ctrl_module, "optional_import", lambda **kwargs: (federated, True))
    monkeypatch.setattr(ctrl_module, "get_open_ports", lambda n: [12345])
    for name in ("server.crt", "server.key", "rootCA.pem"):
        (tmp_path / name).write_text("certificate")
    controller = ctrl_module.XGBFedController()
    controller._engine = context.get_engine()
    controller._engine.server.secure_train = secure
    controller.system_panic = MagicMock()
    controller.broadcast_and_wait = MagicMock()
    controller.start_controller(context)
    assert controller._rank_map == {"site-a": 0, "site-b": 1}
    assert factory.call_args.kwargs["target"] is federated.run_federated_server
    args = factory.call_args.kwargs["args"]
    assert args[:2] == (2, 12345)
    assert len(args) == (5 if secure else 2)
    process.start.assert_called_once_with()
    controller.control_flow(Signal(), context)
    kwargs = controller.broadcast_and_wait.call_args.kwargs
    assert kwargs["min_responses"] == 2
    data = kwargs["task"].data
    assert data.get_header(XGBShareableHeader.RANK_MAP) == {"site-a": 0, "site-b": 1}
    assert data.get_header(XGBShareableHeader.XGB_FL_SERVER_SECURE) is secure
    controller.stop_controller(context)
    process.terminate.assert_called_once_with()
    assert controller._started is False


def test_controller_fails_closed_when_runtime_or_certificates_are_missing(monkeypatch, context):
    controller = ctrl_module.XGBFedController(port=12345)
    controller._engine = context.get_engine()
    controller._engine.server.secure_train = True
    controller.system_panic = MagicMock()
    monkeypatch.setattr(ctrl_module, "optional_import", lambda **kwargs: (None, False))
    controller.start_controller(context)
    assert not controller._started
    monkeypatch.setattr(ctrl_module, "optional_import", lambda **kwargs: (object(), True))
    controller.start_controller(context)
    assert not controller._started
    controller.control_flow(Signal(), context)
    controller.system_panic.assert_called_once()
    with pytest.raises(TypeError):
        ctrl_module.XGBFedController(train_timeout="bad")


@pytest.mark.parametrize(
    "files,expected",
    [
        ([], False),
        (["server.crt"], False),
        (["server.crt", "server.key"], False),
        (["server.crt", "server.key", "rootCA.pem"], True),
    ],
)
def test_controller_secure_mode_requires_every_certificate(tmp_path, context, files, expected):
    for name in files:
        (tmp_path / name).write_text("certificate")
    controller = ctrl_module.XGBFedController(port=1)
    assert controller._get_certificates(context) is expected
    if expected:
        assert controller._server_key_path == str(tmp_path / "server.key")


@pytest.fixture
def executor(context):
    executor = exec_module.FedXGBHistogramExecutor(2, 1, {"objective": "reg:squarederror"}, "loader", use_gpus=True)
    executor.system_panic = MagicMock()
    loader = MagicMock(spec=XGBDataLoader)
    loader.load_data.return_value = (object(), object())
    context.get_engine().get_component.return_value = loader
    executor.initialize(context)
    return executor


def _train_request():
    request = Shareable()
    for key, value in {
        XGBShareableHeader.RANK_MAP: {"site-a": 0},
        XGBShareableHeader.WORLD_SIZE: 1,
        XGBShareableHeader.XGB_FL_SERVER_PORT: 1234,
        XGBShareableHeader.XGB_FL_SERVER_SECURE: False,
    }.items():
        request.set_header(key, value)
    return request


def test_executor_marshals_federated_context_caches_data_and_saves_model(monkeypatch, tmp_path, context, executor):
    communicator = MagicMock()
    monkeypatch.setattr(exec_module.xgb.collective, "CommunicatorContext", communicator)
    printer = MagicMock()
    monkeypatch.setattr(exec_module.xgb.collective, "communicator_print", printer)
    booster = MagicMock()
    train = MagicMock(return_value=booster)
    monkeypatch.setattr(exec_module.xgb, "train", train)
    request = _train_request()
    for _ in range(2):
        assert executor.execute(XGB_TRAIN_TASK, request, context, Signal()).get_return_code() == ReturnCode.OK
    assert communicator.call_args.kwargs == {
        "dmlc_communicator": "federated",
        "federated_server_address": "localhost:1234",
        "federated_world_size": 1,
        "federated_rank": 0,
    }
    executor.data_loader.load_data.assert_called_once_with()
    assert executor.xgb_params["device"] == "cuda:0"
    assert train.call_args.args[0]["objective"] == "reg:squarederror"
    assert train.call_args.args[2] == 2
    assert len(train.call_args.kwargs["evals"]) == 2 and len(train.call_args.kwargs["callbacks"]) == 1
    assert booster.save_model.call_args.args[0] == str(tmp_path / "test.model.json")
    assert printer.call_count == 2
    train.side_effect = RuntimeError("booster")
    assert (
        executor.execute(XGB_TRAIN_TASK, request, context, Signal()).get_return_code() == ReturnCode.EXECUTION_EXCEPTION
    )


@pytest.mark.parametrize(
    "missing",
    [
        XGBShareableHeader.WORLD_SIZE,
        XGBShareableHeader.XGB_FL_SERVER_PORT,
        XGBShareableHeader.XGB_FL_SERVER_SECURE,
        XGBShareableHeader.RANK_MAP,
    ],
)
def test_executor_refuses_incomplete_federated_headers_before_data_loading(context, executor, missing):
    request = _train_request()
    request.set_header(missing, {} if missing == XGBShareableHeader.RANK_MAP else None)
    assert executor.execute(XGB_TRAIN_TASK, request, context, Signal()).get_return_code() == ReturnCode.ERROR
    executor.data_loader.load_data.assert_not_called()


def test_executor_abort_unknown_task_and_secure_certificates(context, executor, tmp_path):
    signal = Signal()
    signal.trigger(True)
    assert (
        executor.execute(XGB_TRAIN_TASK, _train_request(), context, signal).get_return_code() == ReturnCode.TASK_ABORTED
    )
    assert executor.execute("unknown", Shareable(), context, Signal()).get_return_code() == ReturnCode.TASK_UNKNOWN
    for name in ("rootCA.pem", "client.key", "client.crt"):
        assert executor._get_certificates(context) is False
        (tmp_path / name).write_text("certificate")
    assert executor._get_certificates(context) is True
    assert executor._client_cert_path == str(tmp_path / "client.crt")
