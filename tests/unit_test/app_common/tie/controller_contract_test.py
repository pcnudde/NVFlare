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

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from nvflare.apis.fl_constant import ReservedKey
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import ReturnCode, Shareable, make_reply
from nvflare.apis.signal import Signal
from nvflare.app_common.tie import controller as subject
from nvflare.app_common.tie.applet import Applet
from nvflare.app_common.tie.connector import Connector
from nvflare.app_common.tie.defs import Constant
from tests.timing_utils import isolate_time


class ConcreteController(subject.TieController):
    def get_client_config_params(self, fl_ctx):
        return {"clients": 2}

    def get_connector_config_params(self, fl_ctx):
        return {"mode": "server"}

    def get_connector(self, fl_ctx):
        return self.test_connector

    def get_applet(self, fl_ctx):
        return self.test_applet


@pytest.fixture
def runtime(monkeypatch):
    ctl = ConcreteController()
    ctl.test_connector = Mock(spec=Connector)
    ctl.test_applet = Mock(spec=Applet)
    ctl.system_panic = Mock()
    engine = Mock()
    engine.get_clients.return_value = [SimpleNamespace(name="site-1"), SimpleNamespace(name="site-2")]
    ctl._engine = engine
    ctx = FLContext()
    ctx.set_prop(ReservedKey.ENGINE, engine, private=True, sticky=True)
    peer = FLContext()
    peer.set_prop(ReservedKey.IDENTITY_NAME, "site-1", private=False, sticky=True)
    ctx.set_peer_context(peer)
    monkeypatch.setattr(subject.ReliableMessage, "register_request_handler", Mock())
    return ctl, ctx, engine


@pytest.mark.parametrize(
    "field,value",
    [
        ("configure_task_timeout", 0),
        ("start_task_timeout", 0),
        ("job_status_check_interval", 0),
        ("max_client_op_interval", 9),
        ("progress_timeout", 4),
    ],
)
def test_invalid_monitoring_and_task_deadlines_are_rejected(field, value):
    with pytest.raises(ValueError):
        ConcreteController(**{field: value})


def test_start_initializes_components_and_registers_request_boundaries(runtime):
    ctl, ctx, engine = runtime
    ctl.start_controller(ctx)
    assert ctl.participating_clients == ["site-1", "site-2"]
    assert set(ctl.client_statuses) == {"site-1", "site-2"}
    ctl.test_applet.initialize.assert_called_once_with(ctx)
    ctl.test_connector.set_applet.assert_called_once_with(ctl.test_applet)
    ctl.test_connector.initialize.assert_called_once_with(ctx)
    assert ctl.connector is ctl.test_connector
    engine.register_aux_message_handler.assert_called_once_with(
        topic=Constant.TOPIC_CLIENT_DONE, message_handle_func=ctl._process_client_done
    )
    subject.ReliableMessage.register_request_handler.assert_called_once_with(
        topic=Constant.TOPIC_APP_REQUEST, handler_f=ctl._handle_app_request, fl_ctx=ctx
    )


@pytest.mark.parametrize(
    "bad_component,value,reason",
    [
        ("test_connector", None, "cannot get connector"),
        ("test_connector", object(), "invalid connector"),
        ("test_applet", None, "cannot get applet"),
        ("test_applet", object(), "invalid applet"),
    ],
)
def test_start_rejects_missing_or_wrong_components_before_registration(runtime, bad_component, value, reason):
    ctl, ctx, engine = runtime
    setattr(ctl, bad_component, value)
    ctl.start_controller(ctx)
    assert reason in ctl.system_panic.call_args.args[0]
    assert ctl.connector is None
    engine.register_aux_message_handler.assert_not_called()


@pytest.mark.parametrize(
    "method,field", [("_configure_clients", "configured_time"), ("_start_clients", "started_time")]
)
@pytest.mark.parametrize("responds", [False, True])
def test_configure_and_start_require_success_from_every_participant(runtime, method, field, responds):
    ctl, ctx, _ = runtime
    ctl.start_controller(ctx)
    signal = Signal()

    def broadcast(**kwargs):
        assert kwargs["targets"] == ["site-1", "site-2"]
        assert kwargs["min_responses"] == 2
        assert kwargs["abort_signal"] is signal
        task = kwargs["task"]
        if method == "_configure_clients":
            assert task.data[Constant.MSG_KEY_CONFIG] == {"clients": 2}
        for name in kwargs["targets"]:
            reply = make_reply(ReturnCode.OK if responds else ReturnCode.ERROR)
            task.result_received_cb(SimpleNamespace(client=SimpleNamespace(name=name), result=reply), ctx)

    ctl.broadcast_and_wait = Mock(side_effect=broadcast)
    assert getattr(ctl, method)(signal, ctx) is responds
    assert all(bool(getattr(status, field)) is responds for status in ctl.client_statuses.values())
    assert ctl.system_panic.called is not responds


@pytest.mark.parametrize("value", [None, RuntimeError("config failed")])
def test_configuration_failure_never_schedules_client_tasks(runtime, value):
    ctl, ctx, _ = runtime
    ctl.start_controller(ctx)
    ctl.get_client_config_params = Mock(side_effect=value if isinstance(value, Exception) else None, return_value=value)
    ctl.broadcast_and_wait = Mock()
    assert ctl._configure_clients(Signal(), ctx) is False
    ctl.broadcast_and_wait.assert_not_called()
    ctl.system_panic.assert_called_once()


@pytest.mark.parametrize("rc", [0, 7, Constant.EXIT_CODE_CANT_START])
def test_client_done_updates_status_and_only_start_failure_panics(runtime, rc):
    ctl, ctx, _ = runtime
    ctl.start_controller(ctx)
    reply = ctl._process_client_done("done", Shareable({Constant.MSG_KEY_EXIT_CODE: rc}), ctx)
    assert reply.get_return_code() == ReturnCode.OK
    assert ctl.client_statuses["site-1"].app_done
    assert ctl.system_panic.called is (rc == Constant.EXIT_CODE_CANT_START)


@pytest.mark.parametrize("peer", [None, "wrong-type", FLContext()])
def test_status_updates_ignore_requests_without_named_peer(runtime, peer):
    ctl, ctx, _ = runtime
    ctl.start_controller(ctx)
    ctx.set_peer_context(peer)
    ctl.log_error = Mock()
    ctl._update_client_status(ctx, op="train", client_done=True)
    assert not ctl.client_statuses["site-1"].app_done
    ctl.log_error.assert_called_once()


@pytest.mark.parametrize("outcome", ["ok", "exception", "aborted"])
def test_app_request_echoes_operation_or_stops_on_processing_failure(runtime, outcome):
    ctl, ctx, _ = runtime
    ctl.start_controller(ctx)
    ctl.abort_signal = Signal()
    req = Shareable()
    req.set_header(Constant.MSG_KEY_OP, "fit")
    ctl.connector.process_app_request.return_value = Shareable({"answer": 42})
    if outcome == "exception":
        ctl.connector.process_app_request.side_effect = RuntimeError("broken")
    elif outcome == "aborted":
        ctl.abort_signal.trigger(True)
    reply = ctl._handle_app_request("request", req, ctx)
    if outcome == "ok":
        assert reply["answer"] == 42
        assert reply.get_header(Constant.MSG_KEY_OP) == "fit"
        assert ctl.client_statuses["site-1"].last_op == "fit"
    elif outcome == "exception":
        assert reply.get_return_code() == ReturnCode.EXECUTION_EXCEPTION
        assert ctl.abort_signal.triggered
        ctl.system_panic.assert_called_once()
    else:
        assert reply.get_return_code() == ReturnCode.SERVICE_UNAVAILABLE
        ctl.connector.process_app_request.assert_not_called()


@pytest.mark.parametrize(
    "done,age,progress,expected,error",
    [
        (True, 0, 10, True, False),
        (False, 0, 10, False, False),
        (False, 101, 200, True, True),
        (False, 20, 10, True, True),
    ],
)
def test_health_check_distinguishes_completion_activity_and_progress_deadlines(
    runtime, monkeypatch, done, age, progress, expected, error
):
    ctl, ctx, _ = runtime
    ctl.start_controller(ctx)
    clock = isolate_time(monkeypatch, subject)
    clock.time = lambda: 1000
    ctl.max_client_op_interval = 100
    ctl.progress_timeout = progress
    for status in ctl.client_statuses.values():
        status.app_done = done
        status.last_op_time = 1000 - age
    assert ctl._check_job_status(ctx) is expected
    assert ctl.system_panic.called is error


@pytest.mark.parametrize("failure", [None, "configure", "connector", "start"])
def test_control_flow_orders_configuration_start_monitor_and_shutdown(runtime, failure):
    ctl, ctx, _ = runtime
    ctl.start_controller(ctx)
    ctl._configure_clients = Mock(return_value=failure != "configure")
    ctl._start_clients = Mock(return_value=failure != "start")
    ctl._check_job_status = Mock(return_value=True)
    if failure == "connector":
        ctl.connector.configure.side_effect = RuntimeError("cannot configure")
    abort = Signal()
    ctl.control_flow(abort, ctx)
    ctl.connector.set_abort_signal.assert_called_once_with(abort)
    if failure:
        assert abort.triggered
        ctl.system_panic.assert_called()
    else:
        assert not abort.triggered
        ctl.connector.configure.assert_called_once_with({"mode": "server"}, ctx)
        ctl.connector.start.assert_called_once_with(ctx)
        ctl.connector.monitor.assert_called_once_with(ctx, ctl._app_stopped)
        ctl.connector.stop.assert_called_once_with(ctx)


@pytest.mark.parametrize("rc", [0, 2])
def test_app_completion_signals_abort_and_preserves_abnormal_status(runtime, rc):
    ctl, ctx, _ = runtime
    ctl.abort_signal = Signal()
    ctl._app_stopped(rc, ctx)
    assert ctl.abort_signal.triggered
    assert ctl.system_panic.called is (rc != 0)


def test_stop_controller_stops_present_connector_and_unknown_results_are_ignored(runtime):
    ctl, ctx, _ = runtime
    ctl.start_controller(ctx)
    ctl.stop_controller(ctx)
    ctl.connector.stop.assert_called_once_with(ctx)
    ctl.log_warning = Mock()
    ctl.process_result_of_unknown_task(SimpleNamespace(name="site-1"), "unknown", "task", Shareable(), ctx)
    assert "ignored unknown task" in ctl.log_warning.call_args.args[1]
