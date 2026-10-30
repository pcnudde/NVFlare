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

from nvflare.apis.event_type import EventType
from nvflare.apis.fl_constant import ReservedKey
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import ReturnCode, Shareable, make_reply
from nvflare.apis.signal import Signal
from nvflare.app_common.tie import connector as connector_module
from nvflare.app_common.tie.applet import Applet
from nvflare.app_common.tie.defs import Constant
from nvflare.app_common.tie.executor import TieExecutor


class Connector(connector_module.Connector):
    def start(self, fl_ctx):
        pass

    def stop(self, fl_ctx):
        pass

    def configure(self, config, fl_ctx):
        pass


class Executor(TieExecutor):
    def get_connector(self, fl_ctx):
        return self.test_connector

    def get_applet(self, fl_ctx):
        return self.test_applet


def context():
    ctx = FLContext()
    ctx.set_prop(ReservedKey.ENGINE, Mock(), private=True, sticky=True)
    return ctx


def test_connector_applet_controls_validate_inputs_and_inject_fl_context():
    connector = Connector()
    ctx = context()
    with pytest.raises(TypeError, match="Applet"):
        connector.set_applet(object())
    with pytest.raises(TypeError, match="Signal"):
        connector.set_abort_signal(None)
    with pytest.raises(RuntimeError, match="applet has not been set"):
        connector.start_applet({}, ctx)
    assert connector.is_applet_stopped() == (True, 0)
    applet = Mock(spec=Applet)
    applet.stop.return_value = -15
    applet.is_stopped.return_value = (False, 0)
    connector.set_applet(applet)
    connector.initialize(ctx)
    signal = Signal()
    connector.set_abort_signal(signal)
    app_ctx = {"epoch": 3}
    connector.start_applet(app_ctx, ctx)
    assert app_ctx[Constant.APP_CTX_FL_CONTEXT] is ctx
    applet.start.assert_called_once_with(app_ctx)
    assert connector.stop_applet(7) == -15
    applet.stop.assert_called_once_with(7)
    assert connector._is_stopped() == (False, 0)
    assert connector.engine is ctx.get_engine() and connector.abort_signal is signal


@pytest.mark.parametrize("aborted", [False, True])
def test_connector_monitor_callback_only_reports_observed_completion(aborted):
    connector = Connector()
    connector.set_abort_signal(Signal())
    if aborted:
        connector.abort_signal.trigger(True)
    connector._is_stopped = Mock(return_value=(True, 4))
    callback = Mock()
    ctx = FLContext()
    connector._monitor(ctx, callback)
    assert callback.called is not aborted
    if not aborted:
        callback.assert_called_once_with(4, ctx)


def test_connector_monitor_and_request_use_controlled_launch_and_transport_boundaries(monkeypatch):
    connector = Connector()
    connector.initialize(context())
    connector.set_abort_signal(Signal())
    thread = Mock()
    factory = Mock(return_value=thread)
    monkeypatch.setattr(connector_module, "threading", SimpleNamespace(Thread=factory))
    with pytest.raises(RuntimeError, match="callable"):
        connector.monitor(FLContext(), None)
    callback = Mock()
    ctx = FLContext()
    connector.monitor(ctx, callback)
    assert factory.call_args.kwargs["args"] == (ctx, callback)
    thread.start.assert_called_once()
    send = Mock(return_value=make_reply(ReturnCode.OK))
    monkeypatch.setattr(connector_module.ReliableMessage, "send_request", send)
    request = Shareable({"value": 7})
    connector.send_request(None, "fit", request, 2, 10, None)
    call = send.call_args.kwargs
    assert request.get_header(Constant.MSG_KEY_OP) == "fit"
    assert call["target"] == connector_module.FQCN.ROOT_SERVER
    assert call["topic"] == Constant.TOPIC_APP_REQUEST
    assert call["fl_ctx"] is connector.engine.new_context.return_value
    assert call["per_msg_timeout"] == 2 and call["tx_timeout"] == 10
    assert call["abort_signal"] is connector.abort_signal


@pytest.fixture
def runtime():
    executor = Executor()
    executor.test_connector = Mock(spec=connector_module.Connector)
    executor.test_applet = Mock(spec=Applet)
    executor.system_panic = Mock()
    return executor, context()


def test_executor_start_initializes_applet_and_connector_then_end_signals_stop(runtime):
    executor, ctx = runtime
    executor.handle_event(EventType.START_RUN, ctx)
    executor.test_applet.initialize.assert_called_once_with(ctx)
    executor.connector.set_abort_signal.assert_called_once_with(executor.abort_signal)
    executor.connector.set_applet.assert_called_once_with(executor.test_applet)
    executor.connector.initialize.assert_called_once_with(ctx)
    executor.handle_event(EventType.END_RUN, ctx)
    assert executor.abort_signal.triggered
    executor.connector.stop.assert_called_once_with(ctx)


@pytest.mark.parametrize(
    "field,value,reason",
    [
        ("test_connector", None, "cannot get connector"),
        ("test_connector", object(), "invalid connector"),
        ("test_applet", None, "cannot get applet"),
        ("test_applet", object(), "invalid applet"),
    ],
)
def test_executor_start_requires_typed_applet_and_connector(runtime, field, value, reason):
    executor, ctx = runtime
    setattr(executor, field, value)
    executor.handle_event(EventType.START_RUN, ctx)
    assert reason in executor.system_panic.call_args.args[0]
    assert executor.connector is None


def test_executor_config_start_and_fatal_notification_preserve_protocol(runtime):
    executor, ctx = runtime
    executor.handle_event(EventType.START_RUN, ctx)
    executor.configure = Mock()
    assert (
        executor.execute(executor.configure_task_name, Shareable(), ctx, Signal()).get_return_code()
        == ReturnCode.BAD_TASK_DATA
    )
    executor.connector.configure.assert_not_called()
    request = Shareable({Constant.MSG_KEY_CONFIG: {"epochs": 3}})
    assert executor.execute(executor.configure_task_name, request, ctx, Signal()).get_return_code() == ReturnCode.OK
    executor.configure.assert_called_once_with({"epochs": 3}, ctx)
    executor.connector.configure.assert_called_once_with({}, ctx)
    assert executor.execute(executor.start_task_name, Shareable(), ctx, Signal()).get_return_code() == ReturnCode.OK
    executor.connector.monitor.assert_called_once_with(ctx, executor._notify_client_done)
    executor.connector.start.side_effect = RuntimeError("app failed")
    assert (
        executor.execute(executor.start_task_name, Shareable(), ctx, Signal()).get_return_code()
        == ReturnCode.EXECUTION_EXCEPTION
    )
    assert executor.execute("other", Shareable(), ctx, Signal()).get_return_code() == ReturnCode.TASK_UNSUPPORTED
    executor.handle_event(EventType.FATAL_SYSTEM_ERROR, ctx)
    args = ctx.get_engine().send_aux_request.call_args.kwargs
    assert args["request"][Constant.MSG_KEY_EXIT_CODE] == Constant.EXIT_CODE_FATAL_ERROR
    assert args["topic"] == Constant.TOPIC_CLIENT_DONE
    assert args["timeout"] == 0 and args["optional"]
    executor._notify_client_done(0, ctx)
    assert ctx.get_engine().send_aux_request.call_args.kwargs["request"][Constant.MSG_KEY_EXIT_CODE] == 0
