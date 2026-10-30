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

from nvflare.apis.fl_constant import ReservedKey, ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable, make_reply
from nvflare.apis.signal import Signal
from nvflare.app_opt.xgboost.histogram_based_v2 import controller as subject
from nvflare.app_opt.xgboost.histogram_based_v2.adaptors.xgb_adaptor import XGBServerAdaptor
from nvflare.app_opt.xgboost.histogram_based_v2.defs import Constant


@pytest.fixture
def controller(monkeypatch):
    controller = subject.XGBController("adaptor", 2, 0, False, {"objective": "reg:squarederror"})
    engine = MagicMock()
    engine.get_clients.return_value = [SimpleNamespace(name="site-b"), SimpleNamespace(name="site-a")]
    adaptor = MagicMock(spec=XGBServerAdaptor)
    engine.get_component.return_value = adaptor
    controller._engine = engine
    controller.system_panic = MagicMock()
    ctx = FLContext()
    ctx.set_prop(ReservedKey.ENGINE, engine)
    peer = FLContext()
    peer.set_prop(ReservedKey.IDENTITY_NAME, "site-a")
    ctx.set_peer_context(peer)
    monkeypatch.setattr(subject.ReliableMessage, "register_request_handler", MagicMock())
    controller.start_controller(ctx)
    controller.abort_signal = Signal()
    return controller, ctx, adaptor


def test_controller_start_registers_protocol_handlers_and_stop_releases_adaptor(controller):
    controller, ctx, adaptor = controller
    adaptor.initialize.assert_called_once_with(ctx)
    assert set(controller.client_statuses) == {"site-a", "site-b"}
    topics = {c.kwargs["topic"] for c in subject.ReliableMessage.register_request_handler.call_args_list}
    assert topics == {Constant.TOPIC_XGB_REQUEST, Constant.TOPIC_CLIENT_DONE}
    controller.stop_controller(ctx)
    adaptor.stop.assert_called_once_with(ctx)


@pytest.mark.parametrize("bad_adaptor", [None, object()])
def test_controller_rejects_missing_or_wrong_adaptor_before_registering(controller, bad_adaptor):
    controller, ctx, _ = controller
    controller._engine.get_component.return_value = bad_adaptor
    controller.start_controller(ctx)
    controller.system_panic.assert_called_once()


@pytest.mark.parametrize(
    "ranks",
    [
        {"site-a": 0},
        {"site-a": 0, "extra": 1},
        {"site-a": "0", "site-b": 1},
        {"site-a": -1, "site-b": 1},
        {"site-a": 0, "site-b": 0},
    ],
)
def test_configure_rejects_non_consecutive_or_incomplete_rank_assignments(controller, ranks):
    controller, ctx, _ = controller
    controller.client_ranks = ranks
    controller.broadcast_and_wait = MagicMock()
    assert controller._configure_clients(Signal(), ctx) is False
    controller.system_panic.assert_called_once()
    controller.broadcast_and_wait.assert_not_called()


@pytest.mark.parametrize("configured", [False, True])
def test_configuration_distributes_sorted_ranks_and_requires_every_reply(controller, configured):
    controller, ctx, _ = controller
    tasks = []

    def broadcast(**kwargs):
        task = kwargs["task"]
        tasks.append(task)
        assert kwargs["min_responses"] == 2
        if configured:
            for name in controller.participating_clients:
                task.result_received_cb(
                    SimpleNamespace(client=SimpleNamespace(name=name), result=make_reply(ReturnCode.OK)), ctx
                )

    controller.broadcast_and_wait = broadcast
    assert controller._configure_clients(Signal(), ctx) is configured
    config = tasks[0].data
    assert config[Constant.CONF_KEY_CLIENT_RANKS] == {"site-a": 0, "site-b": 1}
    assert config[Constant.CONF_KEY_NUM_ROUNDS] == 2
    assert config[Constant.CONF_KEY_XGB_PARAMS] == {"objective": "reg:squarederror"}
    assert config[Constant.CONF_KEY_SECURE_TRAINING] is False
    assert controller.system_panic.call_count == (0 if configured else 1)


@pytest.mark.parametrize("started", [False, True])
def test_start_requires_successful_reply_from_every_client(controller, started):
    controller, ctx, _ = controller

    def broadcast(**kwargs):
        for name in controller.participating_clients:
            kwargs["task"].result_received_cb(
                SimpleNamespace(
                    client=SimpleNamespace(name=name),
                    result=make_reply(ReturnCode.OK if started else ReturnCode.EXECUTION_EXCEPTION),
                ),
                ctx,
            )

    controller.broadcast_and_wait = broadcast
    assert controller._start_clients(Signal(), ctx) is started
    assert all(bool(s.started_time) is started for s in controller.client_statuses.values())


@pytest.mark.parametrize(
    "operation,method",
    [
        (Constant.OP_ALL_GATHER, "all_gather"),
        (Constant.OP_ALL_GATHER_V, "all_gather_v"),
        (Constant.OP_ALL_REDUCE, "all_reduce"),
        (Constant.OP_BROADCAST, "broadcast"),
    ],
)
def test_collective_dispatch_preserves_protocol_parameters_and_event_transforms(controller, operation, method):
    controller, ctx, adaptor = controller
    getattr(adaptor, method).return_value = b"aggregate"
    events = []

    def event(event_type, context):
        events.append(event_type)
        if event_type in (Constant.EVENT_BEFORE_BROADCAST, Constant.EVENT_BEFORE_ALL_GATHER_V):
            context.set_prop(Constant.PARAM_KEY_SEND_BUF, b"secured-input", sticky=False)
        else:
            context.set_prop(Constant.PARAM_KEY_RCV_BUF, b"processed-result", sticky=False)

    controller.fire_event = event
    request = Shareable(
        {
            Constant.PARAM_KEY_RANK: 1,
            Constant.PARAM_KEY_SEQ: 3,
            Constant.PARAM_KEY_SEND_BUF: b"input",
            Constant.PARAM_KEY_ROOT: 0,
            Constant.PARAM_KEY_DATA_TYPE: 2,
            Constant.PARAM_KEY_REDUCE_OP: 1,
        }
    )
    request.set_header(Constant.MSG_KEY_XGB_OP, operation)
    reply = controller._process_xgb_request("topic", request, ctx)
    assert reply.get_header(Constant.MSG_KEY_XGB_OP) == operation
    expected = b"processed-result" if method in ("broadcast", "all_gather_v") else b"aggregate"
    assert reply[Constant.PARAM_KEY_RCV_BUF] == expected
    args = getattr(adaptor, method).call_args.args
    assert args[:2] == (1, 3) and args[-1] is ctx
    assert controller.client_statuses["site-a"].last_op == operation
    assert len(events) == (2 if method in ("broadcast", "all_gather_v") else 0)


@pytest.mark.parametrize("operation", [None, "unknown"])
def test_bad_collective_request_aborts_protocol(controller, operation):
    controller, ctx, _ = controller
    request = Shareable()
    if operation:
        request.set_header(Constant.MSG_KEY_XGB_OP, operation)
    reply = controller._process_xgb_request("topic", request, ctx)
    assert reply.get_return_code() == ReturnCode.BAD_REQUEST_DATA
    assert controller.abort_signal.triggered
    controller.system_panic.assert_called_once()
    assert controller._process_xgb_request("topic", request, ctx).get_return_code() == ReturnCode.SERVICE_UNAVAILABLE


def test_collective_exception_returns_execution_error_and_triggers_stop(controller):
    controller, ctx, adaptor = controller
    adaptor.all_gather.side_effect = ValueError("bad buffer")
    request = Shareable()
    request.set_header(Constant.MSG_KEY_XGB_OP, Constant.OP_ALL_GATHER)
    assert controller._process_xgb_request("topic", request, ctx).get_return_code() == ReturnCode.EXECUTION_EXCEPTION
    assert controller.abort_signal.triggered


@pytest.mark.parametrize(
    "exit_code,panic",
    [(0, False), (Constant.EXIT_CODE_CANT_START, True), (Constant.EXIT_CODE_JOB_ABORT, True), (9, False)],
)
def test_client_completion_marks_done_and_distinguishes_fatal_exit_codes(controller, exit_code, panic):
    controller, ctx, _ = controller
    reply = controller._process_client_done("done", Shareable({Constant.MSG_KEY_EXIT_CODE: exit_code}), ctx)
    assert reply.get_return_code() == ReturnCode.OK
    assert controller.client_statuses["site-a"].xgb_done
    assert bool(controller.system_panic.call_count) is panic


def test_control_flow_configures_bridge_before_starting_clients_and_monitors_completion(controller, monkeypatch):
    controller, ctx, adaptor = controller
    order = []
    controller._configure_clients = MagicMock(side_effect=lambda *a: order.append("configure-clients") or True)
    adaptor.start.side_effect = lambda *a: order.append("start-server")
    controller._start_clients = MagicMock(side_effect=lambda *a: order.append("start-clients") or True)
    controller._check_job_status = MagicMock(side_effect=[False, True])
    sleep = MagicMock()
    monkeypatch.setattr(subject, "time", SimpleNamespace(time=lambda: 1, sleep=sleep))
    controller.control_flow(Signal(), ctx)
    assert order == ["configure-clients", "start-server", "start-clients"]
    adaptor.configure.assert_called_once_with({Constant.CONF_KEY_WORLD_SIZE: 2}, ctx)
    sleep.assert_called_once_with(controller.job_status_check_interval)
    controller.system_panic.assert_not_called()


@pytest.mark.parametrize("stage", ["configure", "adaptor", "clients"])
def test_control_flow_aborts_at_failed_startup_boundary(controller, stage):
    controller, ctx, adaptor = controller
    controller._configure_clients = MagicMock(return_value=stage != "configure")
    controller._start_clients = MagicMock(return_value=stage != "clients")
    if stage == "adaptor":
        adaptor.start.side_effect = RuntimeError("startup")
    controller.control_flow(Signal(), ctx)
    controller.system_panic.assert_called_once()
    if stage != "clients":
        controller._start_clients.assert_not_called()


@pytest.mark.parametrize("rc", [0, 2])
def test_server_stop_always_sets_abort_and_panics_only_for_error(controller, rc):
    controller, ctx, _ = controller
    controller._xgb_server_stopped(rc, ctx)
    assert controller.abort_signal.triggered
    assert controller.system_panic.call_count == (0 if rc == 0 else 1)
