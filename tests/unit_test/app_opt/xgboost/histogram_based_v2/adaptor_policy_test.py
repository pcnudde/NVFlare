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
from xgboost.core import XGBoostError

from nvflare.apis.fl_constant import ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.apis.signal import Signal
from nvflare.apis.utils.reliable_message import PROP_KEY_DEBUG_INFO
from nvflare.app_opt.xgboost.histogram_based_v2.adaptors import adaptor as lifecycle
from nvflare.app_opt.xgboost.histogram_based_v2.adaptors import xgb_adaptor as protocol
from nvflare.app_opt.xgboost.histogram_based_v2.adaptors.grpc_client_adaptor import GrpcClientAdaptor
from nvflare.app_opt.xgboost.histogram_based_v2.defs import Constant
from nvflare.app_opt.xgboost.histogram_based_v2.runners.xgb_runner import AppRunner


@pytest.mark.parametrize("error", [None, ValueError("failure"), XGBoostError("training failed")])
@pytest.mark.parametrize("in_process", [False, True])
def test_runner_starter_records_completion_and_failure_without_starting_a_worker(monkeypatch, error, in_process):
    runner = MagicMock(spec=AppRunner)
    runner.run.side_effect = error
    configure = MagicMock()
    trace = MagicMock()
    monkeypatch.setattr(lifecycle, "configure_logging", configure)
    monkeypatch.setattr(lifecycle, "secure_log_traceback", trace)
    starter = lifecycle._RunnerStarter("XGB", runner, in_process, "workspace", "job")
    if error is not None and not in_process:
        with pytest.raises(SystemExit) as exc:
            starter.start({"rank": 2})
        assert exc.value.code == Constant.EXIT_CODE_CANT_START
    else:
        starter.start({"rank": 2})
    runner.run.assert_called_once_with({"rank": 2})
    assert starter.stopped is True
    assert starter.started is (error is None)
    assert starter.exit_code == (Constant.EXIT_CODE_CANT_START if error else 0)
    assert configure.call_count == int(not in_process)
    assert trace.call_count == int(error is not None and not isinstance(error, XGBoostError))


@pytest.mark.parametrize("in_process", [True, False])
def test_adaptor_launch_and_cleanup_use_owned_worker_boundaries(monkeypatch, in_process):
    class Worker:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.started = False
            self.killed = False
            self.exitcode = None

        def start(self):
            self.started = True

        def kill(self):
            self.killed = True

    worker_factory = MagicMock(side_effect=lambda **kwargs: Worker(**kwargs))
    monkeypatch.setattr(lifecycle, "threading", SimpleNamespace(Thread=worker_factory))
    monkeypatch.setattr(lifecycle, "multiprocessing", SimpleNamespace(Process=Worker))
    client = GrpcClientAdaptor(in_process=in_process)
    runner = MagicMock(spec=AppRunner)
    runner.is_stopped.return_value = (False, 0)
    client.set_runner(runner)
    context = FLContext()
    context.get_engine = MagicMock()
    context.get_job_id = lambda: "job"
    client.start_runner({"rank": 1}, context)
    assert client.is_runner_stopped() == (False, 0)
    if in_process:
        worker = worker_factory.return_value
        call = worker_factory.call_args.kwargs
        assert call["args"] == ({"rank": 1},)
        assert call["daemon"] is True
        assert call["target"].__self__ is client.starter
        client.starter.stopped = True
        client.starter.exit_code = 7
        assert client.is_runner_stopped() == (True, 7)
    else:
        worker = client.process
        assert worker.started is True
        assert worker.kwargs["args"] == ({"rank": 1},)
        worker.exitcode = 7
        assert client.is_runner_stopped() == (True, 7)
    client.stop_runner()
    client.stop_runner()
    if in_process:
        runner.stop.assert_called_once_with()
        assert client.app_runner is None
    else:
        assert worker.killed
        assert client.process is None
        assert client.is_runner_stopped() == (True, 0)


def test_monitor_handles_abort_and_reports_exit_after_bounded_poll(monkeypatch):
    client = GrpcClientAdaptor()
    client.set_abort_signal(Signal())
    client.stop = MagicMock()
    context = FLContext()
    callback = MagicMock()
    client.abort_signal.trigger(True)
    client._monitor(context, callback)
    client.stop.assert_called_once_with(context)
    callback.assert_not_called()
    client.set_abort_signal(Signal())
    client._is_stopped = MagicMock(side_effect=[(False, 0), (True, 9)])
    sleeps = []
    monkeypatch.setattr(lifecycle, "time", SimpleNamespace(sleep=lambda duration: sleeps.append(duration)))
    client._monitor(context, callback)
    assert sleeps == [0.1]
    callback.assert_called_once_with(9, context)
    with pytest.raises(RuntimeError, match="must be callable"):
        client.monitor_target(context, None)
    worker = MagicMock()
    factory = MagicMock(return_value=worker)
    monkeypatch.setattr(lifecycle, "threading", SimpleNamespace(Thread=factory))
    client.monitor_target(context, callback)
    worker.start.assert_called_once_with()
    assert factory.call_args.kwargs["target"] == client._monitor


@pytest.mark.parametrize("reply_kind", ["ok", "error", "wrong_op", "wrong_type"])
def test_request_checks_return_code_operation_and_response_type(monkeypatch, reply_kind):
    client = GrpcClientAdaptor(per_msg_timeout=3, tx_timeout=8)
    client.set_abort_signal(Signal())
    context = FLContext()
    client.engine = MagicMock()
    client.engine.new_context.return_value = context
    reply = Shareable({Constant.PARAM_KEY_RCV_BUF: b"result"})
    reply.set_header(Constant.MSG_KEY_XGB_OP, "wrong" if reply_kind == "wrong_op" else Constant.OP_ALL_GATHER)
    if reply_kind == "error":
        reply.set_return_code(ReturnCode.EXECUTION_EXCEPTION)
    send = MagicMock(return_value={} if reply_kind == "wrong_type" else reply)
    monkeypatch.setattr(protocol.ReliableMessage, "send_request", send)
    if reply_kind == "ok":
        data, got = client._send_all_gather(2, 4, b"input")
        assert data == b"result"
        assert got is reply
    else:
        with pytest.raises(
            RuntimeError,
            match={"error": "error return code", "wrong_op": "expected op", "wrong_type": "invalid reply"}[reply_kind],
        ):
            client._send_all_gather(2, 4, b"input")
    kwargs = send.call_args.kwargs
    request = kwargs["request"]
    assert request[Constant.PARAM_KEY_RANK] == 2
    assert request[Constant.PARAM_KEY_SEQ] == 4
    assert request[Constant.PARAM_KEY_SEND_BUF] == b"input"
    assert request.get_header(Constant.MSG_KEY_XGB_OP) == Constant.OP_ALL_GATHER
    assert kwargs["per_msg_timeout"] == 3
    assert kwargs["tx_timeout"] == 8
    assert kwargs["abort_signal"] is client.abort_signal
    assert context.get_prop(PROP_KEY_DEBUG_INFO) == {"op": Constant.OP_ALL_GATHER, "rank": 2, "seq": 4}


@pytest.mark.parametrize("operation", ["gatherv", "broadcast", "reduce"])
def test_collective_payloads_and_security_events_preserve_transformed_data(operation):
    client = GrpcClientAdaptor()
    client.engine = MagicMock()
    context = FLContext()
    client.engine.new_context.return_value = context
    reply = Shareable()
    sent = []

    def send(op, req):
        sent.append((op, req))
        return b"wire-result", reply

    client._send_request = send
    events = []

    def event(name, ctx):
        events.append(name)
        if name in (Constant.EVENT_BEFORE_ALL_GATHER_V, Constant.EVENT_BEFORE_BROADCAST):
            ctx.set_prop(Constant.PARAM_KEY_SEND_BUF, b"encrypted", sticky=False)
            ctx.set_prop(Constant.PARAM_KEY_HEADERS, {"secure": True}, sticky=False)
        else:
            assert ctx.get_prop(Constant.PARAM_KEY_REPLY) is reply
            ctx.set_prop(Constant.PARAM_KEY_RCV_BUF, b"decoded", sticky=False)

    client.fire_event = event
    if operation == "gatherv":
        result = client._do_all_gather_v(2, 3, b"plaintext")
        assert events == [Constant.EVENT_BEFORE_ALL_GATHER_V, Constant.EVENT_AFTER_ALL_GATHER_V]
    elif operation == "broadcast":
        result = client._do_broadcast(2, 3, 1, b"plaintext")
        assert events == [Constant.EVENT_BEFORE_BROADCAST, Constant.EVENT_AFTER_BROADCAST]
        assert sent[0][1][Constant.PARAM_KEY_ROOT] == 1
    else:
        result, got = client._send_all_reduce(2, 3, 4, 5, b"plaintext")
        assert got is reply
        assert result == b"wire-result"
        assert sent[0][1][Constant.PARAM_KEY_DATA_TYPE] == 4
        assert sent[0][1][Constant.PARAM_KEY_REDUCE_OP] == 5
        return
    assert result == b"decoded"
    assert sent[0][1][Constant.PARAM_KEY_SEND_BUF] == b"encrypted"
    assert sent[0][1].get_header("secure") is True
