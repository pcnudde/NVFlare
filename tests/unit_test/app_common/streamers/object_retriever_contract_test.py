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
from nvflare.app_common.streamers import object_retriever as subject
from nvflare.app_common.streamers.container_retriever import ContainerRetriever
from tests.timing_utils import isolate_time


@pytest.fixture
def runtime():
    retriever = ContainerRetriever()
    engine = Mock()
    ctx = FLContext()
    ctx.set_prop(ReservedKey.ENGINE, engine, private=True, sticky=True)
    peer = FLContext()
    peer.set_prop(ReservedKey.IDENTITY_NAME, "owner", private=True, sticky=True)
    ctx.set_peer_context(peer)
    return retriever, ctx, engine


def test_start_run_registers_matching_stream_and_request_topics(runtime, monkeypatch):
    retriever, ctx, engine = runtime
    register = Mock()
    monkeypatch.setattr(retriever, "register_stream_processing", register)
    retriever.handle_event(EventType.START_RUN, ctx)
    assert retriever.topic == "ContainerRetriever"
    register.assert_called_once_with(
        fl_ctx=ctx,
        channel="ContainerRetriever",
        topic="ContainerRetriever",
        stream_done_cb=retriever._handle_stream_done,
    )
    engine.register_aux_message_handler.assert_called_once_with(
        topic=retriever.topic, message_handle_func=retriever._handle_request
    )


def test_completed_retrieval_returns_stream_result_and_removes_transaction(runtime, monkeypatch):
    retriever, ctx, engine = runtime
    retriever.get_result = Mock(return_value=(ReturnCode.OK, [1, 2]))

    def request(**kwargs):
        req = kwargs["request"]
        assert req["name"] == "weights"
        assert kwargs["targets"] == ["owner"]
        assert kwargs["timeout"] == 3
        retriever._handle_stream_done({subject.RETRIEVER_TX_ID: req[subject.RETRIEVER_TX_ID]}, ctx)
        return {"owner": make_reply(ReturnCode.OK)}

    engine.send_aux_request.side_effect = request
    assert retriever.retrieve("owner", ctx, 3, name="weights") == (ReturnCode.OK, [1, 2])
    assert retriever.tx_table == {}


@pytest.mark.parametrize(
    "reply,expected",
    [
        (None, ReturnCode.EXECUTION_EXCEPTION),
        ("wrong", ReturnCode.EXECUTION_EXCEPTION),
        (make_reply(ReturnCode.BAD_REQUEST_DATA), ReturnCode.BAD_REQUEST_DATA),
    ],
)
def test_rejected_or_missing_replies_propagate_failure(runtime, reply, expected):
    retriever, ctx, engine = runtime
    engine.send_aux_request.return_value = {"owner": reply}
    assert retriever.retrieve("owner", ctx, 1) == (expected, None)


@pytest.mark.parametrize("outcome", ["timeout", "abort", "exception"])
def test_retrieval_deadline_abort_and_transport_exception_terminate_without_waiting(runtime, monkeypatch, outcome):
    retriever, ctx, engine = runtime
    engine.send_aux_request.return_value = {"owner": make_reply(ReturnCode.OK)}
    clock = isolate_time(monkeypatch, subject)
    clock.time = Mock(side_effect=[0, 2])
    monkeypatch.setattr(subject, "_SHORT_WAIT", 0)
    signal = Signal()
    ctx.set_prop(ReservedKey.RUN_ABORT_SIGNAL, signal, private=True, sticky=True)
    if outcome == "abort":
        signal.trigger(True)
        clock.time = Mock(return_value=0)
    if outcome == "exception":
        engine.send_aux_request.side_effect = RuntimeError("network failed")
    expected = {
        "timeout": ReturnCode.TIMEOUT,
        "abort": ReturnCode.TASK_ABORTED,
        "exception": ReturnCode.EXECUTION_EXCEPTION,
    }[outcome]
    assert retriever.retrieve("owner", ctx, 1) == (expected, None)
    assert retriever.tx_table == {}


def test_stream_completion_ignores_late_transactions_and_contains_result_failure(runtime):
    retriever, ctx, _ = runtime
    retriever.get_result = Mock(side_effect=ValueError("bad container"))
    retriever._handle_stream_done({subject.RETRIEVER_TX_ID: "late"}, ctx)
    retriever.get_result.assert_not_called()
    waiter = subject._Waiter()
    waiter.set_result(ReturnCode.OK, [3])
    assert waiter.result == (ReturnCode.OK, [3])
    retriever.tx_table["live"] = waiter
    retriever._handle_stream_done({subject.RETRIEVER_TX_ID: "live"}, ctx)
    assert waiter.is_set()
    assert waiter.result == (ReturnCode.EXECUTION_EXCEPTION, None)


@pytest.mark.parametrize("invalid", ["transaction", "peer", "peer-type", "peer-name"])
def test_requests_require_transaction_and_named_peer_before_validation(runtime, invalid):
    retriever, ctx, _ = runtime
    req = Shareable({subject.RETRIEVER_TX_ID: "transaction"})
    if invalid == "transaction":
        req.pop(subject.RETRIEVER_TX_ID)
    elif invalid == "peer":
        ctx.set_peer_context(None)
    elif invalid == "peer-type":
        ctx.set_peer_context(object())
        retriever.log_error = Mock()
    else:
        ctx.set_peer_context(FLContext())
    retriever.validate_request = Mock()
    assert retriever._handle_request("topic", req, ctx).get_return_code() == ReturnCode.BAD_REQUEST_DATA
    retriever.validate_request.assert_not_called()


@pytest.mark.parametrize("validation", [ReturnCode.BAD_REQUEST_DATA, "error", ReturnCode.OK])
def test_request_validation_precedes_stream_launch_and_preserves_validated_data(runtime, monkeypatch, validation):
    retriever, ctx, _ = runtime
    req = Shareable({subject.RETRIEVER_TX_ID: "transaction"})
    if validation == "error":
        retriever.validate_request = Mock(side_effect=ValueError("bad request"))
    else:
        retriever.validate_request = Mock(return_value=(validation, [1, 2]))
    thread = Mock()
    factory = Mock(return_value=thread)
    monkeypatch.setattr(subject, "threading", SimpleNamespace(Thread=factory))
    reply = retriever._handle_request("topic", req, ctx)
    if validation == ReturnCode.OK:
        assert reply.get_return_code() == ReturnCode.OK
        assert factory.call_args.kwargs["target"] == retriever._do_stream
        assert factory.call_args.kwargs["args"] == (req, ctx, [1, 2])
        thread.start.assert_called_once()
    else:
        expected = ReturnCode.EXECUTION_EXCEPTION if validation == "error" else validation
        assert reply.get_return_code() == expected
        factory.assert_not_called()


@pytest.mark.parametrize("fails", [False, True])
def test_stream_sender_forwards_transaction_and_peer_and_contains_exception(runtime, fails):
    retriever, ctx, _ = runtime
    retriever.do_stream = Mock(side_effect=ValueError("failed") if fails else None)
    request = Shareable({subject.RETRIEVER_TX_ID: "transaction"})
    retriever._do_stream(request, ctx, [3])
    retriever.do_stream.assert_called_once_with("owner", request, ctx, {subject.RETRIEVER_TX_ID: "transaction"}, [3])
