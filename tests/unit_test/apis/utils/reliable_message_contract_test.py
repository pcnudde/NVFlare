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
from unittest.mock import Mock

import pytest

from nvflare.apis.fl_constant import ReservedKey
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import ReservedHeaderKey, ReturnCode, Shareable, make_reply
from nvflare.apis.signal import Signal
from nvflare.apis.utils import reliable_message as rm
from tests.timing_utils import isolate_time

RM = rm.ReliableMessage


@pytest.fixture(autouse=True)
def isolated_service(monkeypatch):
    for name in ("_topic_to_handle", "_req_receivers", "_req_completed", "_reply_receivers"):
        monkeypatch.setattr(RM, name, {})
    monkeypatch.setattr(RM, "_enabled", True)
    monkeypatch.setattr(RM, "_shutdown_asked", False)
    monkeypatch.setattr(RM, "_executor", Mock())
    monkeypatch.setattr(RM, "_logger", Mock())
    monkeypatch.setattr(RM, "_query_interval", 1.0)
    clock = isolate_time(monkeypatch, rm)
    clock.time = Mock(return_value=100)
    clock.sleep = Mock()
    return clock


@pytest.fixture
def context():
    ctx, peer = FLContext(), FLContext()
    ctx.set_prop(ReservedKey.ENGINE, Mock())
    ctx.set_prop(ReservedKey.IDENTITY_NAME, "server")
    peer.set_prop(ReservedKey.IDENTITY_NAME, "site-1")
    ctx.set_peer_context(peer)
    return ctx


def request(op=rm.OP_REQUEST):
    result = Shareable({"weight": 1})
    for name, value in (
        (rm.HEADER_OP, op),
        (rm.HEADER_TOPIC, "train"),
        (rm.HEADER_TX_ID, "tx"),
        (rm.HEADER_PER_MSG_TIMEOUT, 2),
        (rm.HEADER_TX_TIMEOUT, 10),
    ):
        result.set_header(name, value)
    return result


def receiver():
    return rm._RequestReceiver("train", Mock(return_value=Shareable({"weight": 2})), RM._executor, 2, 10)


@pytest.mark.parametrize("reply", [None, [], {}, {"other": make_reply(ReturnCode.OK)}])
def test_missing_target_reply_becomes_communication_error(reply):
    result, code = rm._extract_result(reply, "site-1")
    assert code == ReturnCode.COMMUNICATION_ERROR == result.get_return_code()


@pytest.mark.parametrize("missing", [rm.HEADER_TX_ID, rm.HEADER_PER_MSG_TIMEOUT, rm.HEADER_TX_TIMEOUT])
def test_receiver_creation_requires_transaction_identity_and_timeouts(missing):
    payload = request()
    payload.set_header(missing, None)
    with pytest.raises(RuntimeError, match="missing"):
        RM._get_or_create_receiver("train", payload, Mock())
    assert RM._req_receivers == {}


def test_request_is_submitted_once_and_duplicate_receives_cached_result(context):
    callback = Mock()
    first = RM._get_or_create_receiver("train", request(), callback)
    assert RM._get_or_create_receiver("train", request(), callback) is first
    assert first.process(request(), context).get_header(rm.HEADER_STATUS) == rm.STATUS_IN_PROCESS
    assert first.process(request(), context).get_header(rm.HEADER_STATUS) == rm.STATUS_IN_PROCESS
    RM._executor.submit.assert_called_once()
    result = Shareable({"weight": 2})
    first.result = result
    assert first.process(request(), context) is result
    assert first.source == "site-1" and first.tx_id == "tx"


@pytest.mark.parametrize("mode", ["disabled", "shutdown", "executor_closed"])
def test_unavailable_receiver_rejects_work(context, monkeypatch, mode):
    if mode == "disabled":
        monkeypatch.setattr(RM, "_enabled", False)
    elif mode == "shutdown":
        monkeypatch.setattr(RM, "_shutdown_asked", True)
    else:
        RM._executor.submit.side_effect = RuntimeError("closed")
    assert receiver().process(request(), context).get_return_code() == ReturnCode.SERVICE_UNAVAILABLE
    if mode != "executor_closed":
        RM._executor.submit.assert_not_called()


@pytest.mark.parametrize(
    "state,expected",
    [
        ("pending", rm.STATUS_IN_PROCESS),
        ("expired", rm.STATUS_ABORTED),
        ("sent", rm.STATUS_REPLIED),
        ("sending", rm.STATUS_IN_REPLY),
        ("unsent", None),
    ],
)
def test_receiver_query_reports_processing_reply_and_deadline_states(context, state, expected):
    entry = receiver()
    entry.rcv_time = 80 if state == "expired" else 95
    if state in ("sent", "sending", "unsent"):
        entry.result = Shareable({"weight": 2})
    entry.reply_time = 99 if state == "sent" else None
    entry.replying = state == "sending"
    result = entry.process(request(rm.OP_QUERY), context)
    assert result.get_header(rm.HEADER_STATUS) == expected
    if state == "unsent":
        assert result is entry.result


@pytest.mark.parametrize("handler_fails,acknowledged", [(False, True), (False, False), (True, True)])
def test_request_execution_labels_result_and_releases_only_acknowledged_transactions(
    context, handler_fails, acknowledged
):
    entry = receiver()
    entry.source, entry.tx_id = "site-1", "tx"
    RM._req_receivers["tx"] = entry
    if handler_fails:
        entry.request_handler_f.side_effect = ValueError("bad payload")
    context.get_engine().send_aux_request.return_value = {
        "site-1": make_reply(ReturnCode.OK if acknowledged else ReturnCode.COMMUNICATION_ERROR)
    }
    payload = request()
    entry._do_request(payload, context)
    assert entry.result.get_header(rm.HEADER_TX_ID) == "tx"
    assert entry.result.get_header(rm.HEADER_TOPIC) == "train"
    assert entry.result.get_header(rm.HEADER_OP) == rm.OP_REPLY
    assert entry.result.get_return_code() == (ReturnCode.EXECUTION_EXCEPTION if handler_fails else ReturnCode.OK)
    if handler_fails:
        assert "bad payload" in entry.result.get_header(ReservedHeaderKey.ERROR)
    else:
        assert entry.result["weight"] == 2
    assert ("tx" in RM._req_receivers) is not acknowledged
    assert ("tx" in RM._req_completed) is acknowledged
    assert (entry.reply_time is not None) is acknowledged
    assert entry.replying is False
    sent = context.get_engine().send_aux_request.call_args.kwargs
    assert sent["request"] is entry.result and sent["targets"] == ["site-1"] and sent["timeout"] == 2


@pytest.mark.parametrize(
    "mode,status,code",
    [
        ("unknown_topic", None, ReturnCode.TOPIC_UNKNOWN),
        ("duplicate", rm.STATUS_DUP_REQUEST, ReturnCode.OK),
        ("completed_query", rm.STATUS_REPLIED, ReturnCode.OK),
        ("unknown_query", rm.STATUS_NOT_RECEIVED, ReturnCode.OK),
        ("bad_operation", None, ReturnCode.BAD_REQUEST_DATA),
        ("new_request", rm.STATUS_IN_PROCESS, ReturnCode.OK),
    ],
)
def test_request_dispatch_preserves_context_and_completed_transaction_policy(context, mode, status, code):
    RM._topic_to_handle["train"] = Mock()
    payload = request(rm.OP_QUERY if "query" in mode else rm.OP_REQUEST)
    if mode == "unknown_topic":
        RM._topic_to_handle.clear()
    elif mode in ("duplicate", "completed_query"):
        RM._req_completed["tx"] = 120
    elif mode == "bad_operation":
        payload.set_header(rm.HEADER_OP, "unknown")
    result = RM._receive_request("transport", payload, context)
    assert result.get_return_code() == code and result.get_header(rm.HEADER_STATUS) == status
    assert context.get_prop(rm.PROP_KEY_TX_ID) == "tx"
    assert context.get_prop(rm.PROP_KEY_TOPIC) == "train"


def test_reply_delivery_sets_waiter_and_late_reply_is_safely_acknowledged(context):
    waiting = rm._ReplyReceiver("tx", 2, 10)
    RM._reply_receivers["tx"] = waiting
    payload = request(rm.OP_REPLY)
    assert RM._receive_reply("reply", payload, context).get_return_code() == ReturnCode.OK
    assert waiting.result_ready.is_set() and waiting.result is payload
    RM._reply_receivers.clear()
    assert RM._receive_reply("reply", payload, context).get_return_code() == ReturnCode.OK


def test_register_handler_requires_enabled_service_and_callable(context, monkeypatch):
    monkeypatch.setattr(RM, "_enabled", False)
    with pytest.raises(RuntimeError, match="not enabled"):
        RM.register_request_handler("train", Mock(), context)
    monkeypatch.setattr(RM, "_enabled", True)
    with pytest.raises(TypeError, match="callable"):
        RM.register_request_handler("train", None, context)
    handler = Mock()
    RM.register_request_handler("train", handler, context)
    assert RM._topic_to_handle == {"train": handler}
    context.get_engine().register_aux_message_handler.assert_called_once_with(
        topic="train", message_handle_func=handler
    )


@pytest.mark.parametrize("tx_timeout", [None, 1, 2])
def test_short_transactions_use_one_direct_aux_request_and_remove_message_root(context, monkeypatch, tx_timeout):
    reply = Shareable({"weight": 3})
    context.get_engine().send_aux_request.return_value = {"site-1": reply}
    cleanup = Mock()
    monkeypatch.setattr(rm, "delete_msg_root", cleanup)
    payload = Shareable()
    assert RM.send_request("site-1", "train", payload, 2, tx_timeout, Signal(), context) is reply
    assert payload.get_header(ReservedHeaderKey.MSG_ROOT_TTL) == 2
    cleanup.assert_called_once_with(payload.get_header(ReservedHeaderKey.MSG_ROOT_ID))
    sent = context.get_engine().send_aux_request.call_args.kwargs
    assert sent["topic"] == "train" and sent["timeout"] == 2


@pytest.mark.parametrize("raises", [False, True])
def test_long_transaction_cleans_waiter_and_root_after_success_or_error(context, monkeypatch, raises):
    cleanup = Mock()
    monkeypatch.setattr(rm, "delete_msg_root", cleanup)
    result = Shareable({"weight": 5})
    send = Mock(return_value=result, side_effect=ValueError("send failed") if raises else None)
    monkeypatch.setattr(RM, "_send_request", send)
    payload = Shareable()
    reply = RM.send_request("site-1", "train", payload, 2, 10, Signal(), context)
    assert RM._reply_receivers == {}
    assert payload.get_header(rm.HEADER_PER_MSG_TIMEOUT) == 2
    assert payload.get_header(rm.HEADER_TX_TIMEOUT) == 10
    assert payload.get_header(rm.HEADER_OP) == rm.OP_REQUEST
    assert reply.get_return_code() == (ReturnCode.ERROR if raises else ReturnCode.OK)
    cleanup.assert_called_once()


@pytest.mark.parametrize(
    "mode,expected",
    [("abort", ReturnCode.TASK_ABORTED), ("expired", ReturnCode.COMMUNICATION_ERROR), ("reply", ReturnCode.OK)],
)
def test_send_phase_honors_abort_deadline_and_direct_result(context, mode, expected, isolated_service):
    waiting = rm._ReplyReceiver("tx", 2, 10)
    if mode == "expired":
        waiting.tx_start_time = 80
    signal = Signal()
    if mode == "abort":
        signal.trigger(True)
    reply = Shareable({"weight": 8})
    reply.set_header(rm.HEADER_OP, rm.OP_REPLY)
    context.get_engine().send_aux_request.return_value = {"site-1": reply}
    result = RM._send_request("site-1", request(), signal, context, waiting)
    assert result.get_return_code() == expected
    if mode == "reply":
        assert result is reply
    else:
        context.get_engine().send_aux_request.assert_not_called()


@pytest.mark.parametrize(
    "status,error", [(rm.STATUS_NOT_RECEIVED, "STATUS_NOT_RECEIVED"), (rm.STATUS_ABORTED, "Aborted")]
)
def test_query_fails_when_peer_lost_or_aborted_request(context, status, error):
    waiting = rm._ReplyReceiver("tx", 2, 10)
    context.get_engine().send_aux_request.return_value = {"site-1": rm._status_reply(status)}
    result = RM._query_result("site-1", Signal(), context, waiting)
    assert result.get_return_code() == ReturnCode.EXECUTION_EXCEPTION
    assert result.get_header(ReservedHeaderKey.ERROR) == error
    query = context.get_engine().send_aux_request.call_args.kwargs["request"]
    assert query.get_header(rm.HEADER_OP) == rm.OP_QUERY and query.get_header(rm.HEADER_TX_ID) == "tx"


@pytest.mark.parametrize("mode", ["ready", "direct_reply", "abort", "expired"])
def test_query_phase_delivers_reply_or_reports_abort_and_expiry(context, mode):
    waiting = rm._ReplyReceiver("tx", 2, 10)
    reply = Shareable({"weight": 9})
    reply.set_header(rm.HEADER_OP, rm.OP_REPLY)
    signal = Signal()
    if mode == "ready":
        waiting.process(reply)
    elif mode == "abort":
        signal.trigger(True)
    elif mode == "expired":
        waiting.tx_start_time = 80
    context.get_engine().send_aux_request.return_value = {"site-1": reply}
    result = RM._query_result("site-1", signal, context, waiting)
    if mode in ("ready", "direct_reply"):
        assert result is reply
    else:
        assert result.get_return_code() == (
            ReturnCode.TASK_ABORTED if mode == "abort" else ReturnCode.COMMUNICATION_ERROR
        )


def test_completed_transaction_cache_prunes_expired_entries_and_doubles_deadline():
    RM._req_completed.update(old=99, current=100, later=110)
    RM._register_completed_req("new", 7)
    assert RM._req_completed == {"current": 100, "later": 110, "new": 114}


def test_enable_and_shutdown_are_idempotent_and_register_both_transport_handlers(context, monkeypatch):
    monkeypatch.setattr(RM, "_enabled", False)
    executor, thread = Mock(), Mock()
    make_executor, make_thread = Mock(return_value=executor), Mock(return_value=thread)
    monkeypatch.setattr(rm, "concurrent", SimpleNamespace(futures=SimpleNamespace(ThreadPoolExecutor=make_executor)))
    monkeypatch.setattr(rm, "threading", SimpleNamespace(Thread=make_thread))
    monkeypatch.setattr(rm.ConfigService, "get_int_var", Mock(return_value=3))
    monkeypatch.setattr(rm.ConfigService, "get_float_var", Mock(return_value=0.5))
    RM.enable(context)
    RM.enable(context)
    assert RM.is_available() and RM._query_interval == 0.5
    make_executor.assert_called_once_with(max_workers=3, thread_name_prefix="rm")
    thread.start.assert_called_once()
    assert [c.kwargs["topic"] for c in context.get_engine().register_aux_message_handler.call_args_list] == [
        rm.TOPIC_RELIABLE_REQUEST,
        rm.TOPIC_RELIABLE_REPLY,
    ]
    RM.shutdown()
    RM.shutdown()
    assert not RM.is_available()
    executor.shutdown.assert_called_once_with(wait=False)


def test_receiver_monitor_evicts_only_expired_requests_in_a_controlled_iteration(isolated_service):
    expired, active, pending = receiver(), receiver(), receiver()
    expired.rcv_time, active.rcv_time = 80, 95
    RM._req_receivers.update(expired=expired, active=active, pending=pending)
    isolated_service.sleep.side_effect = lambda _: setattr(RM, "_shutdown_asked", True)
    RM._monitor_req_receivers()
    assert RM._req_receivers == {"active": active, "pending": pending}
    isolated_service.sleep.assert_called_once_with(2.0)
