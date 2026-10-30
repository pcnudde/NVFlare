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

import threading
from unittest.mock import Mock

import pytest

from nvflare.fuel.f3.cellnet import core_cell as subject
from nvflare.fuel.f3.cellnet.defs import MessageHeaderKey as H
from nvflare.fuel.f3.cellnet.defs import MessageType, ReturnCode
from nvflare.fuel.f3.cellnet.fqcn import FqcnInfo
from nvflare.fuel.f3.cellnet.registry import Registry
from nvflare.fuel.f3.endpoint import Endpoint, EndpointState
from nvflare.fuel.f3.message import Message
from tests.timing_utils import isolate_time


@pytest.fixture
def cell(monkeypatch):
    value = object.__new__(subject.CoreCell)
    value.my_info = FqcnInfo("site-a")
    value.logger = Mock()
    value.log_error = Mock()
    value.log_warning = Mock()
    value.req_reg = Registry()
    value.in_req_filter_reg = Registry()
    value.out_req_filter_reg = Registry()
    value.in_reply_filter_reg = Registry()
    value.out_reply_filter_reg = Registry()
    value.error_handler_reg = Registry()
    value.waiters = {}
    value.msg_stats_pool = Mock()
    value.req_cb_stats_pool = Mock()
    value.sent_msg_counter_pool = Mock()
    value.received_msg_counter_pool = Mock()
    value.decrypt_payload = Mock()
    value.agent_lock = threading.Lock()
    value.agents = {}
    value.credential_manager = Mock()
    value.cell_connected_cb = None
    value.cell_disconnected_cb = None
    value.cell_connected_cb_args = value.cell_disconnected_cb_args = ()
    value.cell_connected_cb_kwargs = value.cell_disconnected_cb_kwargs = {}
    monkeypatch.setattr(subject, "decode_payload", Mock())
    monkeypatch.setattr(value, "ALL_CELLS", {})
    return value


def envelope(origin="site-b", request_ids="request", msg_type=MessageType.REPLY):
    return Message(
        {
            H.ORIGIN: origin,
            H.DESTINATION: "site-a",
            H.FROM_CELL: origin,
            H.TO_CELL: "site-a",
            H.CHANNEL: "channel",
            H.TOPIC: "topic",
            H.REQ_ID: request_ids,
            H.MSG_TYPE: msg_type,
        },
        b"payload",
    )


@pytest.mark.parametrize(
    "error,expected",
    [
        (subject.ServiceUnavailable, ReturnCode.SERVICE_UNAVAILABLE),
        (subject.InvalidSession, ReturnCode.INVALID_SESSION),
        (subject.InvalidRequest, ReturnCode.INVALID_REQUEST),
        (subject.AuthenticationError, ReturnCode.AUTHENTICATION_ERROR),
        (subject.AbortRun, ReturnCode.ABORT_RUN),
        (ValueError, ReturnCode.PROCESS_EXCEPTION),
    ],
)
def test_callback_exceptions_become_protocol_return_codes(cell, error, expected):
    def callback(message):
        raise error("failed")

    result = cell._try_cb(envelope(), callback)
    assert result.get_header(H.RETURN_CODE) == expected
    assert cell.log_error.call_count == int(error is ValueError)


@pytest.mark.parametrize("reply", [None, "invalid", Message({}, "answer")])
def test_incoming_request_requires_message_reply_and_propagates_secure_flag(cell, reply):
    def callback(message):
        return reply

    cell.register_request_cb("channel", "topic", callback)
    request = envelope(msg_type=MessageType.REQ)
    request.set_header(H.SECURE, True)
    result = cell._process_request("site-b", request)
    if reply is None:
        assert result is None
    elif isinstance(reply, Message):
        assert result is reply
        assert result.get_header(H.SECURE) is True
    else:
        assert result.get_header(H.RETURN_CODE) == ReturnCode.PROCESS_EXCEPTION
    cell.decrypt_payload.assert_called_once_with(request)
    cell.req_cb_stats_pool.record_value.assert_called_once()


def test_request_filters_can_short_circuit_callback_and_outgoing_filters_can_replace_reply(cell):
    callback = Mock()
    callback.__name__ = "callback"
    cell.register_request_cb("channel", "topic", callback)
    blocked = Message({}, "blocked")
    cell.add_incoming_request_filter("channel", "topic", lambda message: blocked)
    assert cell._process_request("site-b", envelope()) is blocked
    callback.assert_not_called()
    replacement = Message({}, "replacement")
    cell.add_outgoing_reply_filter("channel", "topic", lambda message: replacement)
    assert cell._filter_outgoing_reply(envelope()) is replacement
    cell.add_outgoing_request_filter("channel", "topic", lambda message: blocked)
    assert cell._filter_outgoing_request("channel", "topic", envelope()) is blocked


@pytest.mark.parametrize(
    "method",
    [
        "add_incoming_filter",
        "add_incoming_request_filter",
        "add_outgoing_reply_filter",
        "add_outgoing_request_filter",
        "add_incoming_reply_filter",
        "add_error_handler",
        "register_request_cb",
    ],
)
def test_filter_and_request_registration_reject_non_callables(cell, method):
    with pytest.raises(ValueError, match="not callable"):
        getattr(cell, method)("channel", "topic", object())


def test_reply_waiter_is_triggered_only_after_all_expected_targets_reply(cell, monkeypatch):
    clock = isolate_time(monkeypatch, subject)
    monkeypatch.setattr(clock, "time", lambda: 100)
    waiter = subject._Waiter(["site-b", "site-c"])
    cell.waiters["request"] = waiter
    first = envelope("site-b")
    cell._process_reply("site-b", first, MessageType.REPLY)
    assert waiter.received_replies == {"site-b": first}
    assert not waiter.is_set()
    second = envelope("site-c")
    cell._process_reply("site-c", second, MessageType.REPLY)
    assert waiter.is_set()
    assert waiter.received_replies == {"site-b": first, "site-c": second}
    assert waiter.reply_time == {"site-b": 100, "site-c": 100}
    assert cell.msg_stats_pool.record_value.call_count == 2


def test_reply_from_unexpected_peer_is_dropped_and_late_reply_is_counted(cell):
    waiter = subject._Waiter(["site-b"])
    cell.waiters["request"] = waiter
    cell._process_reply("unexpected", envelope("unexpected"), MessageType.REPLY)
    assert not waiter.received_replies and not waiter.is_set()
    cell.log_error.assert_called_once()
    cell._process_reply("site-b", envelope(request_ids="late"), MessageType.REPLY)
    cell.log_warning.assert_called_once()
    cell.sent_msg_counter_pool.increment.assert_called_once()


@pytest.mark.parametrize("request_ids", [None, 42])
def test_reply_rejects_missing_or_malformed_request_ids(cell, request_ids):
    with pytest.raises(RuntimeError, match="REQ_ID"):
        cell._process_reply("site-b", envelope(request_ids=request_ids), MessageType.REPLY)


@pytest.mark.parametrize(
    "original", [None, {}, {H.DESTINATION: "site-c", H.CHANNEL: "original-channel", H.TOPIC: "original-topic"}]
)
def test_returned_reply_binds_to_original_destination_and_preserves_category(cell, original):
    message = envelope(msg_type=MessageType.RETURN)
    if original is not None:
        message.set_header(H.ORIGINAL_HEADERS, original)
    waiter = subject._Waiter(["site-c"])
    cell.waiters["request"] = waiter
    if not original:
        with pytest.raises(RuntimeError, match="ORIGINAL_HEADERS"):
            cell._process_reply("site-b", message, MessageType.RETURN)
    else:
        cell._process_reply("site-b", message, MessageType.RETURN)
        assert waiter.is_set() and waiter.received_replies["site-c"] is message
        assert cell._stats_category(message) == "return:original-channel:original-topic"


@pytest.mark.parametrize("failure", ["no-path", "send-error", "success"])
def test_forwarding_sends_return_only_when_requester_expected_reply(cell, monkeypatch, failure):
    incoming = Endpoint("site-b")
    next_hop = Endpoint("site-c")
    cell._find_endpoint = Mock(return_value=("no path", None) if failure == "no-path" else ("", next_hop))
    cell._send_to_endpoint = Mock(return_value="send failed" if failure == "send-error" else "")
    message = envelope(msg_type=MessageType.REQ)
    message.set_header(H.DESTINATION, "site-c")
    message.set_header(H.REPLY_EXPECTED, True)
    cell._forward(incoming, "site-b", "site-c", MessageType.REQ, message)
    if failure == "success":
        cell._send_to_endpoint.assert_called_once_with(to_endpoint=next_hop, message=message)
        assert message.get_header(H.TO_CELL) == "site-c"
        assert message.get_header(H.ROUTE)[0][0] == "site-a"
    else:
        endpoint, reply = cell._send_to_endpoint.call_args.args
        assert endpoint is incoming
        assert reply.get_header(H.MSG_TYPE) == MessageType.RETURN
        assert reply.get_header(H.RETURN_CODE) == ReturnCode.COMM_ERROR
        assert reply.get_header(H.DESTINATION) == "site-b"
        assert reply.get_header(H.REQ_ID) == ["request"]


def test_endpoint_reconnect_replaces_agent_and_contains_application_callback_errors(cell):
    callback = Mock(side_effect=ValueError("app callback failed"))
    cell.cell_connected_cb = callback
    first = Endpoint("site-b")
    first.state = EndpointState.READY
    cell.state_change(first)
    agent = cell.agents["site-b"]
    second = Endpoint("site-b")
    second.state = EndpointState.READY
    cell.state_change(second)
    assert cell.agents["site-b"] is agent and agent.endpoint is second
    assert cell.credential_manager.clear_cert_cache.call_count == 2
    cell.cell_disconnected_cb = Mock(side_effect=ValueError("disconnect failed"))
    second.state = EndpointState.DISCONNECTED
    cell.state_change(second)
    assert not cell.agents
    cell.cell_disconnected_cb.assert_called_once_with(agent)
    assert cell.log_error.call_count == 3


def test_subcell_discovery_unions_local_and_connected_direct_children(cell):
    cell.my_info = FqcnInfo("server")
    cell.agents = {name: subject.CellAgent(name, Endpoint(name)) for name in ("server.job", "site-a")}
    cell.ALL_CELLS = {"server.job": object(), "server.other": object(), "site-b": object(), "site-b.job": object()}
    children, clients = cell.get_sub_cell_names()
    assert set(children) == {"server.job", "server.other"}
    assert set(clients) == {"site-a", "site-b"}
