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

from unittest.mock import Mock

import pytest

from nvflare.fuel.f3.cellnet import core_cell as subject
from nvflare.fuel.f3.cellnet.defs import MessageHeaderKey as H
from nvflare.fuel.f3.cellnet.defs import MessageType, ReturnCode
from nvflare.fuel.f3.cellnet.fqcn import FqcnInfo
from nvflare.fuel.f3.cellnet.registry import Registry
from nvflare.fuel.f3.endpoint import Endpoint
from nvflare.fuel.f3.message import Message


@pytest.fixture
def cell(monkeypatch):
    value = object.__new__(subject.CoreCell)
    value.my_info = FqcnInfo("site-a")
    value.logger = Mock()
    value.log_error = Mock()
    value.running = True
    value.max_msg_size = 10
    value.communicator = Mock()
    value.connector_manager = Mock()
    value.connector_manager.is_adhoc_allowed.return_value = False
    value.out_req_filter_reg = Registry()
    value.in_filter_reg = Registry()
    value.out_reply_filter_reg = Registry()
    value.sent_msg_size_pool = Mock()
    value.sent_msg_counter_pool = Mock()
    value.received_msg_counter_pool = Mock()
    value.received_msg_size_pool = Mock()
    value.msg_travel_stats_pool = Mock()
    value.message_interceptor = None
    value._record_traffic = Mock()
    value.get_fobs_context = Mock(return_value={})
    value.encrypt_payload = Mock()
    value._process_request = Mock(return_value=Message({}, b"reply"))
    value._send_reply = Mock(return_value="")
    value._forward = Mock()
    monkeypatch.setattr(value, "ALL_CELLS", {})
    monkeypatch.setattr(subject, "encode_payload", Mock())
    return value


def request():
    return Message(
        {
            H.ORIGIN: "site-b",
            H.DESTINATION: "site-a",
            H.FROM_CELL: "site-b",
            H.TO_CELL: "site-a",
            H.MSG_TYPE: MessageType.REQ,
            H.REQ_ID: "request",
            H.CHANNEL: "channel",
            H.TOPIC: "topic",
            H.REPLY_EXPECTED: True,
        },
        b"payload",
    )


@pytest.mark.parametrize("outcome", ["success", "too-large", "transport-error", "direct"])
def test_endpoint_send_enforces_encoded_size_limit_and_controls_direct_or_transport_delivery(cell, outcome):
    endpoint = Endpoint("site-b")
    message = Message({}, b"large payload" if outcome == "too-large" else b"payload")
    if outcome == "transport-error":
        cell.communicator.send.side_effect = OSError("offline")
    if outcome == "direct":
        cell.ALL_CELLS["site-b"] = Mock()
    result = cell._send_to_endpoint(endpoint, message)
    assert result == {"too-large": ReturnCode.MSG_TOO_BIG, "transport-error": ReturnCode.COMM_ERROR}.get(outcome, "")
    assert cell.sent_msg_size_pool.record_value.call_count == int(outcome in ("success", "direct"))
    assert cell.communicator.send.call_count == int(outcome in ("success", "transport-error"))
    if outcome == "direct":
        call = cell.ALL_CELLS["site-b"].process_message.call_args.kwargs
        assert call["endpoint"].name == "site-a"
        assert call["connection"] is None
        assert call["message"] is message
    assert H.SEND_TIME in message.headers


@pytest.mark.parametrize("outcome", ["no-path", "filter-error", "send-error", "success"])
def test_target_send_reports_per_target_error_without_mutating_request_headers(cell, outcome):
    endpoint = Endpoint("site-b")
    cell._find_endpoint = Mock(return_value=("missing", None) if outcome == "no-path" else ("", endpoint))
    cell._send_to_endpoint = Mock(return_value=ReturnCode.COMM_ERROR if outcome == "send-error" else "")
    if outcome == "filter-error":
        cell.add_outgoing_request_filter("channel", "topic", lambda message: Message({}, "denied"))
    original = Message({"custom": "value"}, b"payload")
    target = subject.TargetMessage("site-b", "channel", "topic", original)
    headers = dict(original.headers)
    result = cell._send_target_messages({"site-b": target})
    assert result == {
        "site-b": {
            "no-path": "missing",
            "filter-error": ReturnCode.FILTER_ERROR,
            "send-error": ReturnCode.COMM_ERROR,
        }.get(outcome, "")
    }
    assert original.headers == headers
    assert cell._send_to_endpoint.call_count == int(outcome in ("success", "send-error"))
    if outcome == "success":
        sent = cell._send_to_endpoint.call_args.args[1]
        assert sent.get_header(H.ORIGIN) == "site-a"
        assert sent.get_header(H.ROUTE)[0][0] == "site-a"


@pytest.mark.parametrize("reply_expected", [False, True])
def test_request_interceptor_returns_protocol_envelope_only_when_reply_expected(cell, reply_expected):
    cell.message_interceptor = lambda message: Message({H.RETURN_CODE: ReturnCode.INVALID_REQUEST}, b"blocked")
    cell.message_interceptor_args = ()
    cell.message_interceptor_kwargs = {}
    message = request()
    message.set_header(H.REPLY_EXPECTED, reply_expected)
    endpoint = Endpoint("site-b")
    cell._process_received_msg(endpoint, None, message)
    cell._process_request.assert_not_called()
    assert cell._send_reply.call_count == int(reply_expected)
    if reply_expected:
        reply, peer = cell._send_reply.call_args.args
        assert peer is endpoint
        assert reply.get_header(H.MSG_TYPE) == MessageType.RETURN
        assert reply.get_header(H.REQ_ID) == ["request"]
        assert reply.get_header(H.DESTINATION) == "site-b"


@pytest.mark.parametrize("outcome", ["forward", "no-result", "fire-and-forget", "reply"])
def test_received_message_routes_foreign_destination_and_honors_reply_expectation(cell, outcome):
    message = request()
    if outcome == "forward":
        message.set_header(H.DESTINATION, "site-c")
    elif outcome == "no-result":
        cell._process_request.return_value = None
    elif outcome == "fire-and-forget":
        message.set_header(H.REPLY_EXPECTED, False)
    endpoint = Endpoint("site-b")
    cell._process_received_msg(endpoint, None, message)
    if outcome == "forward":
        cell._forward.assert_called_once_with(endpoint, "site-b", "site-c", MessageType.REQ, message)
        cell._process_request.assert_not_called()
    else:
        cell._process_request.assert_called_once_with("site-b", message)
    assert cell._send_reply.call_count == int(outcome == "reply")
    if outcome == "reply":
        reply, peer = cell._send_reply.call_args.args
        assert peer is endpoint
        assert reply.get_header(H.RETURN_CODE) == ReturnCode.OK
        assert reply.get_header(H.REQ_ID) == "request"
        assert reply.get_header(H.DESTINATION) == "site-b"


def test_incoming_filter_blocks_dispatch_before_interceptor_and_reply(cell):
    blocked = Message({H.RETURN_CODE: ReturnCode.AUTHENTICATION_ERROR}, None)
    cell.add_incoming_filter("channel", "topic", lambda message: blocked)
    result = cell._process_received_msg(Endpoint("site-b"), None, request())
    assert result is blocked
    cell._process_request.assert_not_called()
    cell._send_reply.assert_not_called()
