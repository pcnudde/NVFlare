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
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from nvflare.fuel.f3.cellnet import core_cell as module
from nvflare.fuel.f3.cellnet.defs import MessageHeaderKey, ReturnCode
from nvflare.fuel.f3.cellnet.fqcn import FqcnInfo
from nvflare.fuel.f3.endpoint import Endpoint
from nvflare.fuel.f3.message import Message


@pytest.fixture
def cell():
    subject = module.CoreCell.__new__(module.CoreCell)
    subject.my_info = FqcnInfo("site.job.worker")
    subject.logger, subject.log_error, subject.log_warning = Mock(), Mock(), Mock()
    subject.ALL_CELLS, subject.agents, subject.waiters = {}, {}, {}
    subject.max_timeout, subject.running, subject.secure = 5, True, True
    subject.stats_lock, subject.agent_lock = threading.Lock(), threading.Lock()
    subject.sent_msg_counter_pool, subject.in_reply_filter_reg = Mock(), Mock()
    subject.sent_msg_counter_pool.increment = Mock()
    subject.decrypt_payload = Mock()
    subject._try_path = Mock(return_value=None)
    return subject


@pytest.mark.parametrize("mode", ["ancestor-local", "ancestor-agent", "none"])
def test_authenticated_transit_walks_local_ancestors_and_fails_closed_without_uplink(cell, mode):
    message = Message(headers={MessageHeaderKey.SERVER_TRANSIT_REQUIRED: True})
    endpoint = Endpoint("site")
    if mode == "ancestor-local":
        cell.ALL_CELLS["site"] = object()
    elif mode == "ancestor-agent":
        cell.agents["site"] = SimpleNamespace(endpoint=endpoint)
    result = cell._try_find_ep("other.job", message)
    if mode == "none":
        assert result is None
        assert "no server-transit path" in cell.log_warning.call_args.kwargs["log_text"]
    else:
        assert result.name == "site"
    assert not cell.is_server_transit_return(
        Message(
            headers={
                MessageHeaderKey.SERVER_TRANSIT_REQUIRED: True,
                MessageHeaderKey.ROUTE: [(module.FQCN.ROOT_SERVER, 1)],
            }
        )
    )


@pytest.mark.parametrize("mode", ["local-root", "missing-parent", "parent"])
def test_unmarked_routes_fall_back_to_root_or_registered_parent_and_warn_when_unreachable(cell, mode):
    if mode == "local-root":
        cell.ALL_CELLS[module.FQCN.ROOT_SERVER] = object()
    elif mode == "parent":
        cell.agents["site.job"] = SimpleNamespace(endpoint=Endpoint("site.job"))
    result = cell._try_find_ep("other.job", Message())
    assert (result.name if result else None) == {
        "local-root": module.FQCN.ROOT_SERVER,
        "parent": "site.job",
        "missing-parent": None,
    }[mode]
    if mode == "missing-parent":
        assert "no connection to parent" in cell.log_warning.call_args.args[0]


def test_bulk_and_send_admission_reject_disabled_or_stopped_runtime(cell):
    cell.running = False
    with pytest.raises(RuntimeError, match="not running"):
        cell._send_target_messages({})
    cell.max_bulk_size = 0
    with pytest.raises(RuntimeError, match="not enabled"):
        cell.queue_message("channel", "topic", "peer", Message())
    assert cell.is_secure()
    cell._peer_goodbye(Message())
    assert "no endpoint" in cell.log_error.call_args.args[0]
    request = Message()
    request.set_prop(module.MessagePropKey.ENDPOINT, Endpoint("gone"))
    assert isinstance(cell._peer_goodbye(request), Message)
    assert cell.agents == {}


def test_multi_fire_and_forget_and_failed_reply_route_preserve_protocol_headers(cell):
    message = Message()
    target = module.TargetMessage("peer", "channel", "topic", message)
    cell._send_target_messages = Mock(return_value={"peer": ReturnCode.COMM_ERROR})
    assert cell.fire_multi_requests_and_forget({"peer": target}, optional=True, secure=True) == {
        "peer": ReturnCode.COMM_ERROR
    }
    assert message.get_header(MessageHeaderKey.REPLY_EXPECTED) is False
    assert message.get_header(MessageHeaderKey.OPTIONAL) and message.get_header(MessageHeaderKey.SECURE)
    cell._find_endpoint = Mock(return_value=(ReturnCode.COMM_ERROR, None))
    reply = Message()
    assert cell.send_reply(reply, "peer", ["request"], secure=True) == ReturnCode.COMM_ERROR
    assert reply.get_header(MessageHeaderKey.REQ_ID) == ["request"]
    assert reply.get_header(MessageHeaderKey.DESTINATION) == "peer"


def test_broadcast_send_errors_generate_per_target_reply_and_always_release_waiter(cell):
    target = module.TargetMessage("peer", "channel", "topic", Message())
    cell._send_target_messages = Mock(return_value={"peer": ReturnCode.INVALID_REQUEST})
    reply = cell.broadcast_multi_requests({"peer": target}, timeout=0)
    assert reply["peer"].get_header(MessageHeaderKey.RETURN_CODE) == ReturnCode.INVALID_REQUEST
    assert cell.waiters == {}
    cell._send_target_messages.side_effect = OSError("send failed")
    with pytest.raises(OSError, match="send failed"):
        cell.broadcast_multi_requests({"peer": target})
    assert cell.waiters == {}


@pytest.mark.parametrize("missing", [MessageHeaderKey.MSG_TYPE, MessageHeaderKey.ORIGIN, MessageHeaderKey.DESTINATION])
def test_incoming_envelope_requires_type_origin_and_destination_before_processing(cell, missing):
    headers = {
        MessageHeaderKey.MSG_TYPE: module.MessageType.REQ,
        MessageHeaderKey.ORIGIN: "peer",
        MessageHeaderKey.DESTINATION: "site.job.worker",
    }
    del headers[missing]
    with pytest.raises(RuntimeError, match="missing"):
        cell._process_received_msg(Endpoint("peer"), None, Message(headers=headers))


def test_reply_filters_run_before_waiter_lookup_and_invalid_return_headers_fail_closed(cell, monkeypatch):
    monkeypatch.setattr(module, "decode_payload", Mock())
    callback = Mock()
    callback.__name__ = "filter"
    cell.in_reply_filter_reg.find.return_value = [module.Callback(callback, (), {})]
    message = Message(headers={MessageHeaderKey.REQ_ID: "gone"})
    cell._process_reply("peer", message, module.MessageType.REPLY)
    callback.assert_called_once_with(message)
    returned = Message(
        headers={MessageHeaderKey.REQ_ID: "gone", MessageHeaderKey.ORIGINAL_HEADERS: {"custom": "value"}}
    )
    with pytest.raises(RuntimeError, match="missing DESTINATION"):
        cell._process_reply("peer", returned, module.MessageType.RETURN)


def test_unregistered_request_callback_returns_protocol_error_and_receiver_contains_decoder_errors(cell, monkeypatch):
    monkeypatch.setattr(module, "decode_payload", Mock())
    cell.req_reg = SimpleNamespace(find=Mock(return_value=None))
    response = cell._process_request("peer", Message())
    assert response.get_header(MessageHeaderKey.RETURN_CODE) == ReturnCode.PROCESS_EXCEPTION
    assert response.get_header(MessageHeaderKey.ERROR) == "no callback"
    cell._process_received_msg = Mock(side_effect=ValueError("bad decoder"))
    cell.process_message(Endpoint("peer"), None, 1, Message())
    assert "bad decoder" in cell.log_error.call_args.args[0]
    assert cell.log_error.call_args.kwargs["log_except"] is True


def test_backbone_connector_is_unnecessary_without_url_or_with_local_root(cell):
    cell.connector_manager = Mock()
    cell.root_url = None
    cell._create_bb_external_connector()
    cell.root_url = "tcp://root"
    cell.ALL_CELLS[module.FQCN.ROOT_SERVER] = object()
    cell._create_bb_external_connector()
    cell.connector_manager.get_external_connector.assert_not_called()
