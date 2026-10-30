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
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from nvflare.apis.event_type import EventType
from nvflare.apis.fl_constant import ServerCommandKey
from nvflare.apis.fl_context import FLContext
from nvflare.fuel.hci.server.constants import ConnProps
from nvflare.private.admin_defs import Message
from nvflare.private.fed.server import admin as subject
from tests.timing_utils import isolate_time


@pytest.fixture
def admin():
    value = object.__new__(subject.FedAdminServer)
    value.client_lock = threading.Lock()
    value.clients = {}
    value.sai = Mock()
    value.sai.new_context.side_effect = lambda: nullcontext(FLContext())
    value.cell = Mock()
    return value


def test_admin_client_heartbeat_indexes_tokens_and_updates_activity(admin, monkeypatch):
    clock = isolate_time(monkeypatch, subject)
    monkeypatch.setattr(clock, "time", Mock(side_effect=[1, 2]))
    first = admin.client_heartbeat("token", "site-a", "root.site-a")
    assert (first.name, first.fqcn, first.last_heard_time) == ("site-a", "root.site-a", 1)
    assert admin.client_heartbeat("token", "site-a", "root.site-a") is first
    assert first.last_heard_time == 2
    assert admin.get_client_tokens() == ["token"]
    admin.client_dead("token")
    admin.client_dead("missing")
    assert admin.get_client_tokens() == []


@pytest.mark.parametrize("reply", [None, [], [SimpleNamespace(reply="response")]])
def test_single_admin_send_returns_first_reply_or_none(admin, monkeypatch, reply):
    send = Mock(return_value=reply)
    monkeypatch.setattr(admin, "send_requests", send)
    message = Message("inspect", {})
    result = admin.send_request_to_client(message, "token", timeout_secs=7)
    assert result is (reply[0] if reply else None)
    assert send.call_args.args[0] == {"token": message}
    assert send.call_args.kwargs == {"timeout_secs": 7}
    with pytest.raises(TypeError, match="request must be Message"):
        admin.send_request_to_client({}, "token")


def test_admin_reply_dict_preserves_missing_clients_as_none(admin, monkeypatch):
    message = Message("inspect", {})
    reply = Message("inspect", {"ok": True})
    send = Mock(return_value=[SimpleNamespace(client_token="received", reply=reply)])
    monkeypatch.setattr(admin, "send_requests", send)
    result = admin.send_requests_and_get_reply_dict({"received": message, "missing": message}, timeout_secs=3)
    assert result == {"received": reply, "missing": None}
    assert admin.send_requests_and_get_reply_dict({}) == {}
    assert send.call_count == 1


def test_admin_send_fires_policy_event_and_includes_only_public_peer_context(admin, monkeypatch):
    send = Mock(return_value=[])
    monkeypatch.setattr(subject, "send_requests", send)
    ctx = FLContext()
    ctx.set_prop("visible", 1, private=False)
    ctx.set_prop("secret", 2, private=True)
    message = Message("inspect", {})
    assert admin.send_requests({"token": message}, ctx, timeout_secs=8, optional=True) == []
    admin.sai.fire_event.assert_called_once_with(EventType.BEFORE_SEND_ADMIN_COMMAND, ctx)
    peer = message.get_header(ServerCommandKey.PEER_FL_CONTEXT)
    assert peer.get_prop("visible") == 1
    assert peer.get_prop("secret") is None
    send.assert_called_once_with(
        cell=admin.cell,
        command="admin",
        requests={"token": message},
        clients=admin.clients,
        timeout_secs=8,
        optional=True,
    )


def test_new_admin_message_copies_command_and_present_security_properties():
    props = {
        ConnProps.CMD_ENTRY: SimpleNamespace(name="inspect"),
        ConnProps.EVENT_ID: "event",
        ConnProps.USER_NAME: "alice",
        ConnProps.SUBMITTER_NAME: "bob",
        ConnProps.USER_ORG: "org",
    }
    conn = Mock()
    conn.get_prop.side_effect = lambda key, default=None: props.get(key, default)
    message = subject.new_message(conn, "topic", {"payload": 1}, require_authz=True)
    assert message.topic == "topic" and message.body == {"payload": 1}
    assert message.get_header(subject.RequestHeader.ADMIN_COMMAND) == "inspect"
    assert message.get_header(subject.RequestHeader.REQUIRE_AUTHZ) == "true"
    for key in (ConnProps.EVENT_ID, ConnProps.USER_NAME, ConnProps.SUBMITTER_NAME, ConnProps.USER_ORG):
        assert message.get_header(key) == props[key]
    assert message.get_header(ConnProps.USER_ROLE) is None
