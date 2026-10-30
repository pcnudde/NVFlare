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

from threading import Lock
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from nvflare.apis.client import Client
from nvflare.apis.event_type import EventType
from nvflare.apis.fl_constant import FLContextKey, RunProcessKey, ServerCommandKey, ServerCommandNames
from nvflare.apis.fl_context import FLContext
from nvflare.apis.fl_exception import NotAuthenticated
from nvflare.apis.shareable import Shareable
from nvflare.fuel.f3.cellnet.defs import IdentityChallengeKey, MessageHeaderKey
from nvflare.fuel.f3.cellnet.defs import ReturnCode as CellReturnCode
from nvflare.fuel.f3.message import Message
from nvflare.private.defs import CellMessageHeaderKeys, ClientRegMsgKey, ClientType, InternalFLContextKey
from nvflare.private.fed.server import fed_server


@pytest.fixture
def server():
    server = fed_server.FederatedServer.__new__(fed_server.FederatedServer)
    server.engine, server.client_manager, server.admin_server, server.logger = Mock(), Mock(), Mock(), Mock()
    server.reg_lock, server.lock = Lock(), Lock()
    server.name_to_reg, server.tokens = {}, {}
    server.project_name = "project"
    server.server_state = SimpleNamespace(ssid="session")
    context = FLContext()
    server.engine.new_context.side_effect = lambda: context
    server._test_context = context
    server._ready_for_registration = Mock(return_value=None)
    server.sign_auth_token = Mock(return_value="signature")
    server.client_manager.authenticate.return_value = Client("site-1", "token")
    return server


def _request(client_type=ClientType.REGULAR, config=None):
    payload = Shareable()
    payload.set_peer_context(FLContext())
    if config is not None:
        payload[ClientRegMsgKey.SITE_CONFIG] = config
    return Message(
        headers={
            MessageHeaderKey.ORIGIN: "relay.site-1",
            CellMessageHeaderKeys.CLIENT_NAME: "site-1",
            CellMessageHeaderKeys.CLIENT_TYPE: client_type,
        },
        payload=payload,
    )


@pytest.mark.parametrize("client_type", [ClientType.REGULAR, ClientType.RELAY])
def test_registration_returns_signed_token_and_tracks_only_regular_clients(server, client_type):
    request = _request(client_type, {"capability": "test"})
    server._test_context.set_prop("_cc_info", {"attested": True})
    reply = server.register_client(request)
    assert reply.get_header(MessageHeaderKey.RETURN_CODE) == CellReturnCode.OK
    assert reply.payload[CellMessageHeaderKeys.TOKEN] == "token"
    assert reply.payload[CellMessageHeaderKeys.TOKEN_SIGNATURE] == "signature"
    assert reply.payload[CellMessageHeaderKeys.SSID] == "session"
    assert reply.payload["_cc_info"] == {"attested": True}
    assert server._test_context.get_peer_context() is request.payload.get_peer_context()
    assert server._test_context.get_prop(FLContextKey.CLIENT_TYPE) == client_type
    events = [call.args[0] for call in server.engine.fire_event.call_args_list]
    assert events == [EventType.CLIENT_REGISTER_RECEIVED, EventType.CLIENT_REGISTER_PROCESSED]
    if client_type == ClientType.REGULAR:
        assert server._test_context.get_prop(FLContextKey.CLIENT_SITE_CONFIG) == {"capability": "test"}
        assert server.tokens["token"][CellMessageHeaderKeys.CLIENT_NAME] == "site-1"
        server.admin_server.client_heartbeat.assert_called_once_with("token", "site-1", "site-1")
    else:
        assert server._test_context.get_prop(FLContextKey.CLIENT_SITE_CONFIG) is None
        assert server.tokens == {}
        server.admin_server.client_heartbeat.assert_not_called()


@pytest.mark.parametrize("client", [None, Client("site-1", "")])
def test_no_authenticated_token_returns_empty_payload_without_tracking_client(server, client):
    server.client_manager.authenticate.return_value = client
    reply = server.register_client(_request())
    assert reply.payload == {} and server.tokens == {}
    server.sign_auth_token.assert_not_called()


def test_registration_readiness_denial_returns_transport_error_before_authentication(server):
    server._ready_for_registration.return_value = "not ready"
    reply = server.register_client(_request())
    assert reply.get_header(MessageHeaderKey.RETURN_CODE) == CellReturnCode.COMM_ERROR
    assert reply.get_header(MessageHeaderKey.ERROR) == "not ready"
    server.client_manager.authenticate.assert_not_called()


@pytest.mark.parametrize("challenge_exists", [False, True])
def test_secure_registration_requires_and_consumes_origin_bound_challenge(server, challenge_exists):
    server._test_context.set_prop(FLContextKey.SECURE_MODE, True)
    session = object()
    if challenge_exists:
        server.name_to_reg["relay.site-1"] = session
    reply = server.register_client(_request())
    assert server.name_to_reg == {}
    if challenge_exists:
        assert reply.get_header(MessageHeaderKey.RETURN_CODE) == CellReturnCode.OK
        assert server._test_context.get_prop(InternalFLContextKey.CLIENT_REG_SESSION) is session
    else:
        assert reply.get_header(MessageHeaderKey.RETURN_CODE) == CellReturnCode.UNAUTHENTICATED
        server.client_manager.authenticate.assert_not_called()


def test_registration_event_authentication_veto_prevents_token_creation(server):
    def veto(event, fl_ctx):
        fl_ctx.set_prop(FLContextKey.EXCEPTIONS, {"policy": NotAuthenticated("private detail")})

    server.engine.fire_event.side_effect = veto
    reply = server.register_client(_request())
    assert reply.get_header(MessageHeaderKey.RETURN_CODE) == CellReturnCode.UNAUTHENTICATED
    assert "private detail" not in reply.get_header(MessageHeaderKey.ERROR)
    server.client_manager.authenticate.assert_not_called()


@pytest.mark.parametrize(
    "ready,secure,expected",
    [
        (False, True, CellReturnCode.COMM_ERROR),
        (True, False, CellReturnCode.UNAUTHENTICATED),
        (True, True, CellReturnCode.OK),
    ],
)
def test_challenge_requires_ready_secure_server_and_returns_origin_bound_nonce(server, ready, secure, expected):
    server._test_context.set_prop(FLContextKey.SECURE_MODE, secure)
    server._ready_for_registration.return_value = None if ready else "busy"
    asserter = Mock(cn="server.example", cert_data=b"server-chain")
    asserter.sign_common_name.return_value = "signed-client-nonce"
    server._get_id_asserter = Mock(return_value=asserter)
    request = Message(headers={MessageHeaderKey.ORIGIN: "relay.site-1", IdentityChallengeKey.NONCE: "client-nonce"})
    server.name_to_reg["relay.site-1"] = object()
    reply = server.client_challenge(request)
    assert reply.get_header(MessageHeaderKey.RETURN_CODE) == expected
    if ready and secure:
        asserter.sign_common_name.assert_called_once_with("client-nonce")
        assert reply.payload[IdentityChallengeKey.NONCE] == server.name_to_reg["relay.site-1"].nonce
        assert reply.payload[IdentityChallengeKey.COMMON_NAME] == "server.example"
        assert reply.payload[IdentityChallengeKey.CERT] == b"server-chain"
        assert reply.payload[IdentityChallengeKey.SIGNATURE] == "signed-client-nonce"


@pytest.mark.parametrize(
    "command,known",
    [
        (ServerCommandNames.GET_CLIENTS, True),
        (ServerCommandNames.GET_CLIENTS, False),
        (ServerCommandNames.UPDATE_RUN_STATUS, True),
        (ServerCommandNames.UPDATE_RUN_STATUS, False),
        (ServerCommandNames.HEARTBEAT, True),
        (ServerCommandNames.HEARTBEAT, False),
        ("invalid", False),
    ],
)
def test_parent_command_router_handles_participants_status_and_stale_job_heartbeats(server, command, known):
    client = Client("site-1", "token")
    info = {RunProcessKey.PARTICIPANTS: {"token": client}}
    server.engine.run_processes = {"job": info} if known else {}
    server.engine.exception_run_processes = {}
    server._set_job_aborted = Mock()
    request = Message(
        headers={CellMessageHeaderKeys.JOB_ID: "job", MessageHeaderKey.TOPIC: command},
        payload={"execution_error": True},
    )
    reply = server._listen_command(request)
    assert reply.get_header(MessageHeaderKey.RETURN_CODE) == (
        CellReturnCode.INVALID_REQUEST if command == "invalid" else CellReturnCode.OK
    )
    if command == ServerCommandNames.GET_CLIENTS:
        assert reply.payload[ServerCommandKey.CLIENTS] == ({"token": client} if known else None)
    elif command == ServerCommandNames.UPDATE_RUN_STATUS and known:
        assert info[RunProcessKey.PROCESS_EXE_ERROR] is True and info[RunProcessKey.PROCESS_FINISHED] is True
        assert server.engine.exception_run_processes["job"] is info
    elif command == ServerCommandNames.HEARTBEAT:
        if known:
            server.engine.abort_app_on_server.assert_not_called()
        else:
            server.engine.abort_app_on_server.assert_called_once_with("job")
            server._set_job_aborted.assert_called_once_with("job")


def test_quit_validated_client_revokes_token_and_fires_event_with_peer_context(server):
    client = Client("site-1", "token")
    server.client_manager.validate_client.return_value = client
    server.logout_client = Mock()
    request = _request()
    reply = server.quit_client(request)
    server.logout_client.assert_called_once_with("token")
    assert reply.get_header(CellMessageHeaderKeys.MESSAGE) == "Removed client"
    assert server._test_context.get_peer_context() is request.payload.get_peer_context()
    assert server.engine.fire_event.call_args.args[0] == EventType.CLIENT_QUIT


def test_reply_generation_prioritizes_authentication_over_communication_error(server):
    context = FLContext()
    context.set_prop(FLContextKey.COMMUNICATION_ERROR, "offline")
    context.set_prop(FLContextKey.UNAUTHENTICATED, "denied")
    assert (
        server._generate_reply({}, {}, context).get_header(MessageHeaderKey.RETURN_CODE)
        == CellReturnCode.UNAUTHENTICATED
    )
    context.remove_prop(FLContextKey.UNAUTHENTICATED)
    assert server._generate_reply({}, {}, context).get_header(MessageHeaderKey.RETURN_CODE) == CellReturnCode.COMM_ERROR


def test_dead_client_sweep_expires_only_stale_connections_and_logout_revokes_all_views(server, monkeypatch):
    stale, fresh = Client("stale", "old"), Client("fresh", "new")
    stale.last_connect_time, fresh.last_connect_time = 50.0, 95.0
    server.heart_beat_timeout = 10.0
    server.client_manager.get_clients.return_value = {"old": stale, "new": fresh}
    original_logout = server.logout_client
    server.logout_client = Mock(return_value=stale)
    monkeypatch.setattr(fed_server, "time", SimpleNamespace(time=lambda: 100.0))
    server.remove_dead_clients()
    server.logout_client.assert_called_once_with("old")
    server.client_manager.remove_client.return_value = stale
    server.tokens = {"old": {}, "new": {}}
    server.notify_dead_client = Mock()
    assert original_logout("old") is stale
    assert server.tokens == {"new": {}}
    server.admin_server.client_dead.assert_called_once_with("old")
    server.notify_dead_client.assert_called_once_with(stale)
