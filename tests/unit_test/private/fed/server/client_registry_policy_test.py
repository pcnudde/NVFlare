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

from nvflare.apis.client import Client
from nvflare.apis.fl_constant import FLContextKey, ReservedKey
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.fuel.f3.cellnet.defs import IdentityChallengeKey, MessageHeaderKey
from nvflare.private.defs import CellMessageHeaderKeys, ClientRegSession, ClientType, InternalFLContextKey
from nvflare.private.fed.server import client_manager as subject
from tests.timing_utils import isolate_time


def request(name="site-a", payload=None, project="project", token="token-a"):
    headers = {
        CellMessageHeaderKeys.CLIENT_NAME: name,
        CellMessageHeaderKeys.PROJECT_NAME: project,
        CellMessageHeaderKeys.TOKEN: token,
        CellMessageHeaderKeys.CLIENT_IP: "127.0.0.1",
        CellMessageHeaderKeys.CLIENT_TYPE: ClientType.REGULAR,
        MessageHeaderKey.ORIGIN: "root.site-a",
    }
    return Mock(payload=Shareable() if payload is None else payload, get_header=lambda key: headers.get(key))


@pytest.fixture
def manager():
    return subject.ClientManager("project", min_num_clients=1, max_num_clients=1)


def test_authenticate_records_regular_client_and_remove_clears_both_indexes(manager):
    client = manager.authenticate(request(), FLContext())
    assert manager.get_clients() == {client.token: client}
    assert manager.get_client_from_name("site-a") is client
    assert manager.get_min_clients() == manager.get_max_clients() == 1
    assert manager.remove_client(client.token) is client
    assert manager.get_clients() == {}
    assert manager.name_to_clients == {}
    assert manager.remove_client("unknown") is None


def test_project_mismatch_is_rejected_before_login(manager):
    ctx = FLContext()
    assert manager.authenticate(request(project="other"), ctx) is None
    assert "task does not match" in ctx.get_prop(FLContextKey.UNAUTHENTICATED)
    assert manager.clients == {}


@pytest.mark.parametrize(
    "token,project,allow_new,error",
    [
        (None, "project", False, "Could not read"),
        ("unknown", "other", False, "task does not match"),
        ("unknown", "project", False, "Unknown client"),
    ],
)
def test_validation_reports_precise_identity_failure(manager, token, project, allow_new, error):
    ctx = FLContext()
    assert manager.validate_client(request(token=token, project=project), ctx, allow_new) is None
    assert error in ctx.get_prop(FLContextKey.UNAUTHENTICATED)


def test_validation_requires_known_token_unless_new_clients_are_allowed(manager):
    client = Client("site-a", "token-a")
    manager.set_clients({client.token: client})
    ctx = FLContext()
    assert manager.validate_client(request(), ctx) is client
    assert manager.validate_client(request(token="new"), ctx, allow_new=True) is None
    assert ctx.get_prop(FLContextKey.UNAUTHENTICATED) is None
    assert manager.get_all_clients_from_inputs([client.token, client.name, "missing"]) == (
        [client, client],
        ["missing"],
    )


@pytest.mark.parametrize("payload,secure,expected", [("bad", False, "payload"), (Shareable(), True, "cert")])
def test_registration_rejects_malformed_identity_material(manager, payload, secure, expected):
    ctx = FLContext()
    ctx.set_prop(FLContextKey.SECURE_MODE, secure)
    manager.logger = Mock()
    assert manager.authenticated_client(request(payload=payload), ctx, ClientType.REGULAR) is None
    assert expected in manager.logger.error.call_args.args[0]
    assert manager.clients == {}


@pytest.mark.parametrize("failure", ["signature", "missing-session", "wrong-session", "verify"])
def test_secure_registration_checks_signature_session_and_verification(manager, monkeypatch, failure):
    payload = Shareable()
    payload[IdentityChallengeKey.CERT] = b"cert"
    if failure != "signature":
        payload[IdentityChallengeKey.SIGNATURE] = b"signature"
    ctx = FLContext()
    ctx.set_prop(FLContextKey.SECURE_MODE, True)
    if failure in ("wrong-session", "verify"):
        ctx.set_prop(
            InternalFLContextKey.CLIENT_REG_SESSION, "bad" if failure == "wrong-session" else ClientRegSession("site-a")
        )
    verifier = Mock()
    verifier.verify_common_name.side_effect = ValueError("untrusted identity")
    monkeypatch.setattr(subject, "load_crt_chain_bytes", Mock(return_value=[object()]))
    monkeypatch.setattr(manager, "_get_id_verifier", Mock(return_value=verifier))
    assert manager.authenticated_client(request(payload=payload), ctx, ClientType.REGULAR) is None
    assert verifier.verify_common_name.call_count == (1 if failure == "verify" else 0)
    assert manager.clients == {}


def test_regular_capacity_limit_does_not_apply_to_relays(manager):
    existing = Client("site-existing", "token-old")
    manager.set_clients({existing.token: existing})
    ctx = FLContext()
    assert manager.authenticated_client(request(), ctx, ClientType.REGULAR) is None
    assert ctx.get_prop(FLContextKey.UNAUTHENTICATED) == "Maximum number of clients reached"
    relay = manager.authenticated_client(request(name="relay"), FLContext(), ClientType.RELAY)
    assert relay.name == "relay"
    assert manager.has_relays()
    assert manager.num_relays == 1
    assert manager.clients == {existing.token: existing}


def test_relogin_replaces_stale_token_and_peer_properties(manager, monkeypatch):
    old = Client("site-a", "stale")
    manager.set_clients({old.token: old})
    ctx = FLContext()
    peer = FLContext()
    peer.set_prop(ReservedKey.FQSN, "org.site-a")
    peer.set_prop(ReservedKey.IS_LEAF, True)
    ctx.set_peer_context(peer)
    ctx.set_prop(FLContextKey.CLIENT_SITE_CONFIG, {"role": "trainer"})
    clock = isolate_time(monkeypatch, subject)
    monkeypatch.setattr(clock, "time", lambda: 123)
    client = manager.authenticate(request(), ctx)
    assert client.token != old.token
    assert manager.clients == {client.token: client}
    assert client.get_fqcn() == "root.site-a"
    assert client.get_fqsn() == "org.site-a"
    assert client.get_is_leaf() is True
    assert client.get_site_config() == {"role": "trainer"}
    assert client.last_connect_time == 123


def test_heartbeat_updates_known_client_and_rejects_duplicate_name(manager, monkeypatch):
    client = Client("site-a", "token-a")
    manager.set_clients({client.token: client})
    clock = isolate_time(monkeypatch, subject)
    monkeypatch.setattr(clock, "time", lambda: 456)
    assert not manager.heartbeat(client.token, client.name, "root.site-a", FLContext())
    assert client.last_connect_time == 456
    ctx = FLContext()
    assert not manager.heartbeat("other-token", client.name, "root.site-a", ctx)
    assert "already registered" in ctx.get_prop(FLContextKey.COMMUNICATION_ERROR)
    assert manager.clients == {client.token: client}


def test_heartbeat_reactivates_an_unknown_client(manager):
    assert manager.heartbeat("new", "site-a", "root.site-a", FLContext())
    client = manager.clients["new"]
    assert manager.name_to_clients["site-a"] is client
    assert client.get_fqcn() == "root.site-a"


def test_enable_persists_policy_and_rolls_back_on_failure(manager, monkeypatch, tmp_path):
    manager.set_disabled_clients_file(str(tmp_path / "disabled.json"))
    manager.disable_client("site-a")
    assert manager.enable_client("site-a")
    assert not manager.is_client_disabled("site-a")
    assert not manager.enable_client("site-a")
    manager.disable_client("site-a")
    monkeypatch.setattr(manager, "_save_disabled_clients", Mock(side_effect=OSError("full")))
    with pytest.raises(OSError, match="full"):
        manager.enable_client("site-a")
    assert manager.is_client_disabled("site-a")


def test_credential_verifier_is_obtained_for_the_current_context(manager):
    ctx = FLContext()
    manager.cred_keeper = Mock()
    verifier = manager._get_id_verifier(ctx)
    manager.cred_keeper.get_id_verifier.assert_called_once_with(ctx)
    assert verifier is manager.cred_keeper.get_id_verifier.return_value


def test_admin_lookup_constructs_ephemeral_identity_without_registering(manager, monkeypatch):
    monkeypatch.setattr(subject, "is_valid_admin_client_name", lambda value: value == "admin-user")
    client = manager.get_client_from_name("admin-user")
    assert client.name == "admin-user"
    assert client.token is None
    assert client.get_fqcn() == "admin-user"
    assert manager.get_client_from_name("unknown") is None
    assert manager.clients == manager.name_to_clients == {}
