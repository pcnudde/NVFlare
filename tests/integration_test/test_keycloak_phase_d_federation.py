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

import json
import os
from pathlib import Path

import pytest

from nvflare.fuel.f3.cellnet.defs import MessageHeaderKey
from nvflare.fuel.f3.message import Message as CellMessage
from nvflare.fuel.hci.conn import Connection
from nvflare.fuel.hci.proto import InternalCommands, ProtoKey
from nvflare.fuel.hci.server.constants import ConnProps
from nvflare.fuel.hci.server.login import LoginModule
from nvflare.fuel.hci.server.sess import Session, SessionManager
from nvflare.fuel.hci.server.token_auth import ClaimMapper, ClaimMappingConfig, TokenValidationConfig, TokenValidator
from tests.integration_test.src.keycloak_harness import KeycloakHarness, KeycloakHarnessError


class _DummyCell:
    def fire_and_forget(self, **kwargs):
        return None


class _DummyIdentityAsserter:
    cert = "dummy-cert"

    def sign(self, data, return_str=True):
        return "signature"


class _DummyIdentityVerifier:
    def verify_common_name(self, **kwargs):
        return True


class _DummyHCI:
    def __init__(self):
        self.id_asserter = _DummyIdentityAsserter()
        self.id_verifier = _DummyIdentityVerifier()

    def get_id_asserter(self):
        return self.id_asserter

    def get_id_verifier(self):
        return self.id_verifier


@pytest.fixture(scope="session")
def keycloak_runtime_phase_d():
    import_dir = Path(__file__).parent / "data" / "keycloak" / "federation"
    harness = KeycloakHarness(
        import_path=import_dir,
        realm="nvflare-broker",
        client_id="nvflare-admin",
        client_secret="nvflare-secret",
        expected_audience="nvflare-admin",
    )
    try:
        runtime = harness.start()
    except KeycloakHarnessError as e:
        if os.environ.get("KEYCLOAK_REQUIRED", "0").lower() in ("1", "true", "yes"):
            raise
        pytest.skip(str(e))
    yield runtime
    harness.stop()


@pytest.fixture()
def login_module_phase_d(keycloak_runtime_phase_d):
    token_validator = TokenValidator(
        TokenValidationConfig(
            issuer=keycloak_runtime_phase_d.issuer,
            audience=keycloak_runtime_phase_d.expected_audience,
            alg_allowlist=("RS256",),
            required_claims=("iss", "aud", "exp", "iat"),
        )
    )
    claim_mapper = ClaimMapper(
        ClaimMappingConfig(
            user_name_claims=("preferred_username", "email"),
            user_org_claim="org",
            user_role_claim="nvf_role",
            groups_claim="groups",
        )
    )
    session_mgr = SessionManager(cell=_DummyCell(), idle_timeout=300, monitor_interval=3600, session_ttl=1800)
    lm = LoginModule(
        sess_mgr=session_mgr,
        token_validator=token_validator,
        claim_mapper=claim_mapper,
        token_jwks=keycloak_runtime_phase_d.get_jwks(),
    )
    yield lm
    session_mgr.shutdown()


def _make_login_conn(token: str, origin="admin@site"):
    hci = _DummyHCI()
    conn = Connection(props={ConnProps.HCI_SERVER: hci})
    conn.set_prop(ConnProps.CMD_HEADERS, {"authorization": f"Bearer {token}"})
    conn.set_prop(ConnProps.REQUEST, CellMessage(headers={MessageHeaderKey.ORIGIN: origin}))
    return conn


def _make_command_conn(session_token: str, origin="admin@site"):
    conn = Connection(props={ConnProps.HCI_SERVER: _DummyHCI()})
    conn.request = {
        ProtoKey.DATA: [
            {ProtoKey.TYPE: ProtoKey.COMMAND, ProtoKey.DATA: "list_jobs"},
            {ProtoKey.TYPE: ProtoKey.TOKEN, ProtoKey.DATA: session_token},
        ]
    }
    conn.set_prop(ConnProps.REQUEST, CellMessage(headers={MessageHeaderKey.ORIGIN: origin}))
    return conn


def _read_items(conn: Connection):
    payload = conn.close()
    proto = json.loads(payload)
    return proto.get(ProtoKey.DATA, [])


def _extract_string(items):
    for item in items:
        if item.get(ProtoKey.TYPE) == ProtoKey.STRING:
            return item.get(ProtoKey.DATA)
    return None


def _extract_token(items):
    for item in items:
        if item.get(ProtoKey.TYPE) == ProtoKey.TOKEN:
            return item.get(ProtoKey.DATA)
    return None


def test_federation_org_a_user_through_broker(keycloak_runtime_phase_d, login_module_phase_d):
    broker_token = keycloak_runtime_phase_d.request_password_token(username="alice-broker", password="alicebrokerpass")
    conn = _make_login_conn(broker_token)
    login_module_phase_d.handle_token_login(conn, [InternalCommands.TOKEN_LOGIN])
    items = _read_items(conn)
    assert _extract_string(items) == "OK"
    session_token = _extract_token(items)
    assert session_token
    session = Session.decode_token(session_token)
    assert session.user_org == "org_a"
    assert session.user_role == "lead"


def test_federation_org_b_user_through_broker(keycloak_runtime_phase_d, login_module_phase_d):
    broker_token = keycloak_runtime_phase_d.request_password_token(username="bob-broker", password="bobbrokerpass")
    conn = _make_login_conn(broker_token)
    login_module_phase_d.handle_token_login(conn, [InternalCommands.TOKEN_LOGIN])
    items = _read_items(conn)
    assert _extract_string(items) == "OK"
    session_token = _extract_token(items)
    assert session_token
    session = Session.decode_token(session_token)
    assert session.user_org == "org_b"
    assert session.user_role == "org_admin"


def test_federation_org_contexts_remain_distinct(keycloak_runtime_phase_d, login_module_phase_d):
    broker_token_a = keycloak_runtime_phase_d.request_password_token(username="alice-broker", password="alicebrokerpass")
    broker_token_b = keycloak_runtime_phase_d.request_password_token(username="bob-broker", password="bobbrokerpass")

    login_conn_a = _make_login_conn(broker_token_a)
    login_conn_b = _make_login_conn(broker_token_b)
    login_module_phase_d.handle_token_login(login_conn_a, [InternalCommands.TOKEN_LOGIN])
    login_module_phase_d.handle_token_login(login_conn_b, [InternalCommands.TOKEN_LOGIN])
    session_token_a = _extract_token(_read_items(login_conn_a))
    session_token_b = _extract_token(_read_items(login_conn_b))
    assert session_token_a and session_token_b

    cmd_conn_a = _make_command_conn(session_token_a)
    cmd_conn_b = _make_command_conn(session_token_b)
    assert login_module_phase_d.pre_command(cmd_conn_a, ["list_jobs"])
    assert login_module_phase_d.pre_command(cmd_conn_b, ["list_jobs"])
    assert cmd_conn_a.get_prop(ConnProps.USER_ORG) == "org_a"
    assert cmd_conn_b.get_prop(ConnProps.USER_ORG) == "org_b"
    assert cmd_conn_a.get_prop(ConnProps.USER_ORG) != cmd_conn_b.get_prop(ConnProps.USER_ORG)


def test_federation_rejects_invalid_broker_credentials(keycloak_runtime_phase_d):
    with pytest.raises(KeycloakHarnessError, match="failed to obtain token"):
        keycloak_runtime_phase_d.request_password_token(username="alice-broker", password="wrong-password")
