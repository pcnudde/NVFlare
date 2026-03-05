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
def keycloak_runtime():
    realm_file = Path(__file__).parent / "data" / "keycloak" / "nvflare_realm_phase_c.json"
    harness = KeycloakHarness(import_path=realm_file)
    try:
        runtime = harness.start()
    except KeycloakHarnessError as e:
        if os.environ.get("KEYCLOAK_REQUIRED", "0").lower() in ("1", "true", "yes"):
            raise
        pytest.skip(str(e))
    yield runtime
    harness.stop()


@pytest.fixture()
def login_module(keycloak_runtime):
    token_validator = TokenValidator(
        TokenValidationConfig(
            issuer=keycloak_runtime.issuer,
            audience=keycloak_runtime.expected_audience,
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
        token_jwks=keycloak_runtime.get_jwks(),
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


def test_keycloak_token_login_success(keycloak_runtime, login_module):
    access_token = keycloak_runtime.request_password_token(username="alice", password="alicepass")
    conn = _make_login_conn(access_token)

    login_module.handle_token_login(conn, [InternalCommands.TOKEN_LOGIN])

    items = _read_items(conn)
    assert _extract_string(items) == "OK"
    session_token = _extract_token(items)
    assert isinstance(session_token, str) and session_token

    session = Session.decode_token(session_token)
    assert session.user_name == "alice"
    assert session.user_org == "org_a"
    assert session.user_role == "lead"


def test_keycloak_token_login_rejects_missing_role_claim(keycloak_runtime, login_module):
    access_token = keycloak_runtime.request_password_token(username="no_role", password="norolepass")
    conn = _make_login_conn(access_token)

    login_module.handle_token_login(conn, [InternalCommands.TOKEN_LOGIN])

    items = _read_items(conn)
    assert _extract_string(items) == "REJECT"
    assert _extract_token(items) is None


def test_keycloak_token_login_sets_conn_props_for_authz(keycloak_runtime, login_module):
    access_token = keycloak_runtime.request_password_token(username="alice", password="alicepass")
    login_conn = _make_login_conn(access_token)
    login_module.handle_token_login(login_conn, [InternalCommands.TOKEN_LOGIN])
    session_token = _extract_token(_read_items(login_conn))
    assert session_token

    cmd_conn = _make_command_conn(session_token)
    assert login_module.pre_command(cmd_conn, ["list_jobs"])
    assert cmd_conn.get_prop(ConnProps.USER_NAME) == "alice"
    assert cmd_conn.get_prop(ConnProps.USER_ORG) == "org_a"
    assert cmd_conn.get_prop(ConnProps.USER_ROLE) == "lead"
