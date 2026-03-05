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
import time

import jwt
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa

from nvflare.fuel.f3.cellnet.defs import MessageHeaderKey
from nvflare.fuel.f3.message import Message as CellMessage
from nvflare.fuel.hci.conn import Connection
from nvflare.fuel.hci.proto import InternalCommands, ProtoKey
from nvflare.fuel.hci.security import IdentityKey
from nvflare.fuel.hci.server.constants import ConnProps
from nvflare.fuel.hci.server.login import LoginModule
from nvflare.fuel.hci.server.sess import Session, SessionManager
from nvflare.fuel.hci.server.token_auth import ClaimMapper, ClaimMappingConfig, TokenValidationConfig, TokenValidator

ISSUER = "https://kc.example.com/realms/nvflare"
AUDIENCE = "nvflare-admin"


class _DummyCell:
    def fire_and_forget(self, **kwargs):
        return None


class _DummyIdentityAsserter:
    cert = "dummy-cert"

    def sign(self, data, return_str=True):
        return "signature"


class _DummyIdentityVerifier:
    def __init__(self, verify_result=True):
        self.verify_result = verify_result

    def verify_common_name(self, **kwargs):
        return self.verify_result


class _DummyHCI:
    def __init__(self, verify_result=True):
        self.id_asserter = _DummyIdentityAsserter()
        self.id_verifier = _DummyIdentityVerifier(verify_result=verify_result)

    def get_id_asserter(self):
        return self.id_asserter

    def get_id_verifier(self):
        return self.id_verifier


@pytest.fixture()
def session_mgr():
    mgr = SessionManager(cell=_DummyCell(), idle_timeout=3600, monitor_interval=3600, session_ttl=7200)
    yield mgr
    mgr.shutdown()


@pytest.fixture()
def signing_material():
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    private_key_pem = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    ).decode("ascii")
    public_jwk = json.loads(jwt.algorithms.RSAAlgorithm.to_jwk(private_key.public_key()))
    public_jwk["kid"] = "primary-kid"
    public_jwk["alg"] = "RS256"
    public_jwk["use"] = "sig"
    return private_key_pem, {"keys": [public_jwk]}


@pytest.fixture()
def token_login_module(session_mgr, signing_material):
    _, jwks = signing_material
    token_validator = TokenValidator(TokenValidationConfig(issuer=ISSUER, audience=AUDIENCE, alg_allowlist=["RS256"]))
    claim_mapper = ClaimMapper(
        ClaimMappingConfig(
            user_name_claims=("preferred_username", "email"),
            user_org_claim="org",
            user_role_claim="nvf_role",
            groups_claim="groups",
        )
    )
    return LoginModule(
        sess_mgr=session_mgr,
        token_validator=token_validator,
        claim_mapper=claim_mapper,
        token_jwks=jwks,
    )


def _make_conn(hci, cmd_headers=None, origin="admin@site"):
    conn = Connection(props={ConnProps.HCI_SERVER: hci})
    conn.set_prop(ConnProps.CMD_HEADERS, cmd_headers or {})
    conn.set_prop(ConnProps.REQUEST, CellMessage(headers={MessageHeaderKey.ORIGIN: origin}))
    return conn


def _make_command_conn(hci, token, origin="admin@site"):
    conn = Connection(props={ConnProps.HCI_SERVER: hci})
    conn.request = {
        ProtoKey.DATA: [
            {ProtoKey.TYPE: ProtoKey.COMMAND, ProtoKey.DATA: "list_jobs"},
            {ProtoKey.TYPE: ProtoKey.TOKEN, ProtoKey.DATA: token},
        ]
    }
    conn.set_prop(ConnProps.REQUEST, CellMessage(headers={MessageHeaderKey.ORIGIN: origin}))
    return conn


def _read_conn_items(conn):
    payload = conn.close()
    proto = json.loads(payload)
    return proto.get(ProtoKey.DATA, [])


def _extract_first_string(items):
    for item in items:
        if item.get(ProtoKey.TYPE) == ProtoKey.STRING:
            return item.get(ProtoKey.DATA)
    return None


def _extract_token(items):
    for item in items:
        if item.get(ProtoKey.TYPE) == ProtoKey.TOKEN:
            return item.get(ProtoKey.DATA)
    return None


def _make_token(private_key_pem, now, **overrides):
    claims = {
        "sub": "alice-sub",
        "preferred_username": "alice",
        "org": "org_a",
        "nvf_role": "lead",
        "iss": ISSUER,
        "aud": AUDIENCE,
        "iat": now - 1,
        "nbf": now - 1,
        "exp": now + 300,
    }
    claims.update(overrides)
    token = jwt.encode(claims, private_key_pem, algorithm="RS256", headers={"kid": "primary-kid"})
    return token, claims


def test_token_login_success(token_login_module, signing_material):
    private_key_pem, _ = signing_material
    now = int(time.time())
    hci = _DummyHCI()
    jwt_token, claims = _make_token(private_key_pem, now)
    conn = _make_conn(hci=hci, cmd_headers={"authorization": f"Bearer {jwt_token}"})

    token_login_module.handle_token_login(conn, [InternalCommands.TOKEN_LOGIN])

    items = _read_conn_items(conn)
    assert _extract_first_string(items) == "OK"
    session_token = _extract_token(items)
    assert session_token

    sess = Session.decode_token(session_token)
    assert sess.user_name == "alice"
    assert sess.user_org == "org_a"
    assert sess.user_role == "lead"
    assert sess.token_expiry_time == claims["exp"]
    assert sess.auth_source == "token"


def test_token_login_rejects_invalid_token(token_login_module, signing_material):
    private_key_pem, _ = signing_material
    now = int(time.time())
    hci = _DummyHCI()
    jwt_token, _ = _make_token(private_key_pem, now, aud="wrong-audience")
    conn = _make_conn(hci=hci, cmd_headers={"token": jwt_token})

    token_login_module.handle_token_login(conn, [InternalCommands.TOKEN_LOGIN])

    items = _read_conn_items(conn)
    assert _extract_first_string(items) == "REJECT"
    assert _extract_token(items) is None
    assert token_login_module.session_mgr.get_sessions() == []


def test_cert_login_compatibility(monkeypatch, session_mgr):
    hci = _DummyHCI(verify_result=True)
    login_module = LoginModule(sess_mgr=session_mgr)
    conn = _make_conn(hci=hci, cmd_headers={"cert": "dummy-cert-bytes", "signature": "signed-cn"})

    monkeypatch.setattr("nvflare.fuel.hci.server.login.load_crt_bytes", lambda _: "cert-object")
    monkeypatch.setattr("nvflare.fuel.hci.server.login.cert_to_dict", lambda _: {"subject": "dummy"})
    monkeypatch.setattr(
        "nvflare.fuel.hci.server.login.get_identity_info",
        lambda _: {IdentityKey.ORG: "org_a", IdentityKey.ROLE: "lead"},
    )

    login_module.handle_cert_login(conn, [InternalCommands.CERT_LOGIN, "alice"])

    items = _read_conn_items(conn)
    assert _extract_first_string(items) == "OK"
    session_token = _extract_token(items)
    assert session_token
    sess = Session.decode_token(session_token)
    assert sess.user_name == "alice"
    assert sess.user_org == "org_a"
    assert sess.user_role == "lead"
    assert sess.auth_source == "cert"


def test_pre_command_populates_same_authz_context_for_cert_and_token(token_login_module, signing_material, monkeypatch):
    private_key_pem, _ = signing_material
    now = int(time.time())
    hci = _DummyHCI(verify_result=True)

    cert_conn = _make_conn(hci=hci, cmd_headers={"cert": "dummy-cert-bytes", "signature": "signed-cn"})
    monkeypatch.setattr("nvflare.fuel.hci.server.login.load_crt_bytes", lambda _: "cert-object")
    monkeypatch.setattr("nvflare.fuel.hci.server.login.cert_to_dict", lambda _: {"subject": "dummy"})
    monkeypatch.setattr(
        "nvflare.fuel.hci.server.login.get_identity_info",
        lambda _: {IdentityKey.ORG: "org_a", IdentityKey.ROLE: "lead"},
    )
    token_login_module.handle_cert_login(cert_conn, [InternalCommands.CERT_LOGIN, "alice"])
    cert_token = _extract_token(_read_conn_items(cert_conn))

    jwt_token, _ = _make_token(private_key_pem, now)
    token_conn = _make_conn(hci=hci, cmd_headers={"authorization": f"Bearer {jwt_token}"})
    token_login_module.handle_token_login(token_conn, [InternalCommands.TOKEN_LOGIN])
    token_session_token = _extract_token(_read_conn_items(token_conn))

    cert_cmd_conn = _make_command_conn(hci=hci, token=cert_token)
    token_cmd_conn = _make_command_conn(hci=hci, token=token_session_token)

    assert token_login_module.pre_command(cert_cmd_conn, ["list_jobs"])
    assert token_login_module.pre_command(token_cmd_conn, ["list_jobs"])

    assert cert_cmd_conn.get_prop(ConnProps.USER_NAME) == token_cmd_conn.get_prop(ConnProps.USER_NAME) == "alice"
    assert cert_cmd_conn.get_prop(ConnProps.USER_ORG) == token_cmd_conn.get_prop(ConnProps.USER_ORG) == "org_a"
    assert cert_cmd_conn.get_prop(ConnProps.USER_ROLE) == token_cmd_conn.get_prop(ConnProps.USER_ROLE) == "lead"
    assert cert_cmd_conn.get_prop(ConnProps.AUTH_SOURCE) == "cert"
    assert token_cmd_conn.get_prop(ConnProps.AUTH_SOURCE) == "token"


def test_pre_command_allows_token_login_without_existing_session(token_login_module):
    conn = Connection()
    assert token_login_module.pre_command(conn, [InternalCommands.TOKEN_LOGIN])


def test_pre_command_rejects_expired_token_after_session_recreate_attempt(
    token_login_module, signing_material, monkeypatch
):
    private_key_pem, _ = signing_material
    now = int(time.time())
    hci = _DummyHCI()
    jwt_token, claims = _make_token(private_key_pem, now, exp=now + 2)
    login_conn = _make_conn(hci=hci, cmd_headers={"authorization": f"Bearer {jwt_token}"})
    token_login_module.handle_token_login(login_conn, [InternalCommands.TOKEN_LOGIN])
    session_token = _extract_token(_read_conn_items(login_conn))
    assert session_token

    # Simulate session cache loss (server restart) so pre_command tries recreate_session().
    token_login_module.session_mgr.sessions = {}

    monkeypatch.setattr("nvflare.fuel.hci.server.sess.time.time", lambda: claims["exp"] + 1)
    expired_cmd_conn = _make_command_conn(hci=hci, token=session_token)
    assert not token_login_module.pre_command(expired_cmd_conn, ["list_jobs"])
    items = _read_conn_items(expired_cmd_conn)
    assert any(item.get(ProtoKey.TYPE) == ProtoKey.ERROR and item.get(ProtoKey.DATA) == "session_inactive" for item in items)
