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

from nvflare.fuel.hci.client.oidc import OIDCTokenManager


def _make_manager():
    cfg = {
        "oidc_issuer": "http://127.0.0.1:38080/realms/nvflare",
        "oidc_client_id": "nvflare-admin",
        "oidc_authorization_endpoint": "http://127.0.0.1:38080/auth",
        "oidc_token_endpoint": "http://127.0.0.1:38080/token",
        "oidc_refresh_skew_seconds": 30,
    }
    return OIDCTokenManager(config=cfg)


def test_default_scopes_do_not_request_offline_access():
    manager = OIDCTokenManager(
        config={
            "oidc_issuer": "http://127.0.0.1:38080/realms/nvflare",
            "oidc_client_id": "nvflare-admin",
        }
    )
    assert manager.scopes == "openid profile email"


def _unsigned_jwt(exp: int) -> str:
    return jwt.encode({"sub": "alice", "exp": exp}, key="", algorithm="none")


def test_get_access_token_uses_browser_flow_when_missing_tokens(monkeypatch):
    manager = _make_manager()
    calls = {"browser": 0}

    def _fake_browser():
        calls["browser"] += 1
        now = int(time.time())
        return {
            "access_token": _unsigned_jwt(now + 300),
            "refresh_token": "r1",
            "expires_in": 300,
        }

    monkeypatch.setattr(manager, "_authorize_code_with_browser", _fake_browser)
    token = manager.get_access_token()
    assert token
    assert calls["browser"] == 1

    token2 = manager.get_access_token()
    assert token2 == token
    assert calls["browser"] == 1


def test_get_access_token_uses_refresh_token_before_expiry(monkeypatch):
    manager = _make_manager()
    now = int(time.time())
    manager._access_token = _unsigned_jwt(now + 10)
    manager._access_token_expiry = float(now + 10)
    manager._refresh_token = "r0"

    calls = {"refresh": 0}

    def _fake_refresh(refresh_token: str):
        assert refresh_token == "r0"
        calls["refresh"] += 1
        now2 = int(time.time())
        return {
            "access_token": _unsigned_jwt(now2 + 500),
            "refresh_token": "r2",
            "expires_in": 500,
        }

    monkeypatch.setattr(manager, "_refresh_with_refresh_token", _fake_refresh)
    token = manager.get_access_token()
    assert token
    assert calls["refresh"] == 1
    assert manager._refresh_token == "r2"


def test_get_access_token_falls_back_to_browser_when_refresh_fails(monkeypatch):
    manager = _make_manager()
    now = int(time.time())
    manager._access_token = _unsigned_jwt(now + 5)
    manager._access_token_expiry = float(now + 5)
    manager._refresh_token = "r0"
    warnings = []

    monkeypatch.setattr(
        manager, "_refresh_with_refresh_token", lambda refresh_token: (_ for _ in ()).throw(RuntimeError("bad"))
    )
    monkeypatch.setattr(manager.logger, "warning", lambda msg: warnings.append(msg))
    monkeypatch.setattr(
        manager,
        "_authorize_code_with_browser",
        lambda: {
            "access_token": _unsigned_jwt(int(time.time()) + 600),
            "refresh_token": "r3",
            "expires_in": 600,
        },
    )
    token = manager.get_access_token()
    assert token
    assert manager._refresh_token == "r3"
    assert warnings
    assert "OIDC refresh failed" in warnings[0]


def test_update_tokens_uses_jwt_exp_when_expires_in_missing():
    manager = _make_manager()
    now = int(time.time())
    token = _unsigned_jwt(now + 123)
    manager._update_tokens({"access_token": token, "refresh_token": "r4"})
    assert manager._access_token == token
    assert manager._refresh_token == "r4"
    assert manager._access_token_expiry == pytest.approx(float(now + 123), abs=2.0)


def test_parse_endpoints_from_discovery(monkeypatch):
    manager = OIDCTokenManager(
        config={
            "oidc_issuer": "https://issuer.example",
            "oidc_client_id": "nvflare-admin",
        }
    )
    metadata = {
        "authorization_endpoint": "https://issuer.example/auth",
        "token_endpoint": "https://issuer.example/token",
    }
    monkeypatch.setattr(manager, "_fetch_json", lambda url: json.loads(json.dumps(metadata)))
    endpoints = manager._resolve_endpoints()
    assert endpoints["authorization_endpoint"] == metadata["authorization_endpoint"]
    assert endpoints["token_endpoint"] == metadata["token_endpoint"]


def test_callback_host_must_be_loopback():
    with pytest.raises(ValueError, match="loopback"):
        OIDCTokenManager(
            config={
                "oidc_issuer": "http://127.0.0.1:38080/realms/nvflare",
                "oidc_client_id": "nvflare-admin",
                "oidc_callback_host": "0.0.0.0",
            }
        )


def test_resolve_endpoints_rejects_non_http_schemes():
    manager = OIDCTokenManager(
        config={
            "oidc_issuer": "http://127.0.0.1:38080/realms/nvflare",
            "oidc_client_id": "nvflare-admin",
            "oidc_discovery_url": "file:///tmp/oidc.json",
        }
    )

    with pytest.raises(ValueError, match="http or https"):
        manager._resolve_endpoints()


def test_resolve_endpoints_rejects_remote_http_endpoints(monkeypatch):
    manager = OIDCTokenManager(
        config={
            "oidc_issuer": "https://issuer.example",
            "oidc_client_id": "nvflare-admin",
        }
    )
    metadata = {
        "authorization_endpoint": "http://issuer.example/auth",
        "token_endpoint": "http://issuer.example/token",
    }
    monkeypatch.setattr(manager, "_fetch_json", lambda url: json.loads(json.dumps(metadata)))

    with pytest.raises(ValueError, match="must use https unless the host is loopback"):
        manager._resolve_endpoints()
