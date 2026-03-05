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
from argparse import Namespace
from unittest.mock import Mock, patch

import pytest

from nvflare.fuel.hci.server.token_auth import ClaimMapper, TokenValidator
from nvflare.private.fed.app.utils import build_admin_token_login_kwargs, create_admin_server


def _make_server_conf(token_login_overrides=None):
    token_login = {
        "issuer": "https://id.example.com/realms/nvflare",
        "audience": "nvflare-admin",
        "jwks": {"keys": [{"kid": "test-kid"}]},
        "claim_mappings": {
            "user_name_claims": ["preferred_username", "email"],
            "user_org_claim": "org",
            "user_role_claim": "nvf_role",
        },
    }
    if token_login_overrides:
        token_login.update(token_login_overrides)

    return {
        "admin_storage": "transfer",
        "download_job_url": "http://download.server.com/",
        "admin_timeout": 10.0,
        "admin_auth": {
            "token_login": token_login,
        },
    }


def test_build_admin_token_login_kwargs_ignores_missing_config():
    assert build_admin_token_login_kwargs(server_conf=None, workspace_dir="/tmp") == {}
    assert build_admin_token_login_kwargs(server_conf={}, workspace_dir="/tmp") == {}


def test_build_admin_token_login_kwargs_builds_components_from_inline_jwks():
    conf = _make_server_conf()

    kwargs = build_admin_token_login_kwargs(server_conf=conf, workspace_dir="/tmp")

    assert isinstance(kwargs["token_validator"], TokenValidator)
    assert kwargs["token_validator"].config.issuer == "https://id.example.com/realms/nvflare"
    assert kwargs["token_validator"].config.audience == "nvflare-admin"
    assert isinstance(kwargs["claim_mapper"], ClaimMapper)
    assert kwargs["claim_mapper"].config.user_org_claim == "org"
    assert kwargs["token_jwks"] == {"keys": [{"kid": "test-kid"}]}


def test_build_admin_token_login_kwargs_loads_jwks_from_workspace_relative_file(tmp_path):
    jwks = {"keys": [{"kid": "from-file"}]}
    jwks_file = tmp_path / "jwks.json"
    jwks_file.write_text(json.dumps(jwks))
    conf = _make_server_conf({"jwks_file": "jwks.json"})
    del conf["admin_auth"]["token_login"]["jwks"]

    kwargs = build_admin_token_login_kwargs(server_conf=conf, workspace_dir=str(tmp_path))

    assert kwargs["token_jwks"] == jwks


def test_build_admin_token_login_kwargs_builds_cached_jwks_fetcher(monkeypatch):
    conf = _make_server_conf({"jwks_uri": "https://id.example.com/protocol/openid-connect/certs"})
    del conf["admin_auth"]["token_login"]["jwks"]

    calls = {"count": 0}
    fetched_jwks = {"keys": [{"kid": "remote"}]}

    def _fake_fetch(url, timeout):
        calls["count"] += 1
        assert url == "https://id.example.com/protocol/openid-connect/certs"
        assert timeout == 5.0
        return fetched_jwks

    monkeypatch.setattr("nvflare.private.fed.app.utils._fetch_json_from_url", _fake_fetch)

    kwargs = build_admin_token_login_kwargs(server_conf=conf, workspace_dir="/tmp")
    assert "jwks_fetcher" in kwargs
    assert "token_jwks" not in kwargs

    assert kwargs["jwks_fetcher"]() == fetched_jwks
    assert kwargs["jwks_fetcher"]() == fetched_jwks
    assert calls["count"] == 1


def test_build_admin_token_login_kwargs_jwks_fetcher_respects_zero_cache_ttl(monkeypatch):
    conf = _make_server_conf(
        {
            "jwks_uri": "https://id.example.com/protocol/openid-connect/certs",
            "jwks_cache_ttl_seconds": 0,
        }
    )
    del conf["admin_auth"]["token_login"]["jwks"]

    calls = {"count": 0}

    def _fake_fetch(url, timeout):
        calls["count"] += 1
        return {"keys": [{"kid": f"remote-{calls['count']}"}]}

    monkeypatch.setattr("nvflare.private.fed.app.utils._fetch_json_from_url", _fake_fetch)

    kwargs = build_admin_token_login_kwargs(server_conf=conf, workspace_dir="/tmp")
    first = kwargs["jwks_fetcher"]()
    second = kwargs["jwks_fetcher"]()
    assert first != second
    assert calls["count"] == 2


def test_build_admin_token_login_kwargs_rejects_multiple_jwks_sources():
    conf = _make_server_conf({"jwks_uri": "https://id.example.com/certs"})

    with pytest.raises(ValueError, match="only one JWKS source"):
        build_admin_token_login_kwargs(server_conf=conf, workspace_dir="/tmp")


def test_build_admin_token_login_kwargs_rejects_missing_jwks_source():
    conf = _make_server_conf()
    del conf["admin_auth"]["token_login"]["jwks"]

    with pytest.raises(ValueError, match="requires 'jwks', 'jwks_file', 'jwks_uri', or 'discovery_url'"):
        build_admin_token_login_kwargs(server_conf=conf, workspace_dir="/tmp")


def test_create_admin_server_passes_token_login_components(tmp_path):
    fl_server = Mock()
    fl_server.cell = object()
    fl_server.engine = object()
    fl_server.cmd_modules = []
    args = Namespace(workspace=str(tmp_path))
    conf = _make_server_conf()

    with patch("nvflare.private.fed.app.utils.FedAdminServer") as fed_admin_server:
        create_admin_server(fl_server=fl_server, server_conf=conf, args=args)

    kwargs = fed_admin_server.call_args.kwargs
    assert isinstance(kwargs["token_validator"], TokenValidator)
    assert isinstance(kwargs["claim_mapper"], ClaimMapper)
    assert kwargs["token_jwks"] == {"keys": [{"kid": "test-kid"}]}
