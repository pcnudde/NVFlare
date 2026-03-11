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
from nvflare.private.defs import CellChannel, CellChannelTopic
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


def test_build_admin_token_login_kwargs_enforces_minimum_required_claim_floor():
    conf = _make_server_conf({"required_claims": ["iss", "aud"]})

    kwargs = build_admin_token_login_kwargs(server_conf=conf, workspace_dir="/tmp")

    assert kwargs["token_validator"].config.required_claims == ("iss", "aud", "exp", "iat")


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

    client = object()
    pyjwk_client = Mock(return_value=client)
    monkeypatch.setattr("nvflare.private.fed.app.utils.jwt.PyJWKClient", pyjwk_client)

    kwargs = build_admin_token_login_kwargs(server_conf=conf, workspace_dir="/tmp")
    assert "jwks_fetcher" in kwargs
    assert "token_jwks" not in kwargs

    assert kwargs["jwks_fetcher"]() is client
    assert kwargs["jwks_fetcher"]() is client
    pyjwk_client.assert_called_once_with(
        "https://id.example.com/protocol/openid-connect/certs",
        cache_jwk_set=True,
        lifespan=300,
        timeout=5.0,
    )


def test_build_admin_token_login_kwargs_jwks_fetcher_respects_zero_cache_ttl(monkeypatch):
    conf = _make_server_conf(
        {
            "jwks_uri": "https://id.example.com/protocol/openid-connect/certs",
            "jwks_cache_ttl_seconds": 0,
        }
    )
    del conf["admin_auth"]["token_login"]["jwks"]

    client = object()
    pyjwk_client = Mock(return_value=client)
    monkeypatch.setattr("nvflare.private.fed.app.utils.jwt.PyJWKClient", pyjwk_client)

    kwargs = build_admin_token_login_kwargs(server_conf=conf, workspace_dir="/tmp")
    assert kwargs["jwks_fetcher"]() is client
    pyjwk_client.assert_called_once_with(
        "https://id.example.com/protocol/openid-connect/certs",
        cache_jwk_set=False,
        lifespan=300,
        timeout=5.0,
    )


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


def test_create_admin_server_uses_resource_override_for_admin_connection_security(tmp_path):
    fl_server = Mock()
    fl_server.cell = object()
    fl_server.engine = object()
    fl_server.cmd_modules = []
    args = Namespace(workspace=str(tmp_path))
    conf = {
        "service": {"target": "server:8002", "scheme": "http"},
        "admin_port": 8003,
        "connection_security": "mtls",
        "ssl_root_cert": "rootCA.pem",
        "ssl_cert": "server.crt",
        "ssl_private_key": "server.key",
    }

    with patch(
        "nvflare.private.fed.app.utils.ConfigService._get_from_config",
        return_value=[{"admin_connection_security": "tls"}],
    ):
        with patch("nvflare.private.fed.app.utils.Cell") as cell_ctor:
            cell_instance = Mock()
            cell_ctor.return_value = cell_instance
            fed_admin_server = Mock()
            with patch("nvflare.private.fed.app.utils.FedAdminServer", fed_admin_server):
                create_admin_server(fl_server=fl_server, server_conf=conf, args=args)

    _, cell_kwargs = cell_ctor.call_args
    assert cell_kwargs["credentials"]["connection_security"] == "tls"
    assert fed_admin_server.call_args.kwargs["cell"] == cell_instance


def test_create_admin_server_registers_challenge_handler_on_separate_admin_cell(tmp_path):
    fl_server = Mock()
    fl_server.cell = object()
    fl_server.engine = object()
    fl_server.cmd_modules = []
    args = Namespace(workspace=str(tmp_path))
    conf = {
        "service": {"target": "server:8002", "scheme": "http"},
        "admin_port": 8003,
        "connection_security": "mtls",
        "ssl_root_cert": "rootCA.pem",
        "ssl_cert": "server.crt",
        "ssl_private_key": "server.key",
    }

    with patch("nvflare.private.fed.app.utils.Cell") as cell_ctor:
        cell_instance = Mock()
        cell_ctor.return_value = cell_instance
        with patch("nvflare.private.fed.app.utils.FedAdminServer"):
            create_admin_server(fl_server=fl_server, server_conf=conf, args=args)

    challenge_calls = []
    for call in cell_instance.register_request_cb.call_args_list:
        channel = call.kwargs.get("channel")
        topic = call.kwargs.get("topic")
        cb = call.kwargs.get("cb")
        if len(call.args) >= 3:
            channel = call.args[0]
            topic = call.args[1]
            cb = call.args[2]
        if channel == CellChannel.SERVER_MAIN and topic == CellChannelTopic.Challenge:
            challenge_calls.append(cb)

    assert challenge_calls == [fl_server.client_challenge]
