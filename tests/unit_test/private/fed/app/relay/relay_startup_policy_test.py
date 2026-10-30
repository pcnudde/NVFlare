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
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from nvflare.apis.fl_constant import ConnectionSecurity, ConnPropKey
from nvflare.private.fed.app.relay import relay


@pytest.fixture
def startup(monkeypatch, tmp_path):
    (tmp_path / "startup").mkdir()
    (tmp_path / "local").mkdir()
    config = {
        "project_name": "project",
        "server_identity": "server.example",
        "identity": "relay-1",
        "connect_to": {ConnPropKey.ADDRESS: "host:8004", ConnPropKey.SCHEME: "tcp", ConnPropKey.FQCN: "server"},
    }
    args = SimpleNamespace(workspace=str(tmp_path), relay_config="relay.json", set=[])
    path = tmp_path / "startup" / "relay.json"
    cell, auth = Mock(), Mock(client_name="relay-1")
    auth.authenticate.return_value = ("token", "signature", "session", Mock(spec=relay.TokenVerifier))
    factory = Mock(return_value=cell)
    monkeypatch.setattr(relay, "Cell", factory)
    callbacks = []

    def create_agent(cell, agent_closed_cb):
        callbacks.append(agent_closed_cb)

    monkeypatch.setattr(relay, "NetAgent", Mock(side_effect=create_agent))
    # Model a transport closure through its registered callback without starting
    # a cell, listener, socket, or background thread.
    cell.start.side_effect = lambda: callbacks[0]()
    monkeypatch.setattr(relay, "Authenticator", Mock(return_value=auth))
    monkeypatch.setattr(relay, "configure_logging", Mock())
    monkeypatch.setattr(relay.ConfigService, "initialize", Mock())
    monkeypatch.setattr(relay, "search_file", Mock(return_value="/kit/rootCA.pem"))
    monkeypatch.setattr(relay, "enhance_credential_info", Mock())
    monkeypatch.setattr(relay.flare_decomposers, "register", Mock())
    return SimpleNamespace(config=config, args=args, path=path, cell=cell, auth=auth, factory=factory)


@pytest.mark.parametrize("parent,secure", [("server", False), ("relay-0", False), ("server", True), ("relay-0", True)])
def test_relay_bootstrap_derives_route_and_auth_identity_then_cleans_up_on_cell_stop(startup, parent, secure):
    startup.config["connect_to"][ConnPropKey.FQCN] = parent
    startup.config["connect_to"][ConnPropKey.CONNECTION_SECURITY] = (
        ConnectionSecurity.MTLS if secure else ConnectionSecurity.CLEAR
    )
    startup.config["auth_identity"] = "relay-auth-cn"
    startup.config["auth_identity_map"] = {"additional": "custom-cn"}
    startup.args.set = ["secure_train=true"] if secure else []
    startup.path.write_text(json.dumps(startup.config))
    relay.main(startup.args)
    kwargs = startup.factory.call_args.kwargs
    assert kwargs["fqcn"] == ("relay-1" if parent == "server" else "relay-0.relay-1")
    assert kwargs["auth_identity"] == "relay-auth-cn"
    assert kwargs["auth_identity_map"] == {
        parent: "server.example" if parent == "server" else "relay-0",
        "additional": "custom-cn",
    }
    assert kwargs["secure"] is secure
    url = "stcp://host:8004" if secure else "tcp://host:8004"
    assert kwargs["root_url"] == (url if parent == "server" else None)
    assert kwargs["parent_url"] == (None if parent == "server" else url)
    auth_args = relay.Authenticator.call_args.kwargs
    assert auth_args["project_name"] == "project" and auth_args["client_name"] == "relay-1"
    assert auth_args["expected_sp_identity"] == "server.example"
    assert auth_args["secure_mode"] is secure
    startup.auth.authenticate.assert_called_once()
    startup.cell.start.assert_called_once_with()
    startup.cell.stop.assert_called_once_with()
    assert startup.cell.core_cell.add_incoming_filter.called is secure
    if secure:
        assert startup.cell.core_cell.add_incoming_filter.call_args.kwargs["local_cell_fqcn"] == kwargs["fqcn"]
    assert (startup.path.parent.parent / "shutdown.fl").exists()


@pytest.mark.parametrize(
    "field,error",
    [
        ("project_name", "project_name"),
        ("server_identity", "server_identity"),
        ("identity", "identity"),
        ("connect_to", "connect_to"),
        ("address", "parent.address"),
        ("scheme", "parent.scheme"),
        ("fqcn", "parent.fqcn"),
    ],
)
def test_relay_required_configuration_is_validated_before_cell_creation(startup, field, error):
    if field in {"address", "scheme", "fqcn"}:
        startup.config["connect_to"].pop(field)
    else:
        startup.config.pop(field)
    startup.path.write_text(json.dumps(startup.config))
    with pytest.raises(RuntimeError, match=error):
        relay.main(startup.args)
    startup.factory.assert_not_called()


def test_relay_rejects_non_mapping_config_and_missing_root_credential(startup, monkeypatch):
    startup.path.write_text("[]")
    with pytest.raises(RuntimeError, match="invalid relay config"):
        relay.main(startup.args)
    startup.path.write_text(json.dumps(startup.config))
    monkeypatch.setattr(relay, "search_file", lambda *args: None)
    with pytest.raises(ValueError, match="cannot find"):
        relay.main(startup.args)
    startup.factory.assert_not_called()


def test_secure_relay_requires_a_real_token_verifier_before_registering_auth_filters(startup):
    startup.args.set = ["secure_train=true"]
    startup.path.write_text(json.dumps(startup.config))
    startup.auth.authenticate.return_value = ("token", "signature", "session", object())
    with pytest.raises(RuntimeError, match="TokenVerifier"):
        relay.main(startup.args)
    startup.cell.core_cell.add_incoming_filter.assert_not_called()


def test_relay_argument_parser_keeps_workspace_configuration_and_overrides(monkeypatch):
    monkeypatch.setattr(
        relay.sys,
        "argv",
        ["relay", "--workspace", "/workspace", "--relay_config", "relay.json", "--set", "secure_train=true"],
    )
    args = relay.parse_arguments()
    assert args.workspace == "/workspace" and args.relay_config == "relay.json" and args.set == ["secure_train=true"]
    monkeypatch.setattr(relay.sys, "argv", ["relay"])
    with pytest.raises(SystemExit):
        relay.parse_arguments()
