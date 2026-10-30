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

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from nvflare.lighter.constants import CtxKey, PropKey, ProvFileName, ProvisionMode, TemplateSectionKey
from nvflare.lighter.entity import Participant, Project
from nvflare.lighter.impl import static_file as module


@pytest.mark.parametrize("scheme", [None, ""])
def test_static_builder_rejects_missing_or_nonstring_transport_scheme(scheme):
    with pytest.raises(ValueError, match="scheme"):
        module.StaticFileBuilder(scheme=scheme)


def test_static_builder_warns_for_custom_transport_and_omits_absent_optional_json_fields(capsys):
    assert module.StaticFileBuilder(scheme="custom").scheme == "custom"
    assert "not a builtin scheme" in capsys.readouterr().out
    fields = module.StaticFileBuilder._build_optional_json_fields([("missing", None), ("value", "present")], 2)
    assert "missing" not in fields and '"value": "present"' in fields


def test_static_connection_properties_validate_security_and_copy_custom_ca(tmp_path):
    source = tmp_path / "root.pem"
    source.write_bytes(b"trusted-root")
    destination = tmp_path / "kit"
    destination.mkdir()
    site = Participant("client", "site", "org", {PropKey.CUSTOM_CA_CERT: str(source)})
    context = SimpleNamespace(get_kit_dir=lambda participant: str(destination))
    assert module.StaticFileBuilder._build_conn_properties(site, context) == "mtls"
    assert (destination / ProvFileName.CUSTOM_CA_CERT_FILE_NAME).read_bytes() == b"trusted-root"
    site.set_prop(PropKey.CONN_SECURITY, "invalid")
    with pytest.raises(ValueError, match="invalid connection_security"):
        module.StaticFileBuilder._build_conn_properties(site, context)


@pytest.mark.parametrize("kind", ["relay", "client"])
@pytest.mark.parametrize("cycle", [True, False])
def test_topology_validation_rejects_cycles_and_unknown_parent_with_finite_lookup_budget(monkeypatch, kind, cycle):
    project = Project("project", "")
    project.set_server("server", "org", {})
    if kind == "relay":
        first = project.add_relay(
            "first", "org", {PropKey.CONNECT_TO: {PropKey.NAME: "second" if cycle else "missing"}}
        )
        second = project.add_relay("second", "org", {PropKey.CONNECT_TO: {PropKey.NAME: "first"}})
        action = module.StaticFileBuilder._determine_relay_hierarchy
    else:
        first = project.add_client("first", "org", {PropKey.PARENT: "second" if cycle else "missing"})
        second = project.add_client("second", "org", {PropKey.PARENT: "first"})
        action = module.StaticFileBuilder._determine_client_hierarchy
    lookups = []
    for participant in [first, second]:
        original = participant.get_prop

        def get(key, default=None, original=original):
            if key == PropKey.PARENT:
                lookups.append(key)
                assert len(lookups) <= 10, "topology lookup did not detect cycle"
            return original(key, default)

        monkeypatch.setattr(participant, "get_prop", get)
    with pytest.raises(ValueError, match="bad .* definitions|undefined parent"):
        action(project, {})


def test_listener_host_validation_and_direct_connection_overrides_are_preserved():
    builder = module.StaticFileBuilder()
    for host in ["default.example", "alternate.example"]:
        assert (
            builder._validate_host_name_against_listener(host, "server", "default.example", ["alternate.example"]) == ""
        )
    project = Project("project", "")
    client = project.add_client(
        "site", "org", {PropKey.CONNECT_TO: {PropKey.HOST: "other.example", PropKey.PORT: 9000}}
    )
    context = Mock()
    context.get_project.return_value = project
    with pytest.raises(ValueError, match="Missing server definition"):
        builder._determine_conn_target(client, context)
    project.set_server("server", "org", {PropKey.DEFAULT_HOST: "default.example"})
    assert builder._determine_conn_target(client, context) == ("other.example", 9000)
    context.warning.assert_called_once()
    with pytest.raises(ValueError, match="missing listening_host"):
        builder._build_relay(Participant("relay", "relay", "org"), context)


def test_poc_admin_uses_certificate_identity_and_generates_docker_launch_template(tmp_path):
    project = Project("project", "")
    project.set_server("server", "org", {})
    admin = project.add_admin("alice@example.com", "org", {PropKey.ROLE: "project_admin"})
    context = Mock()
    context.get_project.return_value = project
    context.get_provision_mode.return_value = ProvisionMode.POC
    context.get_kit_dir.return_value = str(tmp_path)
    values = {CtxKey.ADMIN_PORT: 8003, CtxKey.SERVER_NAME: "server"}
    context.get.side_effect = lambda key: values.get(key)
    module.StaticFileBuilder(docker_image="image:tag")._build_admin(admin, context)
    admin_config = next(
        call.kwargs
        for call in context.build_from_template.call_args_list
        if call.kwargs.get("file_name") == ProvFileName.FED_ADMIN_JSON
    )
    assert admin_config["replacement"]["uid_source"] == "cert"
    assert admin_config["replacement"]["username"] == ""
    docker = next(
        call
        for call in context.build_from_template.call_args_list
        if len(call.args) > 1 and call.args[1] == TemplateSectionKey.DOCKER_ADMIN_SH
    )
    assert docker.args[3]["docker_image"] == "image:tag" and docker.kwargs["exe"]
