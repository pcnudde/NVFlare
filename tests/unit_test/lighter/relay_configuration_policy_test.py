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

from nvflare.lighter.constants import CtxKey, PropKey, ProvFileName
from nvflare.lighter.entity import Participant, ParticipantType, Project
from nvflare.lighter.impl.static_file import StaticFileBuilder, check_parent


@pytest.fixture
def project():
    value = Project("project", "relay policy", props={PropKey.CONN_SECURITY: "mtls"})
    value.add_participant(
        Participant(
            ParticipantType.SERVER,
            "server",
            "org",
            {PropKey.DEFAULT_HOST: "server.example", PropKey.HOST_NAMES: ["server.example", "alias.example"]},
        )
    )
    return value


@pytest.fixture
def context(project, tmp_path):
    value = Mock()
    value.get_project.return_value = project
    value.get_kit_dir.side_effect = lambda participant: str(tmp_path / participant.name / "startup")
    value.get_local_dir.side_effect = lambda participant: str(tmp_path / participant.name / "local")
    value.get_ws_dir.side_effect = lambda participant: str(tmp_path / participant.name)
    return value


@pytest.mark.parametrize("parent_kind", ["server", "relay", "missing"])
def test_relay_config_uses_connect_to_overrides_and_reports_unknown_parent(project, context, parent_kind):
    parent = project.add_relay(
        "relay-parent",
        "org",
        {
            PropKey.LISTENING_HOST: {
                "default_host": "relay.example",
                "port": 9100,
                "scheme": "tcp",
                "connection_security": "mtls",
            }
        },
    )
    parent.set_prop(PropKey.FQCN, "relay-parent")
    target = "server" if parent_kind == "server" else "relay-parent" if parent_kind == "relay" else "missing"
    relay = project.add_relay(
        "relay-child",
        "org",
        {
            PropKey.LISTENING_HOST: {"default_host": "child.example", "port": 9200},
            PropKey.CONNECT_TO: {
                "name": target,
                "host": "external.example",
                "port": 9300,
                "connection_security": "tls",
                "auth_identity": "parent-cert",
            },
        },
    )
    relay.set_prop(PropKey.COMM_CONFIG_ARGS, {})
    relay.set_prop(PropKey.FQCN, "relay-parent.relay-child")
    values = {CtxKey.FED_LEARN_PORT: 8002, CtxKey.RELAY_MAP: {parent.name: parent}}
    context.get.side_effect = lambda key: values.get(key)
    builder = StaticFileBuilder(scheme="http")
    if parent_kind == "missing":
        with pytest.raises(RuntimeError, match="cannot get parent relay"):
            builder._build_relay(relay, context)
        return
    builder._build_relay(relay, context)
    generated = next(
        call
        for call in context.build_from_template.call_args_list
        if call.kwargs.get("file_name") == ProvFileName.FED_RELAY_JSON
    )
    fields = generated.kwargs["replacement"]
    assert fields["address"] == "external.example:9300"
    assert fields["conn_sec"] == "tls"
    assert fields["scheme"] == ("http" if parent_kind == "server" else "tcp")
    assert fields["fqcn"] == ("server" if parent_kind == "server" else "relay-parent")
    assert "parent-cert" in fields["parent_auth_identity_config"]
    assert context.warning.call_count == 1


def test_relay_config_inherits_parent_listener_when_connect_to_omits_overrides(project, context):
    parent = project.add_relay(
        "relay-parent",
        "org",
        {
            PropKey.LISTENING_HOST: {
                "default_host": "relay.example",
                "port": 9100,
                "scheme": "tcp",
                "connection_security": "tls",
            }
        },
    )
    parent.set_prop(PropKey.FQCN, "relay-parent")
    relay = project.add_relay(
        "relay-child", "org", {PropKey.LISTENING_HOST: "child.example", PropKey.CONNECT_TO: {"name": "relay-parent"}}
    )
    relay.set_prop(PropKey.COMM_CONFIG_ARGS, {})
    relay.set_prop(PropKey.FQCN, "relay-parent.relay-child")
    values = {CtxKey.FED_LEARN_PORT: 8002, CtxKey.RELAY_MAP: {parent.name: parent}}
    context.get.side_effect = lambda key: values.get(key)
    StaticFileBuilder()._build_relay(relay, context)
    fields = next(
        c.kwargs["replacement"]
        for c in context.build_from_template.call_args_list
        if c.kwargs.get("file_name") == ProvFileName.FED_RELAY_JSON
    )
    assert fields["address"] == "relay.example:9100"
    assert fields["conn_sec"] == "tls"
    context.warning.assert_not_called()


def test_start_all_orders_relay_ancestors_before_descendants(project, tmp_path):
    child = project.add_relay("child", "org", {})
    child.set_prop(PropKey.FQCN, "root.child")
    root = project.add_relay("root", "org", {})
    root.set_prop(PropKey.FQCN, "root")
    project.add_client("site-a", "org", {})
    context = SimpleNamespace(get_wip_dir=lambda: str(tmp_path))
    StaticFileBuilder()._create_start_all(project, context)
    lines = (tmp_path / ProvFileName.START_ALL_SH).read_text().splitlines()[2:]
    assert lines == [
        "$BASE_DIR/server/startup/start.sh",
        "$BASE_DIR/root/startup/start.sh",
        "$BASE_DIR/child/startup/start.sh",
        "$BASE_DIR/site-a/startup/start.sh",
    ]


@pytest.mark.parametrize("value", [None, False, True])
def test_signed_job_policy_rendering_is_explicit_and_validated(tmp_path, value):
    builder = StaticFileBuilder(require_signed_jobs=value)
    rendered = builder._build_require_signed_jobs_config()
    assert rendered == ("" if value is None else ',\n  "require_signed_jobs": ' + json.dumps(value))
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"require_signed_jobs": value}))
    builder._validate_require_signed_jobs_config(str(tmp_path), "config.json")
    if value is not None:
        path.write_text(json.dumps({"require_signed_jobs": not value}))
        with pytest.raises(ValueError, match="did not render"):
            builder._validate_require_signed_jobs_config(str(tmp_path), "config.json")


def test_parent_validation_rejects_cycles_before_provisioning(project):
    first = project.add_client("first", "org", {})
    second = project.add_client("second", "org", {})
    first.set_prop(PropKey.PARENT, second)
    second.set_prop(PropKey.PARENT, first)
    path = []
    assert "circular parent ref first" == check_parent(first, path)
    assert first.get_prop(PropKey.IS_LEAF) is False
    assert second.get_prop(PropKey.IS_LEAF) is False
