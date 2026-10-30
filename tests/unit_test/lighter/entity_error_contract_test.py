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

import pytest

from nvflare.lighter import entity as module
from nvflare.lighter.constants import ParticipantType, PropKey


@pytest.mark.parametrize(
    "properties, message",
    [
        ({PropKey.ROLE: 7}, "must be str"),
        ({PropKey.ROLE: ""}, "empty value"),
        ({PropKey.CONN_SECURITY: 7}, "must be a str"),
        ({PropKey.CONN_SECURITY: "invalid"}, "must be one of"),
        ({PropKey.CONNECT_TO: {PropKey.HOST: "bad host"}}, "bad value"),
        ({PropKey.CONNECT_TO: {PropKey.PORT: "8002"}}, "must be int"),
        ({PropKey.CONNECT_TO: {PropKey.PORT: -1}}, "invalid port"),
        ({PropKey.LISTENING_HOST: {PropKey.PORT: "8002"}}, "must be int"),
        ({PropKey.LISTENING_HOST: {PropKey.PORT: -1}}, "invalid port"),
        ({PropKey.LISTENING_HOST: {PropKey.DEFAULT_HOST: "bad host"}}, "bad value"),
        ({PropKey.HOST_NAMES: ["valid.example", "bad host"]}, "bad value"),
    ],
)
def test_entity_common_properties_reject_invalid_host_security_role_and_port(properties, message):
    with pytest.raises(ValueError, match=message):
        module.Entity("client", "site", properties)


@pytest.mark.parametrize("operation", [module.parse_connect_to, module.parse_listening_host])
def test_connection_parsers_reject_unsupported_shape(operation):
    with pytest.raises(ValueError, match="must be str or dict"):
        operation(123, "client", "connection")


@pytest.mark.parametrize(
    "props, kind, message",
    [
        ({PropKey.EXTERNAL_CERT: "yes"}, "client", "must be bool"),
        ({PropKey.EXTERNAL_CERT: True}, "custom", "supported only"),
        ({PropKey.EXTERNAL_JOB_CA: "yes"}, "server", "must be bool"),
        ({PropKey.EXTERNAL_JOB_CA: True}, "client", "only for server"),
    ],
)
def test_participant_external_credential_flags_reject_wrong_types_or_inapplicable_roles(props, kind, message):
    with pytest.raises(ValueError, match=message):
        module.Participant(kind, "site", "org", props)


def test_custom_participant_and_unknown_admin_role_warn_but_preserve_identity(capsys):
    custom = module.Participant("worker", "site", "org")
    admin = module.Participant(ParticipantType.ADMIN, "alice@example.com", "org", {PropKey.ROLE: "researcher"})
    assert custom.type == "worker" and admin.get_prop(PropKey.ROLE) == "researcher"
    output = capsys.readouterr().out
    assert "not a defined type" in output and "not a defined role" in output
    with pytest.raises(ValueError):
        module.Participant("bad type", "site", "org")


def test_project_rejects_invalid_participant_collections_and_missing_root_private_key():
    with pytest.raises(ValueError, match="missing root_private_key"):
        module.Project("project", "", serialized_root_cert="certificate")
    with pytest.raises(ValueError, match="participants must be a list"):
        module.Project("project", "", participants={"site": "org"})
    with pytest.raises(ValueError, match="bad item"):
        module.Project("project", "", participants=[object()])
    project = module.Project("project", "")
    with pytest.raises(ValueError, match="types must be a str or List"):
        project.get_all_participants(7)
    with pytest.raises(ValueError, match="participant_def must be dict"):
        module.participant_from_dict([])
    with pytest.raises(ValueError, match="missing participant"):
        module.participant_from_dict({"type": "client"})


def test_connection_defaults_and_property_fallback_are_explicit_for_nonserver():
    parent = module.Entity("project", "project", {"global": "fallback"})
    child = module.Participant(
        "client", "site", "org", {PropKey.LISTENING_HOST: {PropKey.HOST_NAMES: ["host.example"]}}
    )
    child.parent = parent
    assert child.get_prop_fb("missing", "global") == "fallback"
    assert child.get_prop_fb("missing", "unknown", "default") == "default"
    host = child.get_listening_host()
    assert host.default_host == "localhost" and host.port == 0 and host.scheme == "tcp"
    assert "host.example" in str(host)
    assert "site" in repr(child)
    connection = module.parse_connect_to({PropKey.NAME: "root", PropKey.HOST: "host", PropKey.AUTH_IDENTITY: "auth"})
    assert "auth" in str(connection) and connection.name == "root"
