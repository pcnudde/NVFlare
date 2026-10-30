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

from argparse import Namespace
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import yaml

from nvflare.lighter import provision as subject
from nvflare.lighter.impl.cert import CertBuilder
from nvflare.lighter.tplt_utils import Template


@pytest.mark.parametrize("function,kind", [("add_extra_clients", "client"), ("add_extra_users", "admin")])
def test_extra_participant_yaml_gets_expected_type_and_invalid_shapes_do_not_append(
    tmp_path, function, kind, monkeypatch
):
    path = tmp_path / "participant.yaml"
    path.write_text(yaml.safe_dump({"name": "site-a", "org": "org"}))
    participants = []
    getattr(subject, function)(str(path), participants)
    assert participants == [{"name": "site-a", "org": "org", "type": kind}]
    path.write_text("- invalid\n")
    with pytest.raises(SystemExit) as raised:
        getattr(subject, function)(str(path), participants)
    assert raised.value.code == 4
    assert len(participants) == 1


def test_provision_resolves_public_root_relative_to_project_file_before_delegating(tmp_path, monkeypatch):
    project = Mock()
    builder = Mock(spec=CertBuilder)
    builder.public_root_cert = "public-root.pem"
    packager = Mock()
    monkeypatch.setattr(subject, "prepare_project", Mock(return_value=project))
    monkeypatch.setattr(subject, "prepare_builders", Mock(return_value=[builder]))
    monkeypatch.setattr(subject, "prepare_packager", Mock(return_value=packager))
    runtime = Mock()
    factory = Mock(return_value=runtime)
    monkeypatch.setattr(subject, "Provisioner", factory)
    data = {}
    result = subject.provision(
        Namespace(gen_scripts=True), data, str(tmp_path / "project.yml"), str(tmp_path / "output")
    )
    assert data["gen_scripts"] is True
    assert builder.public_root_cert == str(tmp_path / "public-root.pem")
    factory.assert_called_once_with(str(tmp_path / "output"), [builder], packager)
    runtime.provision.assert_called_once_with(project)
    assert result is runtime.provision.return_value


@pytest.mark.parametrize("failure", ["public-root", "missing-participants", "edge-exception"])
def test_edge_provisioning_rejects_unsupported_credentials_or_missing_participants(monkeypatch, failure):
    builder = Mock(spec=CertBuilder)
    builder.public_root_cert = "root.pem" if failure == "public-root" else None
    monkeypatch.setattr(subject, "prepare_builders", Mock(return_value=[builder]))
    if failure == "edge-exception":
        monkeypatch.setattr(subject, "provision_for_edge", Mock(side_effect=RuntimeError("edge failed")))
        with pytest.raises(SystemExit) as raised:
            subject.provision(Namespace(gen_scripts=False), {"edge": {"depth": 1}}, "project.yml", "workspace")
        assert raised.value.code == 5
    else:
        with pytest.raises(ValueError, match="public_root_cert" if failure == "public-root" else "participants"):
            subject.provision_for_edge({}, {"name": "project", "api_version": 3})


@pytest.mark.parametrize("definition", [{"site_orgs": None, "admins": None}, {"site_orgs": {}, "admins": []}, None])
def test_study_normalization_accepts_empty_study_with_optional_collections(definition):
    data = {"studies": {"study-a": definition}}
    normalized = subject._normalize_and_validate_studies(data, [], 4)
    assert normalized == {"study-a": {} if definition is None else {"site_orgs": {}, "admins": []}}


@pytest.mark.parametrize(
    "definition", [{"site_orgs": []}, {"admins": "admin@example.com"}, {"site_orgs": {"unknown": []}}]
)
def test_study_normalization_rejects_wrong_shapes_and_unknown_organization(definition):
    with pytest.raises(ValueError):
        subject._normalize_and_validate_studies({"studies": {"study-a": definition}}, [], 4)


@pytest.mark.parametrize("csp,role", [("aws", "server"), ("aws", "client"), ("azure", "server"), ("azure", "client")])
def test_cloud_templates_render_role_identity_and_network_flags_without_execution(csp, role):
    values = {
        "cloud_script_header": "HEADER\n",
        "aws_start_sh": "type={~~type~~} uid={~~cln_uid~~} org={~~ORG~~} server={~~server_name~~} rule={~~inbound_rule~~}",
        "azure_start_svr_header_sh": "SERVER {~~server_name~~}\n",
        "azure_start_cln_header_sh": "CLIENT {~~cln_uid~~}\n",
        "azure_start_common_sh": "type={~~type~~} network={~~docker_network~~} org={~~ORG~~}",
        "sub_start_sh": "SUBSTART",
    }
    if role == "client":
        values["aws_start_sh"] = values["aws_start_sh"].replace(" server={~~server_name~~}", "")
    template = Template(values)
    entity = SimpleNamespace(name="site-a", org="org-a")
    script = template.get_start_sh(csp, role, entity)
    assert script.startswith("HEADER\n") and "type=" + role in script
    assert "{~~" not in script
    if role == "client":
        assert "uid=site-a" in script and "org-a" in script
    if csp == "azure" and role == "server":
        assert "--network host" in script
    assert template.get_start_sh("unsupported", role, entity) == ""
    assert template.get_sub_start_sh() == "SUBSTART"
    assert (
        template.get_azure_svr_sh()
        == "HEADER\nSERVER {~~server_name~~}\ntype={~~type~~} network={~~docker_network~~} org={~~ORG~~}"
    )
    assert template.get_azure_cln_sh().startswith("HEADER\nCLIENT")
