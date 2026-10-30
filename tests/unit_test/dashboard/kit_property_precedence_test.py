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
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from nvflare.dashboard.application import blob as module
from nvflare.lighter.constants import PropKey


@pytest.mark.parametrize("operator_props", [{}, {PropKey.ROLE: "project_admin", "operator": "trusted"}])
def test_admin_kit_uses_operator_properties_after_stored_user_properties(monkeypatch, operator_props):
    user = SimpleNamespace(
        email="alice@example.org",
        role=SimpleNamespace(name="org_admin"),
        props=json.dumps({PropKey.ROLE: "project_admin", "stored": True}),
        organization=SimpleNamespace(name="org"),
    )
    query = Mock()
    query.get.return_value = user
    monkeypatch.setattr(module, "User", SimpleNamespace(query=query))
    increment = Mock()
    monkeypatch.setattr(module, "inc_dl", increment)
    project = Mock()
    properties = SimpleNamespace(get_admin_props=Mock(return_value=operator_props))
    assert module._prepare_user(properties, project, 7) is project.add_admin.return_value
    expected = {PropKey.ROLE: "project_admin", "stored": True, **operator_props}
    project.add_admin.assert_called_once_with(name=user.email, org="org", props=expected)
    increment.assert_called_once_with(module.User, 7)


@pytest.mark.parametrize("target", ["server", "admin"])
def test_kit_provisioning_merges_operator_project_and_server_properties_and_fakes_archive_boundary(monkeypatch, target):
    project = SimpleNamespace(
        scheme="grpc",
        app_location="registry/image:tag",
        root_key="private",
        root_cert="certificate",
        project_props=json.dumps({"project_setting": 7}),
        server_props=json.dumps({PropKey.ADMIN_PORT: 9000, "stored": True}),
        short_name="project",
        description="description",
        server1="server",
    )
    monkeypatch.setattr(module, "Project", SimpleNamespace(query=SimpleNamespace(first=lambda: project)))
    monkeypatch.setattr(module, "Store", SimpleNamespace(get_user=lambda user_id: {"user": {"organization": "org"}}))
    manager = Mock()
    manager.get_server_prop.side_effect = lambda key, default: default
    manager.get_server_props.return_value = {PropKey.ADMIN_PORT: 9001, "operator": True}
    monkeypatch.setattr(module, "PropertyManager", lambda: manager)
    monkeypatch.setattr(module, "deserialize_ca_key", Mock(return_value="decoded key"))
    prov_project = Mock()
    prov_project.set_server.return_value = SimpleNamespace(name="server")
    project_factory = Mock(return_value=prov_project)
    monkeypatch.setattr(module, "ProvProject", project_factory)
    provisioner = Mock()

    def create(manager, directory, scheme, image):
        output = Path(directory) / "output"
        for name in ("server", "admin"):
            path = output / name
            path.mkdir(parents=True)
            (path / "tmp.zip").write_bytes(b"archive content")
        provisioner.provision.return_value.get_result_location.return_value = str(output)
        return provisioner

    monkeypatch.setattr(module, "_get_provisioner", create)
    archive = Mock()
    monkeypatch.setattr(module, "subprocess", SimpleNamespace(run=archive))
    prepare = Mock(return_value=SimpleNamespace(name="admin")) if target == "admin" else None
    if prepare:
        result, filename = module._gen_kit("safeKey123", prepare, user_id=7)
        prepare.assert_called_once_with(manager, prov_project, user_id=7)
    else:
        result, filename = module.gen_server_blob("safeKey123")
    assert result.read() == b"archive content" and filename == target + ".zip"
    assert project_factory.call_args.kwargs["props"] == {PropKey.API_VERSION: 3, "project_setting": 7}
    assert prov_project.set_server.call_args.kwargs["props"] == {
        PropKey.FED_LEARN_PORT: 8002,
        PropKey.ADMIN_PORT: 9001,
        PropKey.DEFAULT_HOST: "server",
        "stored": True,
        "operator": True,
    }
    assert archive.call_args.args[0] == ["zip", "-rq", "-P", "safeKey123", "tmp.zip", "."]
    assert Path(archive.call_args.kwargs["cwd"]).name == target
    assert not Path(archive.call_args.kwargs["cwd"]).exists()


def test_unsafe_download_key_is_rejected_before_database_or_provisioning(monkeypatch):
    store = Mock()
    monkeypatch.setattr(module, "Store", store)
    with pytest.raises(RuntimeError, match="unsafe download key"):
        module.gen_server_blob("../unsafe")
    store.get_user.assert_not_called()


def test_user_blob_selects_user_preparation_and_error_logger_keeps_failure_visible(monkeypatch, capsys):
    generate = Mock(return_value=("bytes", "user.zip"))
    monkeypatch.setattr(module, "_gen_kit", generate)
    assert module.gen_user_blob("safe", 7) == ("bytes", "user.zip")
    generate.assert_called_once_with("safe", module._prepare_user, user_id=7)
    module.DummyLogger().error("provisioning failed")
    assert capsys.readouterr().out == "ERROR: provisioning failed\n"
