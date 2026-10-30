# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

from unittest.mock import Mock

import pytest

from nvflare.tool.poc import poc_commands as subject


@pytest.fixture
def project():
    return {
        "name": "project",
        "participants": [{"type": "server", "name": "server", "org": "org"}],
        "studies": {"existing": {}},
    }


@pytest.mark.parametrize("missing", ["state", "root-ca", "neither"])
def test_dynamic_participant_provisioning_requires_both_existing_ca_state_and_root(tmp_path, project, missing):
    state = tmp_path / "project" / "state" / "cert.json"
    root = tmp_path / "prod" / "server" / "startup" / "rootCA.pem"
    state.parent.mkdir(parents=True)
    root.parent.mkdir(parents=True)
    if missing != "state":
        state.write_text("{}")
    if missing != "root-ca":
        root.write_text("root")
    if missing == "neither":
        assert subject._ensure_dynamic_poc_ca_available(
            str(tmp_path), "project", str(tmp_path / "prod"), project
        ) == str(root)
    else:
        with pytest.raises(subject.CLIException, match="CA state|rootCA.pem"):
            subject._ensure_dynamic_poc_ca_available(str(tmp_path), "project", str(tmp_path / "prod"), project)


@pytest.mark.parametrize("outcome", ["build-error", "missing-output", "missing-kit", "exists", "replace", "docker"])
def test_dynamic_provision_validates_generated_output_and_replaces_only_selected_kit(
    tmp_path, project, monkeypatch, outcome
):
    target = tmp_path / "prod"
    target.mkdir()
    generated = tmp_path / "generated"
    generated.mkdir()
    if outcome != "missing-kit":
        (generated / "site").mkdir()
        (generated / "site" / "new").write_text("new kit")
    if outcome in ("exists", "replace"):
        (target / "site").mkdir()
        (target / "site" / "old").write_text("old kit")
    (target / "other").mkdir()
    (target / "other" / "keep").write_text("preserve")
    monkeypatch.setattr(subject, "_ensure_dynamic_poc_ca_available", Mock(return_value="root.pem"))
    for name in ("prepare_project", "prepare_builders", "prepare_packager"):
        monkeypatch.setattr(subject, name, Mock())
    runtime = Mock()
    runtime.provision.return_value = {
        subject.CtxKey.BUILD_ERROR: outcome == "build-error",
        subject.CtxKey.CURRENT_PROD_DIR: None if outcome == "missing-output" else str(generated),
    }
    factory = Mock(return_value=runtime)
    monkeypatch.setattr(subject, "Provisioner", factory)
    monkeypatch.setattr(
        subject,
        "get_docker_run_mode",
        lambda config: subject.SC.DOCKER_RUN_MODE_DEPLOY if outcome == "docker" else None,
    )
    monkeypatch.setattr(subject, "get_poc_docker_runtime_config", lambda config: {"image": "example"})
    docker = Mock()
    monkeypatch.setattr(subject, "_prepare_poc_docker_kit", docker)
    participant = {"name": "site", "type": "client", "org": "org"}
    if outcome in ("replace", "docker"):
        assert subject._provision_poc_participant_only(
            str(tmp_path), project, participant, str(target), force=outcome == "replace"
        ) == str(target / "site")
        assert (target / "site" / "new").read_text() == "new kit"
        assert not (target / "site" / "old").exists()
    else:
        with pytest.raises(subject.CLIException):
            subject._provision_poc_participant_only(str(tmp_path), project, participant, str(target))
        if outcome == "exists":
            assert (target / "site" / "old").read_text() == "old kit"
    assert (target / "other" / "keep").read_text() == "preserve"
    assert docker.call_count == int(outcome == "docker")
    if outcome not in ("build-error", "missing-output"):
        assert not generated.exists()
    reduced = subject.prepare_project.call_args.args[0]
    assert reduced["participants"] == [project["participants"][0], participant]
    assert "studies" not in reduced and "studies" in project


@pytest.mark.parametrize("kind", ["file", "directory", "absent"])
def test_existing_path_removal_handles_files_directories_and_missing_paths(tmp_path, kind):
    path = tmp_path / "target"
    if kind == "file":
        path.write_text("remove")
    elif kind == "directory":
        path.mkdir()
        (path / "child").write_text("remove")
    subject._remove_path(str(path))
    assert not path.exists()


def test_dynamic_configuration_rejects_ambiguous_server_topology_without_modifying_input(project):
    project["participants"].append({"type": "server", "name": "duplicate"})
    with pytest.raises(subject.CLIException, match="only have one server"):
        subject._dynamic_poc_project_config(project, {"type": "client", "name": "site"})
    assert len(project["participants"]) == 2
