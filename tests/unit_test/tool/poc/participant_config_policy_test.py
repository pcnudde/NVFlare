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

import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import yaml
from pyhocon import ConfigFactory

from nvflare.cli_exception import CLIException
from nvflare.tool import cli_output, cli_schema
from nvflare.tool.poc import poc_commands as subject


@pytest.fixture
def cli(monkeypatch):
    monkeypatch.setattr(cli_schema, "handle_schema_flag", Mock())
    monkeypatch.setattr(subject, "get_poc_workspace", Mock(return_value="/fake/workspace"))
    cli_output.set_output_format("json")


@pytest.mark.parametrize("kind", ["user", "site"])
@pytest.mark.parametrize(
    "error,code,exit_code",
    [
        (subject.StartupKitConfigError("missing", "activate a kit"), "STARTUP_KIT_MISSING", 4),
        (ValueError("missing"), "STARTUP_KIT_MISSING", 4),
        (CLIException("invalid"), "INVALID_ARGS", 4),
        (RuntimeError("internal"), "INTERNAL_ERROR", 5),
    ],
)
def test_add_participant_reports_classified_cli_failure_without_external_provisioning(
    cli, monkeypatch, capsys, kind, error, code, exit_code
):
    command = subject.add_poc_user if kind == "user" else subject.add_poc_site
    backend = Mock(side_effect=error)
    monkeypatch.setattr(subject, "_add_poc_user" if kind == "user" else "_add_poc_site", backend)
    args = SimpleNamespace(cert_role="lead", email="user@example.com", org="org", name="site-1", force=True)
    with pytest.raises(SystemExit) as raised:
        command(args)
    assert raised.value.code == exit_code
    response = json.loads(capsys.readouterr().out)
    assert response["status"] == "error"
    assert response["error_code"] == code
    assert response["exit_code"] == exit_code
    assert backend.call_args.kwargs["force"] is True


@pytest.mark.parametrize("active", [None, "kit-1"])
def test_add_user_human_output_guides_activation_for_new_and_active_kits(cli, monkeypatch, capsys, active):
    cli_output.set_output_format("txt")
    result = {
        "status": "created",
        "identity": "user@example.com",
        "startup_kit": "/kits/user",
        "active": active,
        "next_step": "nvflare config use kit-1",
    }
    monkeypatch.setattr(subject, "_add_poc_user", Mock(return_value=result))
    subject.add_poc_user(SimpleNamespace(cert_role="lead", email="user@example.com", org="org"))
    output = capsys.readouterr().out
    assert "user@example.com" in output
    assert "/kits/user" in output
    assert ("Active startup kit: kit-1" if active else "nvflare config use kit-1") in output


def test_add_site_human_output_provides_precise_start_command(cli, monkeypatch, capsys):
    cli_output.set_output_format("txt")
    monkeypatch.setattr(
        subject,
        "_add_poc_site",
        Mock(return_value={"status": "created", "name": "site-3", "startup_kit": "/kits/site-3"}),
    )
    subject.add_poc_site(SimpleNamespace(name="site-3", org="org"))
    assert "nvflare poc start -p site-3" in capsys.readouterr().out


def test_config_read_reports_environment_override_and_write_preserves_other_settings(
    cli, monkeypatch, capsys, tmp_path
):
    loaded = ConfigFactory.parse_string("other=9")
    monkeypatch.setattr(
        subject, "load_hidden_config_state", Mock(return_value=(str(tmp_path / "config.conf"), loaded, False))
    )
    monkeypatch.setenv("NVFLARE_POC_WORKSPACE", "/env/workspace")
    cli_output.set_output_format("txt")
    subject.config_poc(SimpleNamespace(poc_workspace_dir=None))
    assert "NVFLARE_POC_WORKSPACE=/env/workspace" in capsys.readouterr().out
    save = Mock()
    monkeypatch.setattr(subject, "save_config", save)
    destination = tmp_path / "poc"
    subject.config_poc(SimpleNamespace(poc_workspace_dir=str(destination)))
    config, path = save.call_args.args
    assert path == str(tmp_path / "config.conf")
    assert config.get("other") == 9
    assert config.get(f"{subject.POC_KEY}.{subject.WORKSPACE_KEY}") == str(destination)
    assert config.get(subject.CONFIG_VERSION) == subject.CURRENT_CONFIG_VERSION
    assert str(destination) in capsys.readouterr().out


def test_config_write_invalid_path_returns_usage_error_without_saving(cli, monkeypatch, capsys):
    monkeypatch.setattr(subject, "load_hidden_config_state", Mock(return_value=("/fake/config", None, False)))
    monkeypatch.setattr(subject, "create_poc_workspace_config", Mock(side_effect=ValueError("bad workspace")))
    save = Mock()
    monkeypatch.setattr(subject, "save_config", save)
    with pytest.raises(SystemExit) as raised:
        subject.config_poc(SimpleNamespace(poc_workspace_dir="bad"))
    assert raised.value.code == 4
    assert json.loads(capsys.readouterr().out)["error_code"] == "INVALID_ARGS"
    save.assert_not_called()


@pytest.mark.parametrize("contents", ["- list", "studies: []", "studies:\n  __poc__: []", "["])
def test_docker_study_runtime_rejects_malformed_configuration_before_replacing_file(tmp_path, contents):
    (tmp_path / "local").mkdir()
    path = tmp_path / "local" / "study_runtime.yaml"
    if "__poc__" in contents:
        contents = contents.replace("__poc__", subject.POC_DOCKER_DATA_STUDY)
    path.write_text(contents)
    with pytest.raises(CLIException):
        subject._write_poc_docker_study_runtime(tmp_path, str(tmp_path))
    assert path.read_text() == contents


def test_docker_study_runtime_rejects_legacy_coexistence_and_preserves_existing_dataset(tmp_path):
    (tmp_path / "local").mkdir()
    legacy = tmp_path / "local" / "study_data.yaml"
    legacy.write_text("legacy")
    with pytest.raises(CLIException, match="legacy"):
        subject._write_poc_docker_study_runtime(tmp_path, str(tmp_path))
    legacy.unlink()
    path = tmp_path / "local" / "study_runtime.yaml"
    dataset = {"source": "/explicit/data", "mode": "ro"}
    data = {"studies": {subject.POC_DOCKER_DATA_STUDY: {"datasets": {subject.POC_DOCKER_DATA_DATASET: dataset}}}}
    path.write_text(yaml.safe_dump(data))
    subject._write_poc_docker_study_runtime(tmp_path, str(tmp_path))
    actual = yaml.safe_load(path.read_text())
    assert actual["format_version"] == 2
    assert actual["studies"][subject.POC_DOCKER_DATA_STUDY]["datasets"][subject.POC_DOCKER_DATA_DATASET] == dataset
