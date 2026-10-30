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

import argparse
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from nvflare.lighter import provision as module
from nvflare.lighter.constants import CtxKey


@pytest.fixture
def command(monkeypatch, tmp_path):
    from nvflare.tool import cli_output, cli_schema, install_skills

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(module, "_provision_parser", None)
    monkeypatch.setattr(
        module,
        "sys",
        SimpleNamespace(path=list(sys.path), argv=["provision"], stdin=SimpleNamespace(isatty=lambda: True)),
    )
    monkeypatch.setattr(cli_schema, "handle_schema_flag", Mock())
    monkeypatch.setattr(install_skills, "install_skills", Mock(side_effect=OSError("offline")))
    monkeypatch.setattr(cli_output, "is_json_mode", lambda: False)
    monkeypatch.setattr(cli_output, "_is_json_mode", lambda: False)
    output = Mock()
    monkeypatch.setattr(cli_output, "output_ok", output)
    parser = argparse.ArgumentParser()
    module.define_provision_parser(parser)
    args = parser.parse_args(["-p", "project.yml"])
    (tmp_path / "project.yml").write_text("name: project")
    monkeypatch.setattr(module, "load_yaml", Mock(return_value={"name": "project"}))
    runtime = Mock(return_value={})
    monkeypatch.setattr(module, "provision", runtime)
    return SimpleNamespace(args=args, path=tmp_path, output=output, runtime=runtime)


@pytest.mark.parametrize("case", ["schema", "missing_name", "user", "client"])
def test_provision_cli_rejects_invalid_loaded_config_or_missing_extra_files(command, monkeypatch, case):
    if case == "schema":
        monkeypatch.setattr(module, "load_yaml", Mock(return_value=["not-mapping"]))
    elif case == "missing_name":
        monkeypatch.setattr(module, "load_yaml", Mock(return_value={"description": "missing name"}))
    else:
        setattr(command.args, "add_" + case, "missing.yml")
    with pytest.raises(SystemExit) as caught:
        module.handle_provision(command.args)
    assert caught.value.code == 4
    command.runtime.assert_not_called()
    command.output.assert_not_called()


@pytest.mark.parametrize("failure,code", [(ValueError("invalid"), 4), (TypeError("unexpected"), 5), (SystemExit(3), 3)])
def test_provision_cli_preserves_fatal_exit_and_maps_expected_versus_internal_failures(command, failure, code):
    command.runtime.side_effect = failure
    with pytest.raises(SystemExit) as caught:
        module.handle_provision(command.args)
    assert caught.value.code == code
    command.output.assert_not_called()


def test_provision_cli_failed_assembly_includes_errors_and_warnings_without_success(command, capsys):
    command.runtime.return_value = {
        CtxKey.BUILD_ERROR: True,
        CtxKey.ERRORS: ["failed cert"],
        CtxKey.WARNINGS: ["missing optional"],
    }
    with pytest.raises(SystemExit) as caught:
        module.handle_provision(command.args)
    assert caught.value.code == 5
    output = capsys.readouterr()
    assert "failed cert" in output.err and "missing optional" in output.err
    command.output.assert_not_called()


def test_provision_cli_existing_workspace_decline_does_not_build(command, monkeypatch):
    from nvflare.tool import cli_output

    folder = command.path / "workspace" / "project"
    folder.mkdir(parents=True)
    (folder / "existing").write_text("keep")
    prompt = Mock(return_value=False)
    monkeypatch.setattr(cli_output, "prompt_yn", prompt)
    module.handle_provision(command.args)
    prompt.assert_called_once()
    command.runtime.assert_not_called()
    assert (folder / "existing").read_text() == "keep"


def test_provision_cli_success_reports_generated_directories_and_contains_skill_install_failure(command):
    folder = command.path / "workspace" / "project"
    (folder / "prod_00").mkdir(parents=True)
    (folder / "metadata.txt").write_text("metadata")
    command.args.force = True
    module.handle_provision(command.args)
    command.output.assert_called_once_with({"workspace": str(command.path / "workspace"), "packages": ["prod_00"]})


@pytest.mark.parametrize("edge", [True, False])
def test_provision_sample_generation_handles_skill_install_failure_without_real_install(command, monkeypatch, edge):
    command.args.project_file = None
    command.args.gen_edge = edge
    copy = Mock()
    monkeypatch.setattr(module, "copy_project", copy)
    module.handle_provision(command.args)
    copy.assert_called_once_with("edge_project.yml" if edge else "dummy_project.yml", str(command.path / "project.yml"))
    assert command.output.call_args.args[0]["suggested_command"] == "nvflare provision -p project.yml"
    command.runtime.assert_not_called()


def test_deprecated_provision_main_parses_generate_flag_before_dispatch(monkeypatch, capsys):
    from nvflare.tool import cli_output

    monkeypatch.setattr(cli_output, "is_json_mode", lambda: False)
    monkeypatch.setattr(cli_output, "_is_json_mode", lambda: False)
    monkeypatch.setattr(module, "_provision_parser", None)
    monkeypatch.setattr(sys, "argv", ["provision", "-g"])
    handler = Mock()
    monkeypatch.setattr(module, "handle_provision", handler)
    module.main()
    assert handler.call_args.args[0].generate
    assert "deprecated" in capsys.readouterr().out


def test_study_normalization_deduplicates_admins_and_rejects_invalid_client_org_mapping():
    participants = [{"type": "client", "name": "site", "org": "org"}, {"type": "admin", "name": "alice", "org": "org"}]
    data = {"studies": {"study": {"site_orgs": {"org": ["site"]}, "admins": ["alice", "alice"]}}}
    result = module._normalize_and_validate_studies(data, participants, 4)
    assert result["study"]["admins"] == ["alice"]
    for groups in [{"org": "site"}, {"org": ["missing"]}]:
        with pytest.raises(ValueError, match="must be a list|unknown client"):
            module._normalize_and_validate_studies({"studies": {"study": {"site_orgs": groups}}}, participants, 4)
    with pytest.raises(ValueError, match="unknown admin"):
        module._normalize_and_validate_studies({"studies": {"study": {"admins": ["unknown"]}}}, participants, 4)
