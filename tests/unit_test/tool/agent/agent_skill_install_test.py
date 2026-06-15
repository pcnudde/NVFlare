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
from unittest.mock import patch

import pytest

from nvflare.tool import cli_output


@pytest.fixture(autouse=True)
def reset_cli_output_state(monkeypatch):
    monkeypatch.setattr(cli_output, "_output_format", "txt")
    monkeypatch.setattr(cli_output, "_connect_timeout", 5.0)


def _run_main(argv):
    from nvflare import cli

    with patch("sys.argv", argv), patch("nvflare.cli.version_check"):
        try:
            cli.main()
        except SystemExit as e:
            return e.code
    return 0


def _load_single_stdout_json(captured):
    stdout = captured.out.strip()
    assert stdout
    assert len(stdout.splitlines()) == 1
    return json.loads(stdout)


def test_install_skills_dry_run_does_not_write(tmp_path):
    from nvflare.tool.install_skills import install_skills

    plan = install_skills(agent="codex", target_dir=tmp_path, dry_run=True)

    assert plan["applied"] is False
    assert plan["skills"][0]["name"] == "nvflare"
    assert plan["skills"][0]["action"] == "copy"
    assert not tmp_path.joinpath("nvflare").exists()


def test_install_skills_copies_single_nvflare_skill(tmp_path):
    from nvflare.tool.install_skills import install_skills, list_skills

    plan = install_skills(agent="codex", target_dir=tmp_path)

    skill_file = tmp_path / "nvflare" / "SKILL.md"
    assert plan["applied"] is True
    assert plan["installed"] == ["nvflare"]
    assert skill_file.is_file()
    assert "Recipe API" in skill_file.read_text(encoding="utf-8")

    listed = list_skills(agent="codex", target_dir=tmp_path)
    assert listed["installed"][0]["name"] == "nvflare"
    assert listed["installed"][0]["status"] == "current"


def test_agent_skills_install_dry_run_json(capsys, tmp_path):
    exit_code = _run_main(
        [
            "nvflare",
            "agent",
            "skills",
            "install",
            "--agent",
            "codex",
            "--target",
            str(tmp_path),
            "--dry-run",
            "--format",
            "json",
        ]
    )

    assert exit_code == 0
    payload = _load_single_stdout_json(capsys.readouterr())
    assert payload["schema_version"] == "1"
    assert payload["status"] == "ok"
    assert payload["data"]["applied"] is False
    assert payload["data"]["skills"][0]["name"] == "nvflare"
    assert not tmp_path.joinpath("nvflare").exists()


def test_agent_skills_install_and_list_json(capsys, tmp_path):
    install_exit = _run_main(
        [
            "nvflare",
            "agent",
            "skills",
            "install",
            "--agent",
            "codex",
            "--target",
            str(tmp_path),
            "--format",
            "json",
        ]
    )
    install_payload = _load_single_stdout_json(capsys.readouterr())

    assert install_exit == 0
    assert install_payload["data"]["applied"] is True
    assert tmp_path.joinpath("nvflare", "SKILL.md").is_file()

    list_exit = _run_main(
        [
            "nvflare",
            "agent",
            "skills",
            "list",
            "--agent",
            "codex",
            "--target",
            str(tmp_path),
            "--format",
            "json",
        ]
    )
    list_payload = _load_single_stdout_json(capsys.readouterr())

    assert list_exit == 0
    assert list_payload["data"]["available"][0]["name"] == "nvflare"
    assert list_payload["data"]["installed"][0]["name"] == "nvflare"


def test_agent_skills_install_schema_exits_zero(capsys):
    exit_code = _run_main(["nvflare", "agent", "skills", "install", "--schema"])

    assert exit_code == 0
    schema = json.loads(capsys.readouterr().out)
    assert schema["command"] == "nvflare agent skills install"
    assert schema["mutating"] is True
    args_by_name = {arg["name"]: arg for arg in schema["args"]}
    assert args_by_name["--agent"]["required"] is True
    assert args_by_name["--skill"]["required"] is False
