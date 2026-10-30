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
from pyhocon import ConfigFactory

from nvflare.tool import cli_output
from nvflare.tool.kit import kit_cli as subject


@pytest.mark.parametrize("status", ["inactive", "unregistered", "ok", "missing", "invalid"])
def test_json_inspection_exposes_identity_or_actionable_missing_kit_findings(monkeypatch, tmp_path, capsys, status):
    entries = {} if status == "unregistered" else {"alice": str(tmp_path / "kit")}
    config = ConfigFactory.from_dict({"version": 2, "startup_kits": {"entries": entries}})
    if status != "inactive":
        config.put("startup_kits.active", "alice")
    monkeypatch.setattr(subject, "load_cli_config", lambda: config)
    monkeypatch.setattr(subject, "get_cli_config_path", lambda: tmp_path / "cli.conf")
    monkeypatch.setattr(subject, "_kit_sub_cmd_parsers", {subject.CMD_KIT_INSPECT: None})
    monkeypatch.setattr(subject, "handle_schema_flag", Mock())
    monkeypatch.setattr(cli_output, "_output_format", "json")
    monkeypatch.delenv(subject.NVFLARE_STARTUP_KIT_DIR, raising=False)
    findings = [{"code": "EXPIRED", "severity": "error"}]
    metadata = {
        "identity": "alice",
        "cert_role": "lead",
        "org": "org",
        "project": "project",
        "certificate": {"status": "expired"},
        "findings": findings,
    }
    probe = Mock(return_value=(status, str(tmp_path / "normalized"), metadata))
    monkeypatch.setattr(subject, "get_startup_kit_status", probe)
    subject.cmd_kit_inspect(SimpleNamespace())
    data = json.loads(capsys.readouterr().out)["data"]
    if status == "inactive":
        assert data["active"] is None and data["findings"][0]["code"] == "NO_ACTIVE_STARTUP_KIT"
    elif status == "unregistered":
        assert data["status"] == "unregistered" and "hint" in data
    else:
        assert data["path"] == str(tmp_path / "normalized") and data["status"] == status
        assert data["findings"] == findings
        if status == "ok":
            assert data["identity"] == "alice" and data["role"] == "lead" and data["org"] == "org"
        else:
            assert data["identity"] == data["role"] == data["project"] == "-"
            assert data["certificate"] is None and "hint" in data
    assert probe.call_count == int(status not in ("inactive", "unregistered"))
