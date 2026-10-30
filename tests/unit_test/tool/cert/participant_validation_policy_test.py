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
from unittest.mock import Mock

import pytest
import yaml

from nvflare.tool import cli_output
from nvflare.tool.cert import cert_commands as subject


@pytest.fixture(autouse=True)
def json_output():
    cli_output.set_output_format("json")


@pytest.mark.parametrize("participant", [None, {}, {"name": "site-1", "org": "org", "type": "unsupported"}])
def test_identity_derivation_reports_invalid_mapping_missing_fields_and_type(capsys, participant):
    with pytest.raises(SystemExit) as raised:
        subject._derive_identity_from_participant("study", participant)
    assert raised.value.code == 4
    assert json.loads(capsys.readouterr().out)["error_code"] == "INVALID_ARGS"


@pytest.mark.parametrize(
    "kind,fields",
    [
        ("server", {"connection_security": "invalid"}),
        ("server", {"fed_learn_port": True}),
        ("site", {"server": []}),
        ("site", {"server": {"host": ""}}),
        ("site", {"server": {"host": "server", "fed_learn_port": 8002, "admin_port": False}}),
    ],
)
def test_participant_routing_fields_reject_invalid_security_ports_and_nested_server(capsys, kind, fields):
    with pytest.raises(SystemExit):
        subject._validate_participant_connection_fields(fields, {"kind": kind})
    assert json.loads(capsys.readouterr().out)["error_code"] == "INVALID_ARGS"


def test_participant_routing_accepts_nested_server_and_normalizes_admin_role(tmp_path):
    participant = {
        "name": "member@example.com",
        "org": "org",
        "type": "admin",
        "role": "org-admin",
        "server": {"host": "server.example.com", "fed_learn_port": 8002, "admin_port": 8003},
    }
    assert subject._validate_participant_connection_fields(participant, {"kind": "user"})
    path = tmp_path / "participant.yaml"
    path.write_text(yaml.safe_dump({"name": "study", "participants": [participant]}))
    normalized, identity = subject._load_participant_definition(str(path))
    assert normalized["participants"][0]["role"] == "org_admin"
    assert identity == {
        "kind": "user",
        "name": "member@example.com",
        "cert_role": "org_admin",
        "cert_type": "org_admin",
    }


@pytest.mark.parametrize(
    "field,value",
    [
        ("scheme", "invalid"),
        ("connection_security", "invalid"),
        ("server", []),
        ("server", {"host": ""}),
        ("server", {"host": "bad host", "fed_learn_port": 8002}),
    ],
)
def test_project_profile_rejects_unsupported_network_settings_before_signing(tmp_path, capsys, field, value):
    profile = {
        "name": "study",
        "scheme": "grpc",
        "connection_security": "tls",
        "server": {"host": "server", "fed_learn_port": 8002},
    }
    profile[field] = value
    path = tmp_path / "profile.yaml"
    path.write_text(yaml.safe_dump(profile))
    with pytest.raises(SystemExit):
        subject._load_project_profile(str(path))
    assert json.loads(capsys.readouterr().out)["error_code"] == "INVALID_ARGS"


@pytest.mark.parametrize(
    "failure,expected",
    [
        ("mkdir", "OUTPUT_DIR_NOT_WRITABLE"),
        ("permission", "OUTPUT_DIR_NOT_WRITABLE"),
        ("generate", "CSR_GENERATION_FAILED"),
        ("write", "OUTPUT_DIR_NOT_WRITABLE"),
    ],
)
def test_csr_creation_failures_leave_no_partial_private_key(tmp_path, monkeypatch, capsys, failure, expected):
    output = tmp_path / "request"
    if failure == "mkdir":
        output.write_bytes(b"not-directory")
    elif failure == "permission":
        monkeypatch.setattr(subject.os, "access", Mock(return_value=False))
    elif failure == "generate":
        monkeypatch.setattr(subject, "_generate_csr", Mock(side_effect=RuntimeError("key generation")))
    else:
        monkeypatch.setattr(subject, "_write_file_nofollow", Mock(side_effect=OSError("csr write")))
    with pytest.raises(SystemExit):
        subject.generate_csr_files("site-1", "org", "client", str(output))
    assert json.loads(capsys.readouterr().out)["error_code"] == expected
    if output.is_dir():
        assert not (output / "site-1.key").exists()
        assert not (output / "site-1.csr").exists()
