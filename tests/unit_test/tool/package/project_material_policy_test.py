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

from nvflare.tool import cli_output
from nvflare.tool.package import package_commands as subject


@pytest.fixture(autouse=True)
def json_output():
    cli_output.set_output_format("json")


@pytest.mark.parametrize(
    "kind,name,expected",
    [
        ("client", "site-1", "client"),
        ("server", "server.example.com", "server"),
        ("lead", "user@example.com", "lead"),
        ("member", "user@example.com", "member"),
        ("org_admin", "user@example.com", "org_admin"),
    ],
)
def test_single_participant_yaml_is_translated_to_typed_project(tmp_path, kind, name, expected):
    path = tmp_path / "site.yaml"
    path.write_text(yaml.safe_dump({"name": name, "org": "org", "type": kind, "project_name": "study"}))
    project, builders = subject._load_project_from_file(str(path))
    assert project.name == "study"
    participant = subject._find_project_participant(project, name)
    assert participant is not None
    assert subject._participant_kit_type(participant) == expected
    assert isinstance(builders, list)
    assert project.get_relays() == []


@pytest.mark.parametrize(
    "case,code",
    [
        ("missing", "PROJECT_FILE_NOT_FOUND"),
        ("yaml", "INVALID_PROJECT_FILE"),
        ("unknown-type", "INVALID_PROJECT_FILE"),
        ("bad-schema", "INVALID_PROJECT_FILE"),
        ("relay", "UNSUPPORTED_TOPOLOGY"),
    ],
)
def test_invalid_project_documents_are_reported_before_packaging(tmp_path, capsys, case, code):
    path = tmp_path / "project.yaml"
    if case == "yaml":
        path.write_text("[")
    elif case == "unknown-type":
        path.write_text(yaml.safe_dump({"name": "site", "org": "org", "type": "unsupported"}))
    elif case == "bad-schema":
        path.write_text("participants: invalid")
    elif case == "relay":
        path.write_text(
            yaml.safe_dump({"name": "study", "participants": [{"name": "relay", "org": "org", "type": "relay"}]})
        )
    with pytest.raises(SystemExit):
        subject._load_project_from_file(str(path))
    assert json.loads(capsys.readouterr().out)["error_code"] == code


def test_project_without_version_is_upgraded_without_mutating_original_yaml(tmp_path):
    content = {"name": "study", "participants": [{"name": "site-1", "org": "org", "type": "client"}]}
    path = tmp_path / "project.yaml"
    path.write_text(yaml.safe_dump(content))
    project, _ = subject._load_project_from_file(str(path))
    assert project.name == "study"
    assert yaml.safe_load(path.read_text()) == content


@pytest.mark.parametrize(
    "case,expected",
    [("absent", "REQUEST_DIR_NOT_FOUND"), ("no-key", "REQUEST_DIR_INCOMPLETE"), ("mismatched", "REQUEST_DIR_MISMATCH")],
)
def test_explicit_request_folder_requires_matching_identity_and_local_key(tmp_path, capsys, case, expected):
    folder = tmp_path / "request"
    if case != "absent":
        folder.mkdir()
        (folder / "request.json").write_text(json.dumps({"request_id": "b" * 32}))
        if case == "mismatched":
            (folder / "site-1.key").write_text("private material")
    with pytest.raises(SystemExit):
        subject._resolve_request_dir(
            SimpleNamespace(request_dir=str(folder)),
            str(tmp_path / "signed.zip"),
            {"name": "site-1", "request_id": "a" * 32},
        )
    assert json.loads(capsys.readouterr().out)["error_code"] == expected


def test_request_folder_discovery_uses_original_request_id_and_rejects_symlink_alias(tmp_path, monkeypatch):
    folder = tmp_path / "site-1"
    folder.mkdir()
    (folder / "request.json").write_text(json.dumps({"request_id": "a" * 32}))
    (folder / "site-1.key").write_text("private material")
    monkeypatch.setattr(subject, "_audit_request_dir", Mock(return_value=None))
    identity = {"name": "site-1", "request_id": "a" * 32}
    assert subject._resolve_request_dir(
        SimpleNamespace(request_dir=None), str(tmp_path / "site-1.signed.zip"), identity
    ) == str(folder)
    alias = tmp_path / "alias"
    alias.symlink_to(folder, target_is_directory=True)
    assert subject._validated_audit_request_dir(str(alias), "a" * 32) is None
    assert subject._validated_audit_request_dir(str(folder), "b" * 32) is None
    assert subject._validated_audit_request_dir(7, "a" * 32) is None


@pytest.mark.parametrize(
    "server",
    [
        None,
        {"host": ""},
        {"host": "bad host", "fed_learn_port": 8002, "admin_port": 8003},
        {"host": "server", "fed_learn_port": True, "admin_port": 8003},
    ],
)
def test_signed_server_endpoint_rejects_ambiguous_or_invalid_routing_metadata(capsys, server):
    with pytest.raises(SystemExit):
        subject._signed_server_endpoint({"server": server})
    assert json.loads(capsys.readouterr().out)["error_code"] == "INVALID_SIGNED_ZIP"


def test_endpoint_resolution_uses_signed_profile_and_normalizes_scheme_and_security():
    metadata = {
        "scheme": " GRPC ",
        "default_connection_security": " TLS ",
        "server": {"host": "server.example.com", "fed_learn_port": 8002, "admin_port": 8003},
    }
    result = subject._resolve_packaging_endpoint(SimpleNamespace(), metadata, {"name": "site-1"}, "client")
    assert result == ("grpc", "server.example.com", 8002, 8003, "tls")
    assert (
        subject._make_package_result("/kits", "site-1", "client", "grpc://server:8002")["next_step"]
        == "cd /kits/site-1 && ./startup/start.sh"
    )
    assert subject._make_package_result("/kits", "user@example.com", "lead", "")["next_step"].endswith(
        "./startup/fl_admin.sh"
    )


@pytest.mark.parametrize("content", [None, "[", "- not-a-mapping"])
def test_local_site_reader_reports_missing_parse_and_shape_failures(tmp_path, capsys, content):
    path = tmp_path / "site.yaml"
    if content is not None:
        path.write_text(content)
    with pytest.raises(SystemExit):
        subject._read_yaml_file_nofollow(str(path), code="LOCAL_SITE_INVALID", hint="original request")
    assert json.loads(capsys.readouterr().out)["error_code"] == "LOCAL_SITE_INVALID"


def test_local_request_metadata_reader_rejects_json_array(tmp_path, capsys):
    (tmp_path / "request.json").write_text("[]")
    with pytest.raises(SystemExit):
        subject._read_local_request_metadata(str(tmp_path))
    assert json.loads(capsys.readouterr().out)["error_code"] == "REQUEST_METADATA_INVALID"
