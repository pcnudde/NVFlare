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

import datetime
import json
import os
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import yaml
from cryptography.exceptions import InvalidSignature

from nvflare.lighter import utils


def test_sign_all_authenticates_files_and_ignores_nested_directories(tmp_path):
    (tmp_path / "one").write_bytes(b"first")
    (tmp_path / "two").write_bytes(b"second")
    (tmp_path / "nested").mkdir()
    (tmp_path / "nested" / "ignored").write_bytes(b"nested")
    private, public = utils.generate_keys()
    signatures = utils.sign_all(str(tmp_path), private)
    assert set(signatures) == {"one", "two"}
    for name, signature in signatures.items():
        assert utils.verify_content((tmp_path / name).read_bytes(), signature, public) is None
    with pytest.raises(InvalidSignature):
        utils.verify_content(b"tampered", signatures["one"], public)
    with pytest.raises(TypeError, match="passphrase"):
        utils.serialize_pri_key(private, "not-bytes")


def test_resource_storage_locations_and_component_merge_preserve_existing_entries(tmp_path):
    original = {
        "snapshot_persistor": {"args": {"storage": {"args": {"root_dir": "old"}}}},
        "components": [{"id": "other", "args": {}}, {"id": "job_manager", "args": {"uri_root": "old"}}],
    }
    (tmp_path / "resources.json.default").write_text(json.dumps(original))
    utils.update_storage_locations(
        str(tmp_path), "/workspace", job_storage_name="jobs", snapshot_storage_name="snapshots"
    )
    result = json.loads((tmp_path / "resources.json").read_text())
    assert result["snapshot_persistor"]["args"]["storage"]["args"]["root_dir"] == "/workspace/snapshots"
    assert result["components"][1]["args"]["uri_root"] == "/workspace/jobs"
    assert json.loads((tmp_path / "resources.json.default").read_text()) == original
    destination = tmp_path / "components.json"
    utils.add_component_to_resources(str(destination), {"id": "first"})
    utils.add_component_to_resources(str(destination), {"id": "second"})
    assert json.loads(destination.read_text()) == {"components": [{"id": "first"}, {"id": "second"}]}


def test_server_name_file_updates_preserve_other_participants_and_default_host(tmp_path):
    path = tmp_path / "project.yml"
    config = {"participants": [{"type": "client", "name": "client"}, {"type": "server", "name": "old"}]}
    path.write_text(yaml.safe_dump(config))
    utils.update_project_server_name(str(path), "old", "new")
    result = yaml.safe_load(path.read_text())
    assert [p["name"] for p in result["participants"]] == ["client", "new"]
    assert utils.update_server_default_host(result, "host.example") is result
    assert result["participants"][1]["default_host"] == "host.example"
    path.write_text("")
    with pytest.raises(RuntimeError, match="project_config is empty"):
        utils.update_project_server_name(str(path), "old", "new")


def test_yaml_bytes_and_custom_path_type_have_explicit_loading_contract():
    assert utils.load_yaml(b"key: value") == {"key": "value"}

    class CustomPath:
        def __fspath__(self):
            return "custom.yml"

    with pytest.raises(ValueError, match="Invalid file type"):
        utils.load_yaml(CustomPath())


def test_include_path_policy_contains_cross_drive_commonpath_failure(monkeypatch, tmp_path):
    common = Mock(side_effect=ValueError("different drives"))
    proxy = SimpleNamespace(path=SimpleNamespace(realpath=os.path.realpath, join=os.path.join, commonpath=common))
    monkeypatch.setattr(utils, "os", proxy)
    with pytest.raises(ValueError, match="include path escapes root"):
        utils._resolve_include_path(str(tmp_path), "include.yml")
    common.assert_called_once()


def test_expired_issuer_and_missing_san_fallback_are_rejected():
    issuer = SimpleNamespace(
        not_valid_after_utc=datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=1)
    )
    with pytest.raises(ValueError, match="expired"):
        utils.bounded_validity(issuer, 7)
    with pytest.raises(ValueError, match="fallback_subject_name"):
        utils.build_subject_alt_names()
    names = utils.build_subject_alt_names("host.example", ["host.example", "second.example"])
    assert [name.value for name in names] == ["host.example", "second.example"]
