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

import io
import json
import zipfile
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from nvflare.apis.fl_constant import ConnPropKey
from nvflare.app_opt.job_launcher import workspace_cell_transfer as subject
from nvflare.fuel.f3.cellnet.defs import MessageHeaderKey, ReturnCode
from nvflare.fuel.f3.cellnet.utils import new_cell_message


@pytest.mark.parametrize("path", [None, "", ".", "a/..", "/absolute", "../escape", "a/../../escape"])
def test_workspace_relative_paths_reject_empty_absolute_and_escaping_locations(path):
    assert subject._safe_rel_path(path) is None


def test_workspace_configuration_paths_and_server_identity_mapping_stay_within_known_roots(tmp_path):
    assert subject._config_path_to_workspace_rel(None, str(tmp_path)) is None
    assert subject._config_path_to_workspace_rel("local/model.yaml", str(tmp_path)) == "local/model.yaml"
    assert subject._config_path_to_workspace_rel("/external/file", str(tmp_path)) is None
    assert subject._server_auth_identity([]) is None
    assert subject._server_auth_identity({ConnPropKey.IDENTITY: ""}) is None
    (tmp_path / "fed_server.json").write_text(
        json.dumps(
            {
                "servers": [
                    {
                        ConnPropKey.AUTH_IDENTITY: "server.example",
                        ConnPropKey.AUTH_IDENTITY_MAP: {"site": "client.example"},
                    }
                ]
            }
        )
    )
    identities = subject._bootstrap_auth_identity_map(str(tmp_path))
    assert identities == {"server": "server.example", "site": "client.example"}
    (tmp_path / "fed_client.json").write_text("not-json")
    assert subject._load_startup_json(str(tmp_path), "fed_client.json") is None


@pytest.mark.parametrize(
    "resources,runtime",
    [
        ("broken-json", "[invalid"),
        ([], []),
        ({"components": [1, {"id": "other"}, {"id": "k8s_launcher", "args": []}]}, {"studies": []}),
        ({"components": []}, {"studies": {"invalid": 4}}),
    ],
)
def test_optional_study_exclusion_config_failures_preserve_default_excludes(tmp_path, resources, runtime):
    local = tmp_path / "local"
    local.mkdir()
    (local / "resources.json").write_text(resources if isinstance(resources, str) else json.dumps(resources))
    (local / "study_runtime.yaml").write_text(runtime if isinstance(runtime, str) else json.dumps(runtime))
    excludes = subject._workspace_download_excludes(str(tmp_path))
    assert {"local/study_data.yaml", "local/study_runtime.yaml"}.issubset(excludes)


def test_zip_validation_rejects_path_escape_before_extracting_and_missing_source_is_empty(tmp_path):
    data = io.BytesIO()
    with zipfile.ZipFile(data, "w") as archive:
        subject._write_dir_to_zip(archive, str(tmp_path / "absent"), str(tmp_path))
        assert archive.namelist() == []
        archive.writestr("../outside", "secret")
    data.seek(0)
    with zipfile.ZipFile(data) as archive, pytest.raises(ValueError, match="unsafe zip member"):
        subject._validate_relative_zip_members(archive)
    missing = tmp_path / "missing"
    subject._cleanup_transfer_files("tx", "complete", [], temp_paths=["", str(missing)])
    assert not missing.exists()


def test_bootstrap_cell_is_created_once_and_cleanup_releases_both_owners_even_if_they_fail(monkeypatch):
    cell, agent = Mock(), Mock()
    cell.stop.side_effect = RuntimeError("cell failure")
    agent.close.side_effect = RuntimeError("agent failure")
    create = Mock(return_value=(cell, agent))
    monkeypatch.setattr(subject, "_bootstrap_cell", None)
    monkeypatch.setattr(subject, "_bootstrap_net_agent", None)
    monkeypatch.setattr(subject, "_create_bootstrap_cell", create)
    args = object()
    assert subject._get_bootstrap_cell(args, "site", False) is cell
    assert subject._get_bootstrap_cell(args, "site", False) is cell
    create.assert_called_once_with(args=args, owner_fqcn="site", secure_mode=False)
    subject._close_bootstrap_cell()
    cell.stop.assert_called_once()
    agent.close.assert_called_once()
    assert subject._bootstrap_cell is None and subject._bootstrap_net_agent is None
    assert subject._get_root_url(SimpleNamespace(sp_scheme="tcp", sp_target="host:9000")) == "tcp://host:9000"
    with pytest.raises(RuntimeError, match="unable to determine root_url"):
        subject._get_root_url(SimpleNamespace())


@pytest.fixture
def manager(tmp_path, monkeypatch):
    cell = Mock()
    cell.get_fqcn.return_value = "site"
    cleanup = Mock()
    monkeypatch.setattr(subject, "_cleanup_download", cleanup)
    result = subject.WorkspaceTransferManager(cell)
    token = result.add_job("job", str(tmp_path / "workspace"))
    yield result, token, cleanup
    result.remove_job("job")


@pytest.mark.parametrize(
    "body,reason",
    [
        (None, "payload must be dict"),
        ({}, "missing job_id"),
        ({"job_id": "other", "transfer_token": "token"}, "unknown job_id"),
    ],
)
def test_workspace_transfer_authorization_rejects_missing_or_unknown_job_without_download(manager, body, reason):
    item, _, _ = manager
    record, origin, error = item._authorize(new_cell_message({MessageHeaderKey.ORIGIN: "pod"}, body), "download")
    assert record is None and origin == "pod"
    assert error.get_header(MessageHeaderKey.RETURN_CODE) != ReturnCode.OK
    assert reason in error.get_header(MessageHeaderKey.ERROR)


@pytest.mark.parametrize("failure", ["missing-ref", "download", "checksum", "zip"])
def test_result_publish_failures_preserve_job_and_do_not_replace_workspace(manager, tmp_path, monkeypatch, failure):
    item, token, _ = manager
    bundle = tmp_path / "result.zip"
    bundle.write_bytes(b"not-a-zip")
    download = Mock(return_value=("lost" if failure == "download" else None, str(bundle)))
    monkeypatch.setattr(subject, "download_file", download)
    body = {"job_id": "job", "transfer_token": token}
    if failure != "missing-ref":
        body["ref_id"] = "ref"
    if failure == "checksum":
        body["sha256"] = "different"
    response = item._handle_publish_results(new_cell_message({MessageHeaderKey.ORIGIN: "pod"}, body))
    assert response.get_header(MessageHeaderKey.RETURN_CODE) != ReturnCode.OK
    assert "job" in item.jobs
    if failure == "missing-ref":
        download.assert_not_called()
    else:
        assert download.call_args.kwargs["from_fqcn"] == "pod"
    assert not (tmp_path / "workspace" / "results").exists()


def test_replaced_job_and_download_completion_cleanup_only_matching_transfer(manager, tmp_path):
    item, _, cleanup = manager
    old = item.jobs["job"]
    old.download_tx_id, old.download_bundle_path = "old", "old.zip"
    replacement = item.add_job("job", str(tmp_path / "new"))
    assert replacement != old.transfer_token
    cleanup.assert_called_once_with("old", "old.zip")
    current = item.jobs["job"]
    current.download_tx_id, current.download_bundle_path = "new", "new.zip"
    file = tmp_path / "complete.zip"
    file.write_bytes(b"complete")
    item._download_transaction_done("old", "complete", [str(file)], "job")
    assert not file.exists() and current.download_tx_id == "new"
    item._download_transaction_done("new", "complete", [], "job")
    assert current.download_tx_id == "" and current.download_bundle_path == ""


@pytest.mark.parametrize(
    "response,reason",
    [
        (new_cell_message({MessageHeaderKey.RETURN_CODE: ReturnCode.TIMEOUT}, {}), "preparation failed"),
        (new_cell_message({MessageHeaderKey.RETURN_CODE: ReturnCode.OK}, []), "invalid workspace download reply"),
    ],
)
def test_workspace_bundle_request_requires_success_and_mapping_payload(monkeypatch, response, reason):
    monkeypatch.setattr(subject, "_wait_for_bootstrap_ready", Mock())
    cell = Mock()
    cell.send_request.return_value = response
    with pytest.raises(RuntimeError, match=reason):
        subject._request_workspace_bundle(cell, "owner", "job", "token")
    assert cell.send_request.call_args.kwargs["request"].payload == {"job_id": "job", "transfer_token": "token"}
