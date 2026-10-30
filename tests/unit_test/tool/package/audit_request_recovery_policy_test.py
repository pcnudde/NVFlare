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

from nvflare.tool import cli_output
from nvflare.tool.package import package_commands as subject


@pytest.mark.parametrize("location", ["nested", "private-key", "legacy"])
def test_audit_discovery_recovers_only_matching_request_folder(tmp_path, monkeypatch, location):
    monkeypatch.setattr(
        subject.os.path, "expanduser", lambda path: str(tmp_path / path[2:]) if path.startswith("~/") else path
    )
    request_id = "a" * 32
    folder = tmp_path / "original"
    folder.mkdir()
    (folder / "request.json").write_text(json.dumps({"request_id": request_id}))
    base = tmp_path / ".nvflare" / "cert_requests"
    audit = base / request_id
    audit.mkdir(parents=True)
    (audit / "audit.json").write_text("invalid json")
    record = (
        {"request": {"request_folder": str(folder)}}
        if location == "nested"
        else {"private_key_path": str(folder / "site.key")}
    )
    path = base / (request_id + ".json") if location == "legacy" else audit / "request.json"
    path.write_text(json.dumps(record))
    assert subject._audit_request_dir(request_id) == str(folder)
    (folder / "request.json").write_text(json.dumps({"request_id": "other"}))
    assert subject._audit_request_dir(request_id) is None


def test_audit_recovery_ignores_invalid_json_and_absent_request_id(tmp_path):
    path = tmp_path / "request.json"
    path.write_text("[")
    assert subject._validated_audit_request_dir(str(tmp_path), "id") is None
    assert subject._audit_request_dir(None) is None


def test_materialization_preserves_existing_local_participant_definition(tmp_path):
    (tmp_path / "site.yaml").write_text("original definition")
    assert subject._write_materialized_signed_files(
        str(tmp_path),
        {"name": "site"},
        {"request_id": "id"},
        {"name": "different"},
        {"cert": b"certificate", "rootCA.pem": b"root"},
    )
    assert (tmp_path / "site.yaml").read_text() == "original definition"
    assert (tmp_path / "site.crt").read_bytes() == b"certificate"
    assert (tmp_path / "rootCA.pem").read_bytes() == b"root"
    assert json.loads((tmp_path / "signed.json").read_text()) == {"request_id": "id"}


def test_legacy_certificate_expiry_uses_naive_utc_comparison(monkeypatch):
    import datetime
    from types import SimpleNamespace

    cert = SimpleNamespace(not_valid_after=datetime.datetime(2099, 1, 1))
    root = Mock()
    loader = Mock(side_effect=[cert, root])
    verify = Mock()
    monkeypatch.setattr(subject, "_load_crt_nofollow", loader)
    monkeypatch.setattr(subject, "verify_cert", verify)
    assert subject._validate_cert_material("cert", "key", "ca") is cert
    verify.assert_called_once_with(cert, root.public_key())


def test_provision_failure_reports_builder_specific_diagnostic(tmp_path, monkeypatch, capsys):
    from types import SimpleNamespace

    monkeypatch.setattr(cli_output, "_output_format", "json")
    context = {
        subject.CtxKey.BUILD_ERROR: True,
        subject._PACKAGE_ERROR_CODE: "OUTPUT_DIR_EXISTS",
        subject._PACKAGE_ERROR_MESSAGE: "participant is already present",
        subject._PACKAGE_ERROR_HINT: "choose a fresh workspace",
        subject._PACKAGE_ERROR_EXIT_CODE: 4,
    }
    provisioner = Mock()
    provisioner.provision.return_value = context
    monkeypatch.setattr(subject, "Provisioner", Mock(return_value=provisioner))
    with pytest.raises(SystemExit) as error:
        subject._build_selected_participant_package(
            args=SimpleNamespace(workspace=str(tmp_path)),
            scheme="grpc",
            host="server",
            port=8002,
            name="site",
            org="org",
            kit_type="client",
            cert_path="cert",
            key_path="key",
            rootca_path="ca",
            project_name="study",
        )
    assert error.value.code == 4
    result = json.loads(capsys.readouterr().out)
    assert result["error_code"] == "OUTPUT_DIR_EXISTS"
    assert result["message"] == "participant is already present"
    assert provisioner.provision.call_args.args[0].get_clients()[0].name == "site"
