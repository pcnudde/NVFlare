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

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from pyhocon import ConfigFactory

from nvflare.tool.kit import kit_config as subject


def test_config_parse_failure_keeps_original_file_and_actionable_hint(tmp_path, monkeypatch):
    path = tmp_path / "config.conf"
    path.write_text("startup_kits = {")
    monkeypatch.setattr(subject, "get_cli_config_path", lambda: path)
    with pytest.raises(subject.StartupKitConfigError, match="cannot parse") as error:
        subject.load_cli_config()
    assert "Fix the config file" in error.value.hint
    assert path.read_text() == "startup_kits = {"


def test_atomic_config_replace_failure_preserves_previous_config_and_removes_temporary_file(tmp_path, monkeypatch):
    path = tmp_path / "config.conf"
    path.write_text("original")
    monkeypatch.setattr(subject, "get_cli_config_path", lambda: path)
    monkeypatch.setattr(subject.os, "replace", Mock(side_effect=OSError("replace denied")))
    with pytest.raises(OSError, match="replace denied"):
        subject.save_cli_config(ConfigFactory.from_dict({"version": 2}))
    assert path.read_text() == "original"
    assert list(tmp_path.iterdir()) == [path]


@pytest.mark.parametrize("legacy", [False, True])
def test_certificate_expiring_soon_metadata_normalizes_naive_utc_dates(legacy):
    expiry = datetime.now(timezone.utc).replace(microsecond=0) + timedelta(days=1)
    cert = SimpleNamespace(**{"not_valid_after" if legacy else "not_valid_after_utc": expiry.replace(tzinfo=None)})
    metadata, findings = subject._certificate_expiration_metadata(cert, "client.crt")
    assert metadata["status"] == "expiring_soon" and metadata["expires_at"].endswith("Z")
    assert findings[0]["code"] == "STARTUP_KIT_CERT_EXPIRING_SOON"
    assert findings[0]["severity"] == "warning"


def test_missing_certificate_and_missing_inspection_dependency_are_separate_findings(tmp_path, monkeypatch):
    metadata = subject._empty_startup_kit_metadata("admin")
    subject._inspect_admin_cert_metadata(str(tmp_path), metadata)
    assert metadata["findings"][0]["code"] == "STARTUP_KIT_CERT_MISSING"
    (tmp_path / "client.crt").write_bytes(b"certificate")
    monkeypatch.setattr(subject, "x509", None)
    metadata = subject._empty_startup_kit_metadata("admin")
    subject._inspect_admin_cert_metadata(str(tmp_path), metadata)
    assert metadata["certificate"]["status"] == "unknown"
    assert metadata["findings"][0]["code"] == "STARTUP_KIT_CERT_UNREADABLE"


def test_empty_registry_id_and_legacy_quoted_key_have_explicit_contracts():
    with pytest.raises(subject.StartupKitConfigError, match="cannot be empty"):
        subject._normalize_kit_id("   ")
    assert subject._decode_hocon_key('"bad\\escape"') == "bad\\escape"
    config = ConfigFactory.from_dict({"version": 2, "startup_kits": {"entries": "legacy"}})
    assert subject.get_startup_kit_entries(config) == {}
