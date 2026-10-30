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

import datetime
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from nvflare.tool import cli_output
from nvflare.tool.cert import cert_commands as subject


@pytest.mark.parametrize("aware", [False, True])
def test_legacy_certificate_dates_are_normalized_to_utc(aware):
    zone = datetime.timezone(datetime.timedelta(hours=3)) if aware else None
    value = datetime.datetime(2030, 1, 2, 3, tzinfo=zone)
    cert = SimpleNamespace(not_valid_before=value, not_valid_after=value)
    expected = value.astimezone(datetime.timezone.utc) if aware else value.replace(tzinfo=datetime.timezone.utc)
    assert subject._get_cert_not_valid_before(cert) == expected
    assert subject._get_cert_not_valid_after(cert) == expected


@pytest.mark.parametrize("kind", ["request", "approve"])
def test_optional_audit_failure_warns_without_losing_command_success(monkeypatch, capsys, kind):
    monkeypatch.setattr(cli_output, "_output_format", "human")
    writer = Mock(side_effect=OSError("audit disk full"))
    monkeypatch.setattr(subject, f"_write_{kind}_audit", writer)
    assert getattr(subject, f"_try_write_{kind}_audit")("id", {"status": "signed"}) is None
    writer.assert_called_once_with("id", {"status": "signed"})
    assert "audit disk full" in capsys.readouterr().out


@pytest.mark.parametrize(
    "writer,args", [("_write_private_key", (b"secret",)), ("_write_json_file", ({"name": "site"},))]
)
def test_file_wrapper_failure_closes_descriptor_and_removes_partial_secret(tmp_path, monkeypatch, writer, args):
    path = tmp_path / "partial"
    real_close = subject.os.close
    closed = []

    def close(fd):
        closed.append(fd)
        return real_close(fd)

    monkeypatch.setattr(subject.os, "close", close)
    monkeypatch.setattr(subject.os, "fdopen", Mock(side_effect=OSError("fdopen rejected")))
    with pytest.raises(OSError, match="fdopen rejected"):
        getattr(subject, writer)(str(path), *args)
    assert len(closed) == 1
    assert not path.exists()


@pytest.mark.parametrize("failure", ["changed", "oversized"])
def test_archive_source_is_rejected_when_identity_or_size_changes(tmp_path, monkeypatch, failure):
    path = tmp_path / "request.json"
    path.write_bytes(b"safe")
    original = path.stat()
    opened = SimpleNamespace(
        st_ino=original.st_ino + int(failure == "changed"),
        st_dev=original.st_dev,
        st_mode=original.st_mode,
        st_size=subject._MAX_ZIP_MEMBER_SIZE + 1 if failure == "oversized" else 4,
    )
    monkeypatch.setattr(subject.os, "fstat", Mock(return_value=opened))
    with pytest.raises(subject._UnsafeZipSourceError, match="changed|too large"):
        subject._read_zip_source_nofollow(str(path))
    assert path.read_bytes() == b"safe"


def test_archive_wrapper_failure_removes_partial_output(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli_output, "_output_format", "json")
    output = tmp_path / "request.zip"
    source = tmp_path / "request.json"
    source.write_text("{}")
    monkeypatch.setattr(subject.zipfile, "ZipFile", Mock(side_effect=OSError("cannot archive")))
    with pytest.raises(SystemExit) as error:
        subject._write_zip_nofollow(str(output), {"request.json": str(source)})
    assert error.value.code == 1
    assert not output.exists()
    assert "OUTPUT_DIR_NOT_WRITABLE" in capsys.readouterr().out
