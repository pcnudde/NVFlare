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

import errno
import json
import os
import stat
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from nvflare.client.cell import attach_rendezvous as module


@pytest.mark.parametrize("site", ["", None, "site.job", "bad name"])
def test_rendezvous_requires_one_valid_site_segment(site):
    with pytest.raises(ValueError, match="site_name"):
        module.attach_claim_dir("/root", site, "trainer_a")


@pytest.mark.parametrize("path", ["", None, "relative"])
def test_rendezvous_requires_absolute_nonempty_trust_root(path):
    with pytest.raises(ValueError, match="rendezvous_dir"):
        module.attach_claim_dir(path, "site-1", "trainer_a")


@pytest.mark.parametrize("stage", ["mode", "serialization", "replace"])
def test_atomic_endpoint_write_cleans_partial_file_and_preserves_existing_record(monkeypatch, tmp_path, stage):
    path = tmp_path / "endpoint.json"
    path.write_text('{"previous":true}')
    allocated = []
    original = module.tempfile.mkstemp

    def allocate(**kwargs):
        result = original(**kwargs)
        allocated.append(result)
        return result

    monkeypatch.setattr(module, "tempfile", SimpleNamespace(mkstemp=allocate))
    local_os = SimpleNamespace(**vars(os))
    monkeypatch.setattr(module, "os", local_os)
    error = OSError("write failed")
    if stage == "mode":
        local_os.fchmod = MagicMock(side_effect=error)
    elif stage == "serialization":

        def failing_dump(data, stream, **kwargs):
            stream.write("partial")
            raise error

        monkeypatch.setattr(module, "json", SimpleNamespace(dump=failing_dump))
    else:
        local_os.replace = MagicMock(side_effect=error)
    with pytest.raises(OSError, match="write failed"):
        module._atomic_write_json(str(path), {"new": 1}, 0o600)
    assert json.loads(path.read_text()) == {"previous": True}
    assert sorted(item.name for item in tmp_path.iterdir()) == ["endpoint.json"]
    assert len(allocated) == 1
    with pytest.raises(OSError):
        os.fstat(allocated[0][0])


@pytest.mark.parametrize("contents", ["[]", "invalid json"])
def test_json_readers_reject_non_object_or_corrupt_owner_records(tmp_path, contents):
    path = tmp_path / "owner.json"
    path.write_text(contents)
    assert module._read_json(str(path)) is None
    with pytest.raises((module.AttachEndpointOwnershipError, json.JSONDecodeError)):
        module._read_json_strict(str(path))


def _record():
    key = module.AttachEndpointKey
    return {
        key.SCHEMA_VERSION: module.ATTACH_ENDPOINT_SCHEMA_VERSION,
        key.INSTANCE_ID: "instance",
        key.ATTACH_ID: "trainer_a",
        key.SITE_NAME: "site-1",
        key.CJ_FQCN: "site-1.job",
        key.TRAINER_FQCN: "site-1.-client_api_trainer_a",
        key.CONNECT_URL: "shared-file://0/tmp/listener",
        key.CONNECTION_SECURITY: "clear",
    }


@pytest.mark.parametrize(
    "field,value",
    [
        ("schema_version", 2),
        ("site_name", "other"),
        ("attach_id", "other"),
        ("instance_id", ""),
        ("cj_fqcn", "site-1"),
        ("cj_fqcn", "other.job"),
        ("trainer_fqcn", "site-1.wrong"),
        ("connect_url", ""),
        ("connect_url", "tcp://127.0.0.1:1"),
        ("connection_security", "tls"),
    ],
)
def test_endpoint_record_rejects_schema_identity_ownership_and_transport_mismatches(field, value):
    record = _record()
    record[field] = value
    with pytest.raises(ValueError):
        module._validate_endpoint_record(record, "site-1", "trainer_a")


def test_valid_endpoint_record_is_copied_without_mutating_discovered_record():
    record = _record()
    result = module._validate_endpoint_record(record, "site-1", "trainer_a")
    assert result == record and result is not record


@pytest.mark.parametrize(
    "mode,private,file",
    [
        (stat.S_IFREG | 0o600, True, False),
        (stat.S_IFDIR | 0o707, True, False),
        (stat.S_IFDIR | 0o777, False, False),
        (stat.S_IFDIR | 0o700, True, True),
        (stat.S_IFREG | 0o644, True, True),
    ],
)
def test_directory_and_private_file_permissions_fail_closed(monkeypatch, mode, private, file):
    local_os = SimpleNamespace(**vars(os))
    local_os.lstat = lambda path: SimpleNamespace(st_mode=mode)
    monkeypatch.setattr(module, "os", local_os)
    with pytest.raises(RuntimeError):
        if file:
            module._validate_private_file("path", "endpoint")
        else:
            module._validate_real_dir("path", private=private, label="trust root")


def _fake_lock_boundary(monkeypatch, *, flock_error=None, open_error=None, mode=None):
    local_os = SimpleNamespace(**vars(os))
    local_os.open = MagicMock(side_effect=open_error, return_value=17)
    local_os.close = MagicMock()
    local_os.fchmod = MagicMock()
    local_os.fstat = lambda fd: SimpleNamespace(st_mode=mode if mode is not None else stat.S_IFREG | 0o660)
    flock = MagicMock(side_effect=flock_error)
    lock = SimpleNamespace(flock=flock, LOCK_EX=1, LOCK_NB=4, LOCK_UN=8)
    monkeypatch.setattr(module, "os", local_os)
    monkeypatch.setattr(module, "fcntl", lock)
    return local_os, flock


@pytest.mark.parametrize("error,expected", [(None, False), (errno.EAGAIN, True), (errno.EACCES, True)])
def test_publisher_liveness_uses_nonblocking_lock_and_always_closes_descriptor(monkeypatch, error, expected):
    local_os, flock = _fake_lock_boundary(monkeypatch, flock_error=OSError(error, "locked") if error else None)
    assert module._claim_has_live_publisher("/root", "site-1", "trainer_a") is expected
    local_os.close.assert_called_once_with(17)
    assert flock.call_args_list[0].args == (17, 5)
    assert flock.call_count == (1 if error else 2)
    if not error:
        assert flock.call_args_list[1].args == (17, 8)


@pytest.mark.parametrize("operation", ["acquire", "probe"])
def test_advisory_lock_failure_requires_working_cross_node_lock_support(monkeypatch, operation):
    local_os, _ = _fake_lock_boundary(monkeypatch, flock_error=OSError(errno.ENOSYS, "not supported"))
    with pytest.raises(RuntimeError, match="working cross-node advisory locks"):
        if operation == "acquire":
            module._acquire_claim_lock("/root", "trainer_a")
        else:
            module._claim_has_live_publisher("/root", "site-1", "trainer_a")
    local_os.close.assert_called_once_with(17)


@pytest.mark.parametrize("mode", [stat.S_IFDIR | 0o660, stat.S_IFREG | 0o666])
@pytest.mark.parametrize("operation", ["acquire", "probe"])
def test_lock_paths_reject_wrong_file_type_or_public_permissions_before_flock(monkeypatch, mode, operation):
    local_os, flock = _fake_lock_boundary(monkeypatch, mode=mode)
    with pytest.raises(RuntimeError):
        if operation == "acquire":
            module._acquire_claim_lock("/root", "trainer_a")
        else:
            module._claim_has_live_publisher("/root", "site-1", "trainer_a")
    flock.assert_not_called()
    local_os.close.assert_called_once_with(17)


@pytest.mark.parametrize("error", [errno.EACCES, errno.ELOOP, errno.EPERM])
def test_liveness_probe_rejects_unsafe_lock_open_without_owning_descriptor(monkeypatch, error):
    local_os, _ = _fake_lock_boundary(monkeypatch, open_error=OSError(error, "unsafe path"))
    with pytest.raises(RuntimeError, match="cannot safely open"):
        module._claim_has_live_publisher("/root", "site-1", "trainer_a")
    local_os.close.assert_not_called()


def test_lock_functions_reject_missing_posix_support(monkeypatch):
    monkeypatch.setattr(module, "fcntl", None)
    with pytest.raises(RuntimeError, match="POSIX"):
        module._acquire_claim_lock("/root", "trainer_a")
    with pytest.raises(RuntimeError, match="POSIX"):
        module._claim_has_live_publisher("/root", "site-1", "trainer_a")
