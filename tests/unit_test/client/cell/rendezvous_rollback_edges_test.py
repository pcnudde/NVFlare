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
import os
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from nvflare.client.cell import attach_rendezvous as module


@pytest.mark.parametrize("failure_stage", ["mode", "replace"])
def test_failed_atomic_publication_preserves_prior_record_and_removes_temporary_file(
    tmp_path, monkeypatch, failure_stage
):
    target = tmp_path / "owner.json"
    target.write_text('{"owner":"old"}')
    local_os = SimpleNamespace(**vars(os))
    failing = Mock(side_effect=OSError("publication failure"))
    setattr(local_os, "fchmod" if failure_stage == "mode" else "replace", failing)
    monkeypatch.setattr(module, "os", local_os)
    with pytest.raises(OSError, match="publication failure"):
        module._atomic_write_json(str(target), {"owner": "new"}, 0o600)
    assert target.read_text() == '{"owner":"old"}'
    assert sorted(path.name for path in tmp_path.iterdir()) == ["owner.json"]
    if failure_stage == "mode":
        with pytest.raises(OSError) as error:
            os.fstat(failing.call_args.args[0])
        assert error.value.errno == errno.EBADF


def test_owner_record_requires_object_and_corruption_cannot_be_mistaken_for_valid_owner(tmp_path):
    path = tmp_path / "owner.json"
    path.write_text("[]")
    with pytest.raises(module.AttachEndpointOwnershipError, match="does not contain an object"):
        module._read_json_strict(str(path))
    assert module._read_json(str(path)) is None
    path.write_text("{")
    assert module._read_json(str(path)) is None


def test_failed_owner_publication_rolls_back_claim_directory_and_releases_lock(tmp_path, monkeypatch):
    if module.fcntl is None:
        pytest.skip("POSIX advisory locking is required for shared-file Attach")
    original_write = module._atomic_write_json
    failing = Mock(side_effect=OSError("owner disk full"))
    monkeypatch.setattr(module, "_atomic_write_json", failing)
    with pytest.raises(OSError, match="owner disk full"):
        module.AttachEndpointPublisher(str(tmp_path), "site", "trainer")
    assert not os.path.exists(module.attach_claim_dir(str(tmp_path), "site", "trainer"))
    monkeypatch.setattr(module, "_atomic_write_json", original_write)
    recovered = module.AttachEndpointPublisher(str(tmp_path), "site", "trainer")
    try:
        assert os.path.isdir(recovered.claim_dir)
        owner = tmp_path / module.ATTACH_RENDEZVOUS_SUBDIR / "site" / "trainer.claim" / module.ATTACH_OWNER_FILE
        owner.write_text("{")
        with pytest.raises(module.AttachEndpointOwnershipError, match="owner record is corrupt"):
            recovered._assert_owner()
    finally:
        recovered.close()


def test_unsupported_advisory_lock_error_does_not_leak_open_descriptor(tmp_path, monkeypatch):
    if module.fcntl is None:
        pytest.skip("POSIX advisory locking is required for shared-file Attach")
    captured = []

    def unsupported(fd, flags):
        captured.append(fd)
        raise OSError(errno.ENOSYS, "locking unavailable")

    lock_api = SimpleNamespace(LOCK_EX=module.fcntl.LOCK_EX, LOCK_NB=module.fcntl.LOCK_NB, flock=unsupported)
    monkeypatch.setattr(module, "fcntl", lock_api)
    with pytest.raises(RuntimeError, match="working cross-node advisory locks"):
        module._acquire_claim_lock(str(tmp_path), "trainer")
    with pytest.raises(OSError) as error:
        os.fstat(captured[0])
    assert error.value.errno == errno.EBADF
