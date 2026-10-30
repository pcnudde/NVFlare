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

from unittest.mock import Mock

import pytest

from nvflare.apis.fl_snapshot import RunSnapshot
from nvflare.apis.storage import DATA, StorageException
from nvflare.app_common.state_persistors.storage_state_persistor import StorageStatePersistor
from nvflare.app_common.storages import filesystem_storage as subject


def test_storage_state_snapshots_round_trip_update_completion_and_delete(tmp_path):
    storage = subject.FilesystemStorage(str(tmp_path), uri_root="/states")
    persistor = StorageStatePersistor(storage, "/states")
    snapshots = []
    for job in ["first", "second"]:
        snapshot = RunSnapshot(job)
        snapshot.set_component_snapshot("controller", {"round": 3, "weights": [1, 2]})
        persistor.save(snapshot)
        snapshots.append(snapshot)
    loaded = persistor.retrieve()
    assert set(loaded.run_snapshots) == {"first", "second"}
    assert loaded.get_snapshot("first").get_component_snapshot("controller") == {"round": 3, "weights": [1, 2]}
    assert persistor.retrieve_run("second").job_id == "second"
    snapshots[0].completed = True
    assert persistor.save(snapshots[0]) == str(tmp_path / "first")
    assert storage.list_objects("/states") == ["/states/second"]
    persistor.delete_run("second")
    assert storage.list_objects("/states") == []
    snapshots[0].completed = False
    for snapshot in snapshots:
        persistor.save(snapshot)
    persistor.delete()
    assert persistor.retrieve().run_snapshots == {}


@pytest.mark.parametrize("operation", ["create", "clone"])
def test_failed_metadata_write_removes_orphan_data(tmp_path, monkeypatch, operation):
    storage = subject.FilesystemStorage(str(tmp_path))
    storage.create_object("/source", b"source", {"version": 1})
    original = subject._write

    def write(path, content, mv_file=True):
        if path.endswith("/target/meta"):
            raise StorageException("metadata")
        return original(path, content, mv_file)

    monkeypatch.setattr(subject, "_write", write)
    with pytest.raises(StorageException, match="metadata"):
        if operation == "create":
            storage.create_object("/target", b"target", {})
        else:
            storage.clone_object("/source", "/target", {})
    assert not (tmp_path / "target" / "data").exists()
    assert storage.get_data("/source") == b"source"


@pytest.mark.parametrize("operation", ["get_meta", "list_components_of_object", "get_detail", "delete_object"])
def test_missing_objects_fail_without_creating_directories(tmp_path, operation):
    storage = subject.FilesystemStorage(str(tmp_path))
    with pytest.raises(StorageException, match="does not exist"):
        getattr(storage, operation)("/missing")
    assert not (tmp_path / "missing").exists()


def test_invalid_storage_components_cannot_read_write_or_download(tmp_path):
    storage = subject.FilesystemStorage(str(tmp_path))
    storage.create_object("/object", b"data", {})
    for operation in [storage.get_data, storage.update_object, storage.get_data_for_download]:
        kwargs = {"uri": "/object", "component_name": "../data"}
        if operation == storage.update_object:
            kwargs["data"] = b"replacement"
        elif operation == storage.get_data_for_download:
            kwargs["download_file"] = str(tmp_path / "download")
        with pytest.raises(StorageException, match="not a valid component"):
            operation(**kwargs)
    assert storage.get_data("/object") == b"data"


def test_download_replaces_prior_file_with_symlink_and_handles_absent_optional_component(tmp_path):
    storage = subject.FilesystemStorage(str(tmp_path / "store"))
    storage.create_object("/object", b"new", {})
    destination = tmp_path / "download"
    destination.write_bytes(b"old")
    storage.get_data_for_download("/object", DATA, str(destination))
    assert destination.is_symlink()
    assert destination.read_bytes() == b"new"
    destination.unlink()
    storage.get_data_for_download("/object", "workspace", str(destination))
    assert not destination.exists()
    with pytest.raises(StorageException, match="does not exist"):
        storage.get_data_for_download("/missing", DATA, str(destination))


def test_clone_rejects_existing_and_nonempty_targets_and_storage_requires_directory(tmp_path):
    storage = subject.FilesystemStorage(str(tmp_path))
    storage.create_object("/source", b"data", {})
    with pytest.raises(StorageException, match="already exists"):
        storage.clone_object("/source", "/source", {})
    directory = tmp_path / "nonempty"
    directory.mkdir()
    (directory / "file").write_bytes(b"unrelated")
    with pytest.raises(StorageException, match="nonempty"):
        storage.clone_object("/source", "/nonempty", {})
    with pytest.raises(ValueError, match="not a directory"):
        subject.FilesystemStorage(str(directory / "file"))


def test_atomic_write_failure_removes_temporary_file_and_preserves_existing_bytes(tmp_path, monkeypatch):
    destination = tmp_path / "data"
    destination.write_bytes(b"old")
    monkeypatch.setattr(subject.os, "replace", Mock(side_effect=OSError("rename")))
    with pytest.raises(StorageException, match="rename"):
        subject._write(str(destination), b"new")
    assert list(tmp_path.iterdir()) == [destination]
    assert destination.read_bytes() == b"old"
