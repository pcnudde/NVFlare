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

import pytest

from nvflare.apis.storage import StorageException
from nvflare.app_common.storages import filesystem_storage as subject


def test_storage_moves_file_content_and_preserves_tag_payload(tmp_path):
    root = tmp_path / "store"
    root.mkdir()
    source = tmp_path / "source"
    source.write_bytes(b"weights")
    storage = subject.FilesystemStorage(str(root))
    uri = storage.create_object("job", str(source), {"model": "saved"})
    assert not source.exists()
    assert storage.get_data("job") == b"weights"
    storage.tag_object("job", "ready", "complete")
    assert (root / "job" / "ready").read_text() == "complete"
    assert uri == str(root / "job")


def test_invalid_source_and_missing_update_directory_cannot_create_storage_object(tmp_path):
    root = tmp_path / "store"
    root.mkdir()
    storage = subject.FilesystemStorage(str(root))
    with pytest.raises(StorageException, match="not a valid file"):
        storage.create_object("job", str(tmp_path), {})
    with pytest.raises(StorageException, match="not a valid directory"):
        storage.update_object("absent", b"data")
    with pytest.raises(ValueError, match="absolute path"):
        subject.FilesystemStorage("relative")


def test_legacy_encoded_metadata_decodes_literal_dictionary():
    expected = {"round": 3, "active": True}
    assert subject._decode_meta(json.dumps(str(expected)).encode()) == expected
