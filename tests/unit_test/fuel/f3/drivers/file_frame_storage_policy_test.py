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

import os
import threading
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from nvflare.fuel.f3.comm_error import CommError
from nvflare.fuel.f3.drivers import file_driver as module
from nvflare.fuel.f3.drivers.driver_params import DriverParams
from nvflare.fuel.f3.sfm.prefix import PREFIX_LEN, Prefix


@pytest.mark.parametrize("path", [None, "relative", "/absolute/path///"])
def test_file_connector_requires_absolute_directory_and_normalizes_trailing_slashes(path):
    params = {DriverParams.PATH.value: path}
    if path and path.startswith("/"):
        assert module.FileDriver._get_dir_from_params(params) == "/absolute/path"
    else:
        with pytest.raises(CommError, match="Missing directory path"):
            module.FileDriver._get_dir_from_params(params)


def test_log_writer_persists_fsync_requested_data_and_rejects_append_after_close(tmp_path, monkeypatch):
    cfg = SimpleNamespace(file_mode=0o600, fsync=True, max_log_size=100)
    writer = module._LogWriter(str(tmp_path), "active", cfg)
    proxy = SimpleNamespace(**vars(os))
    proxy.fsync = Mock()
    monkeypatch.setattr(module, "os", proxy)
    try:
        writer.append(b"data")
        assert (tmp_path / "active.0.log").read_bytes() == b"data"
        assert writer.size == 4
        proxy.fsync.assert_called_once_with(writer.file.fileno())
    finally:
        writer.close()
    with pytest.raises(CommError, match="closed"):
        writer.append(b"other")
    assert module._parse_mode(0o700, 0o777) == 0o700


def test_corrupt_frame_length_is_rejected_before_payload_dispatch(tmp_path):
    cfg = SimpleNamespace()
    reader = module._LogReader(str(tmp_path), "active", cfg)
    data = bytearray(PREFIX_LEN)
    Prefix(length=1, header_len=0, type=1).to_buffer(data, 0)
    reader._read_new_data = Mock(return_value=data)
    with pytest.raises(CommError, match="Corrupt frame length"):
        reader.read_frames()


def test_garbage_collection_removes_only_stale_temporary_connection_directories(tmp_path, monkeypatch):
    old = tmp_path / (module.TMP_PREFIX + "old")
    fresh = tmp_path / (module.TMP_PREFIX + "fresh")
    other = tmp_path / "unrelated"
    for path in (old, fresh, other):
        path.mkdir()
    driver = module.FileDriver.__new__(module.FileDriver)
    driver.dir_lock, driver.done_dirs, driver.handled_dirs = threading.Lock(), {}, set()
    monkeypatch.setattr(
        module, "time", SimpleNamespace(monotonic=lambda: 100, time=lambda: module.TMP_DIR_STALE_TIME + 100)
    )
    monkeypatch.setattr(module, "_mtime", lambda path: 0 if path == str(old) else module.TMP_DIR_STALE_TIME + 90)
    driver._gc_conn_dirs(str(tmp_path), SimpleNamespace(lease_timeout=5), [p.name for p in (old, fresh, other)])
    assert not old.exists() and fresh.exists() and other.exists()


def test_stale_listener_scan_ignores_unavailable_directory(monkeypatch):
    proxy = SimpleNamespace(**vars(os))
    proxy.listdir = Mock(side_effect=PermissionError("unreadable"))
    monkeypatch.setattr(module, "os", proxy)
    worker = Mock()
    monkeypatch.setattr(module, "threading", SimpleNamespace(Thread=worker))
    module._remove_stale_listener_dirs("/unreadable")
    worker.assert_not_called()
