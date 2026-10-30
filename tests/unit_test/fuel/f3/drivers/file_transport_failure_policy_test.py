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
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from nvflare.fuel.f3.comm_error import CommError
from nvflare.fuel.f3.drivers import file_driver as subject
from nvflare.fuel.f3.drivers.connector_info import Mode
from tests.timing_utils import isolate_time


@pytest.fixture
def config(monkeypatch):
    monkeypatch.setattr(subject, "CommConfigurator", lambda: SimpleNamespace(get_config=lambda: {}))
    return subject._ConnConfig({"max_log_size": 10, "lease_timeout": 5})


def test_local_file_transport_tuning_overrides_polling_but_preserves_shared_rotation_size(monkeypatch):
    monkeypatch.setattr(
        subject,
        "CommConfigurator",
        lambda: SimpleNamespace(
            get_config=lambda: {
                subject.SCHEME: {
                    subject.POLL_INTERVAL: 0,
                    subject.MAX_LOG_SIZE: 999,
                    subject.FSYNC: True,
                    subject.DIR_MODE: "770",
                }
            }
        ),
    )
    config = subject._ConnConfig({subject.MAX_LOG_SIZE: 10})
    assert config.poll_interval == subject.MIN_POLL_INTERVAL
    assert config.max_log_size == 10
    assert config.fsync is True
    assert config.dir_mode == 0o770


def test_touch_contains_permission_errors_for_existing_and_missing_files(monkeypatch, tmp_path):
    os_proxy = SimpleNamespace(**vars(os))
    os_proxy.utime = Mock(side_effect=PermissionError("read only"))
    monkeypatch.setattr(subject, "os", os_proxy)
    subject._touch(str(tmp_path / "existing"))
    os_proxy.utime.side_effect = FileNotFoundError()
    subject._touch(str(tmp_path / "missing-parent" / "file"))
    assert os_proxy.utime.call_count == 2


def test_log_writer_rotation_failure_does_not_leave_closed_file_handle(config, tmp_path, monkeypatch):
    writer = subject._LogWriter(str(tmp_path), "active", config)
    writer.file = Mock()
    previous = writer.file
    os_proxy = SimpleNamespace(**vars(os))
    os_proxy.fsync = Mock(side_effect=OSError("fsync failed"))
    monkeypatch.setattr(subject, "os", os_proxy)
    with pytest.raises(OSError, match="fsync failed"):
        writer._rotate()
    writer.close()
    assert writer.closed and writer.file is None
    previous.close.assert_called_once_with()


def test_log_reader_ignores_unreadable_seal_and_rejects_partial_final_frame(config, tmp_path):
    reader = subject._LogReader(str(tmp_path), "active", config)
    sealed = tmp_path / "active.0.sealed"
    sealed.write_text("invalid")
    reader._advance_if_sealed()
    assert reader.index == 0
    sealed.write_text("12")
    reader.offset = 10
    reader._advance_if_sealed()
    assert reader.index == 0
    reader.offset = 12
    reader.buffer.extend(b"partial")
    with pytest.raises(CommError, match="Partial frame"):
        reader._advance_if_sealed()
    assert reader.index == 0


def test_log_reader_and_writer_close_contain_filesystem_errors(config, tmp_path):
    for cls in (subject._LogReader, subject._LogWriter):
        value = cls(str(tmp_path), "active", config)
        value.file = Mock()
        value.file.close.side_effect = OSError("already closed")
        value.close()
        assert value.file is None


@pytest.mark.parametrize("error", [OSError("write failed"), CommError(CommError.CLOSED, "already closed")])
def test_failed_file_frame_append_closes_connection_before_propagating_error(config, monkeypatch, error):
    connection = object.__new__(subject.FileConnection)
    connection.name = "connection"
    connection.closing = False
    connection.writer = Mock()
    connection.writer.append.side_effect = error
    connection.cfg = config
    connection.closed_file = "closed"
    touch = Mock()
    monkeypatch.setattr(subject, "_touch", touch)
    with pytest.raises(CommError):
        connection.send_frame(b"frame")
    assert connection.closing
    connection.writer.close.assert_called_once_with()
    touch.assert_called_once_with("closed", config.file_mode)
    with pytest.raises(CommError, match="closed"):
        connection.send_frame(b"again")
    assert connection.writer.append.call_count == 1


def test_adoption_ignores_temporary_and_stale_directories_and_starts_only_admitted_connection(
    config, tmp_path, monkeypatch
):
    for name in ("active", "rejected", "closed", "incomplete", ".temporary"):
        path = tmp_path / name
        path.mkdir()
        if name != "incomplete":
            (path / "active.0.log").write_bytes(b"")
    (tmp_path / "closed" / subject.CLOSED_FILE).write_bytes(b"")
    driver = subject.FileDriver()
    driver.add_connection = Mock(side_effect=[True, False])
    connection = Mock()
    monkeypatch.setattr(subject, "FileConnection", Mock(return_value=connection))
    thread = Mock()
    monkeypatch.setattr(subject, "threading", SimpleNamespace(Thread=Mock(return_value=thread)))
    clock = isolate_time(monkeypatch, subject)
    monkeypatch.setattr(clock, "monotonic", lambda: 100)
    connector = SimpleNamespace(mode=Mode.PASSIVE)
    adopted = driver._adopt_new_dirs(
        str(tmp_path), [".temporary", "missing", "closed", "incomplete", "active", "rejected"], connector, config
    )
    assert adopted == 1
    assert driver.handled_dirs == {"active", "rejected", "closed", "incomplete"}
    assert driver.done_dirs == {"closed": 100, "incomplete": 100, "rejected": 100}
    thread.start.assert_called_once_with()
    assert (tmp_path / "incomplete" / subject.CLOSED_FILE).exists()


def test_connection_loop_contains_reader_failure_and_records_cleanup_time(config, monkeypatch):
    driver = subject.FileDriver()
    driver.close_connection = Mock()
    connection = Mock()
    connection.read_loop.side_effect = OSError("peer gone")
    clock = isolate_time(monkeypatch, subject)
    monkeypatch.setattr(clock, "monotonic", lambda: 123)
    driver._conn_loop(connection, "entry")
    connection.close.assert_called_once_with()
    driver.close_connection.assert_called_once_with(connection)
    assert driver.done_dirs == {"entry": 123}


def test_connection_directory_gc_retains_failed_deletions_and_removes_expired_owned_entries(
    config, tmp_path, monkeypatch
):
    driver = subject.FileDriver()
    (tmp_path / "done").mkdir()
    driver.done_dirs = {"done": 0}
    driver.handled_dirs = {"done"}
    clock = isolate_time(monkeypatch, subject)
    monkeypatch.setattr(clock, "monotonic", lambda: 100)
    monkeypatch.setattr(clock, "time", lambda: 10000)
    real_shutil = subject.shutil
    monkeypatch.setattr(subject, "shutil", SimpleNamespace(rmtree=Mock()))
    driver._gc_conn_dirs(str(tmp_path), config, [])
    assert driver.done_dirs == {"done": 0} and driver.handled_dirs == {"done"}
    monkeypatch.setattr(subject, "shutil", real_shutil)
    driver._gc_conn_dirs(str(tmp_path), config, [])
    assert not driver.done_dirs and not driver.handled_dirs
    assert not (tmp_path / "done").exists()


def test_stale_listener_gc_only_schedules_owned_expired_directories(tmp_path, monkeypatch):
    for name in ("lst_owned", "lst_unowned", "unrelated"):
        (tmp_path / name).mkdir()
    (tmp_path / "lst_owned" / subject.OWNER_MARKER).write_bytes(b"")
    clock = isolate_time(monkeypatch, subject)
    monkeypatch.setattr(clock, "time", lambda: 10000)
    monkeypatch.setattr(subject, "_mtime", lambda path: 1)
    factory = Mock()
    monkeypatch.setattr(subject, "threading", SimpleNamespace(Thread=factory))
    subject._remove_stale_listener_dirs(str(tmp_path))
    factory.assert_called_once()
    assert factory.call_args.kwargs["args"] == (str(tmp_path / "lst_owned"),)
    assert factory.call_args.kwargs["kwargs"] == {"ignore_errors": True}
    factory.return_value.start.assert_called_once_with()
    assert (tmp_path / "unrelated").exists()
