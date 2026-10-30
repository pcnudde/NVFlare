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

import threading
from unittest.mock import Mock

import pytest

from nvflare.apis.fl_context import FLContext
from nvflare.app_common.streamers import log_streamer as subject
from tests.timing_utils import isolate_time


@pytest.fixture
def producer(tmp_path, monkeypatch):
    path = tmp_path / "log.txt"
    path.write_bytes(b"first")
    clock = isolate_time(monkeypatch, subject)
    monkeypatch.setattr(clock, "time", Mock(side_effect=range(100, 150)))
    value = subject._LogTailProducer(str(path), 64, 3, 0.25, threading.Event(), 5)
    try:
        yield value, path, clock
    finally:
        value.close()


def test_stop_drains_existing_data_then_performs_one_final_empty_retry(producer, monkeypatch):
    value, _, clock = producer
    value.stop_event.set()
    monkeypatch.setattr(clock, "sleep", Mock(side_effect=[None]))
    first, timeout = value.produce({}, FLContext())
    assert first[subject.KEY_DATA] == b"first" and timeout == 3
    final, timeout = value.produce({}, FLContext())
    assert final[subject.KEY_DATA] is None and final[subject.KEY_DATA_SIZE] == 0
    assert final[subject.KEY_EOF] is True and value.eof
    clock.sleep.assert_called_once_with(0.25)


def test_rotation_closes_stale_handle_and_reads_replacement_from_start(producer):
    value, path, _ = producer
    previous = value.file
    old = path.with_name("old.log")
    path.rename(old)
    path.write_bytes(b"replacement")
    result, _ = value.produce({}, FLContext())
    assert result[subject.KEY_DATA] == b"replacement"
    assert previous.closed and value.file is not previous
    path.unlink()
    assert value._check_rotation() is False


def test_reappearing_file_is_reopened_and_idle_stream_sends_heartbeat(producer, monkeypatch):
    value, path, clock = producer
    value.close()
    path.write_bytes(b"")
    value._last_send_time = 0
    result, timeout = value.produce({}, FLContext())
    assert value.file is not None
    assert result[subject.KEY_HEARTBEAT] is True and result[subject.KEY_DATA_SIZE] == 0
    assert result[subject.KEY_EOF] is False and timeout == 3
    assert value._last_was_data is False


def test_lost_offset_bookkeeping_does_not_discard_readable_log_bytes(producer, monkeypatch):
    value, _, _ = producer
    value.close()
    stream = Mock()
    stream.tell.side_effect = OSError("offset unavailable")
    stream.read.return_value = b"still readable"
    value.file = stream
    monkeypatch.setattr(value, "_check_rotation", lambda: False)
    result, _ = value.produce({}, FLContext())
    assert result[subject.KEY_DATA] == b"still readable"
    assert value._last_data_offset is None and value._last_was_data
