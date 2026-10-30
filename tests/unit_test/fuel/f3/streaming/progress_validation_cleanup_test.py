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

import pytest

from nvflare.fuel.f3.streaming import transfer_progress as module


@pytest.fixture
def tracker():
    return module.TransferProgressTracker(idle_timeout=10, clock=lambda: 100)


@pytest.fixture
def identity():
    return dict(job_id="job", task_id="task", transfer_id="transfer", direction=module.DIRECTION_RESULT_UPLOAD)


@pytest.mark.parametrize(
    "field,value,error",
    [
        ("sequence", "1", TypeError),
        ("sequence", -1, ValueError),
        ("bytes_done", "1", TypeError),
        ("bytes_done", -1, ValueError),
        ("items_done", "1", TypeError),
        ("items_done", -1, ValueError),
        ("state", "invented", ValueError),
    ],
)
def test_invalid_progress_update_does_not_create_record(tracker, identity, field, value, error):
    update = dict(identity, sequence=0, bytes_done=0)
    update[field] = value
    with pytest.raises(error):
        tracker.update(**update)
    assert list(tracker.records()) == []


def test_transfer_terminal_sequence_and_kind_are_monotonic_and_terminal_record_is_not_stalled(tracker, identity):
    assert not tracker.is_stalled(**identity)
    first = tracker.update(**identity, sequence=3, bytes_done=5, items_done=2)
    update = tracker.update(**identity, sequence=4, bytes_done=6, items_done=3, transfer_id_kind="stream")
    assert update.record is first.record and update.record.items_done == 3
    assert update.record.transfer_id_kind == "stream"
    result = tracker.mark_terminal(**identity, state=module.TransferProgressState.COMPLETED)
    assert result.accepted and result.record.sequence == 5 and result.record.bytes_done == 6
    assert result.record.terminal and not tracker.is_stalled(**identity, now=1000)
    assert tracker.prune() == 1
    assert not tracker.remove(**identity)


def test_remove_and_clear_are_idempotent_and_active_state_cannot_be_terminal(tracker, identity):
    tracker.update(**identity, sequence=0, bytes_done=0)
    with pytest.raises(ValueError, match="terminal state"):
        tracker.mark_terminal(**identity, state=module.TransferProgressState.ACTIVE)
    assert tracker.remove(**identity) and not tracker.remove(**identity)
    tracker.update(**identity, sequence=0, bytes_done=0)
    tracker.clear()
    tracker.clear()
    assert list(tracker.records()) == []
