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

from nvflare.apis.task_launcher_spec import (
    TaskExecutionPhase,
    TaskExecutionStatus,
    TaskLauncherError,
    TaskLaunchRequest,
)
from nvflare.app_common.task_launcher import process_launcher as subject
from tests.timing_utils import isolate_time


@pytest.fixture
def handle():
    request = TaskLaunchRequest(job_id="job", site_name="site", task_id="task", attempt_id="attempt", argv=("python",))
    adapter = Mock(pid=123, reaped=False, owns_exited_leader=False)
    return subject.ProcessTaskHandle(
        request, adapter, stop_grace_period=1, descendant_settle_timeout=1, poll_interval=0.1
    )


def test_running_status_and_failed_reap_cannot_claim_settlement(handle, monkeypatch):
    monkeypatch.setattr(handle, "_group_exists", Mock(return_value=False))
    handle._adapter.poll.return_value = None
    running = handle.poll()
    assert running.phase == TaskExecutionPhase.RUNNING
    assert running.exit_code is None and running.termination_signal is None and not running.settled
    handle._adapter.poll.return_value = 0
    handle._adapter.reap.side_effect = TaskLauncherError("lost exclusive ownership")
    failed = handle.poll()
    assert failed.phase == TaskExecutionPhase.TERMINAL and not failed.settled
    assert failed.failure_reason == "lost exclusive ownership"
    assert handle._settled_status is None


@pytest.mark.parametrize("terminal", [False, True])
def test_wait_for_group_exit_honors_deadline_and_poll_budget(handle, monkeypatch, terminal):
    running = TaskExecutionStatus(TaskExecutionPhase.RUNNING)
    settled = TaskExecutionStatus(TaskExecutionPhase.TERMINAL, exit_code=0, settled=True)
    monkeypatch.setattr(handle, "poll", Mock(side_effect=[running, settled] if terminal else [running]))
    clock = isolate_time(monkeypatch, subject)
    clock.monotonic = Mock(side_effect=[0, 0.1] if terminal else [0, 2])
    clock.sleep = Mock(side_effect=[None])
    assert handle._wait_for_group_exit(1) is terminal
    assert clock.sleep.call_count == int(terminal)


@pytest.mark.parametrize("timeout", [None, 0.2])
def test_wait_for_settlement_retries_running_status_without_real_sleep(handle, monkeypatch, timeout):
    settled = TaskExecutionStatus(TaskExecutionPhase.TERMINAL, exit_code=0, settled=True)
    poll = Mock(side_effect=[TaskExecutionStatus(TaskExecutionPhase.RUNNING), settled])
    monkeypatch.setattr(handle, "poll", poll)
    clock = isolate_time(monkeypatch, subject)
    clock.monotonic = Mock(side_effect=[0, 0.1])
    clock.sleep = Mock(side_effect=[None])
    assert handle.wait_for_settlement(timeout) is settled
    poll.assert_called_with()
    assert poll.call_count == 2
    clock.sleep.assert_called_once_with(0.1)


def test_running_worker_timeout_does_not_cancel_or_signal_group(handle, monkeypatch):
    monkeypatch.setattr(handle, "poll", Mock(side_effect=[TaskExecutionStatus(TaskExecutionPhase.RUNNING)]))
    signal = Mock()
    monkeypatch.setattr(handle, "_signal_group", signal)
    clock = isolate_time(monkeypatch, subject)
    clock.monotonic = Mock(side_effect=[0, 1])
    with pytest.raises(TimeoutError, match="did not settle within"):
        handle.wait_for_settlement(1)
    signal.assert_not_called()
    assert not handle._cancel_requested


def test_exited_leader_with_live_descendants_is_faulted_before_scope_cleanup(handle, monkeypatch):
    status = TaskExecutionStatus(TaskExecutionPhase.TERMINAL, exit_code=0, settled=False)
    monkeypatch.setattr(handle, "poll", Mock(side_effect=[status]))
    monkeypatch.setattr(handle, "_status_unlocked", Mock(side_effect=[status]))
    wait = Mock(return_value=False)
    monkeypatch.setattr(handle, "_wait_for_group_exit", wait)
    cleanup = Mock(return_value=status)
    monkeypatch.setattr(handle, "_terminate_group", cleanup)
    assert handle.wait_for_settlement() is status
    assert handle._failure_reason == "task leader exited while descendants remained alive"
    wait.assert_called_once_with(1)
    cleanup.assert_called_once_with(deadline=None)


@pytest.mark.parametrize("lock_acquired", [False, True])
def test_termination_deadline_releases_owned_lock_and_sends_no_late_signal(handle, monkeypatch, lock_acquired):
    lock = Mock()
    lock.acquire.return_value = lock_acquired
    handle._termination_lock = lock
    monkeypatch.setattr(handle, "poll", Mock(side_effect=[TaskExecutionStatus(TaskExecutionPhase.RUNNING)]))
    signal = Mock()
    monkeypatch.setattr(handle, "_signal_group", signal)
    clock = isolate_time(monkeypatch, subject)
    clock.monotonic = Mock(side_effect=[0, 2])
    with pytest.raises(TimeoutError, match="deadline"):
        handle._terminate_group(deadline=1)
    assert lock.release.call_count == int(lock_acquired)
    signal.assert_not_called()
