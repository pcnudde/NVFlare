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

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from nvflare.apis.task_launcher_spec import (
    TaskExecutionPhase,
    TaskExecutionStatus,
    TaskLauncherError,
    TaskLaunchError,
    TaskLaunchRequest,
)
from nvflare.app_common.task_launcher import process_launcher as subject


def request():
    return TaskLaunchRequest(
        job_id="job", site_name="site", task_id="task", attempt_id="attempt", argv=("python",), environment={}
    )


@pytest.mark.parametrize("failure", ["externally-reaped", "not-exited"])
def test_owned_leader_reap_never_releases_identity_without_confirmed_waitpid(monkeypatch, failure):
    fake_os = SimpleNamespace(
        P_PID=1, WEXITED=2, WNOHANG=4, WNOWAIT=8, waitid=Mock(return_value=None), waitpid=Mock(return_value=(0, 0))
    )
    if failure == "externally-reaped":
        fake_os.waitpid.side_effect = ChildProcessError("reaped")
    monkeypatch.setattr(subject, "os", fake_os)
    adapter = subject._OwnedProcessAdapter(SimpleNamespace(pid=123, process=Mock()))
    with pytest.raises(
        TaskLauncherError, match="externally reaped" if failure == "externally-reaped" else "did not exit"
    ):
        adapter.reap()
    assert not adapter.reaped
    if failure == "externally-reaped":
        with pytest.raises(TaskLauncherError, match="ownership is lost"):
            adapter.poll()
        assert fake_os.waitid.call_count == 1


@pytest.mark.parametrize("cleanup_fails", [False, True])
def test_post_acquisition_launch_failure_retains_owned_handle_and_fences_attempt(monkeypatch, cleanup_fails):
    adapter = SimpleNamespace(pid=123)
    acquired = subject.ProcessSpawnError(adapter)
    monkeypatch.setattr(subject, "spawn_process", Mock(side_effect=acquired))
    handle = Mock(execution_id="fake:123")
    if cleanup_fails:
        handle.cancel.side_effect = RuntimeError("cleanup")
    monkeypatch.setattr(subject, "ProcessTaskHandle", Mock(return_value=handle))
    launcher = subject.ProcessTaskLauncher()
    with pytest.raises(TaskLaunchError, match="unconfirmed" if cleanup_fails else "settled") as error:
        launcher.launch_task(request())
    assert error.value.handle is handle
    handle.cancel.assert_called_once()
    with pytest.raises(TaskLauncherError, match="already launched"):
        launcher.launch_task(request())
    assert subject.spawn_process.call_count == 1


def test_failure_before_acquisition_releases_attempt_identity_for_retry(monkeypatch):
    monkeypatch.setattr(subject, "spawn_process", Mock(side_effect=RuntimeError("before spawn")))
    launcher = subject.ProcessTaskLauncher()
    for _ in range(2):
        with pytest.raises(RuntimeError, match="before spawn"):
            launcher.launch_task(request())
    assert subject.spawn_process.call_count == 2
    assert launcher._attempt_identities == set()


def test_launcher_rejects_unsupported_platform_and_exhausted_identity_capacity(monkeypatch):
    launcher = subject.ProcessTaskLauncher(max_attempt_identities=1)
    launcher._attempt_identities.add(("different",))
    with pytest.raises(TaskLauncherError, match="capacity exhausted"):
        launcher.launch_task(request())
    monkeypatch.setattr(subject, "os", SimpleNamespace(name="nt"))
    with pytest.raises(TaskLauncherError, match="POSIX"):
        launcher.launch_task(request())


def make_handle():
    adapter = Mock(pid=123, reaped=False, owns_exited_leader=False)
    adapter.poll.return_value = None
    return (
        subject.ProcessTaskHandle(
            request(), adapter, stop_grace_period=1, descendant_settle_timeout=1, poll_interval=0.1
        ),
        adapter,
    )


def test_group_probe_conservatively_retains_unreadable_or_already_reaped_identity(monkeypatch):
    handle, adapter = make_handle()
    fake_os = SimpleNamespace(killpg=Mock(side_effect=PermissionError("probe")))
    monkeypatch.setattr(subject, "os", fake_os)
    assert handle._group_exists()
    fake_os.killpg.side_effect = None
    adapter.reaped = True
    assert handle._group_exists()


def test_dead_group_snapshot_uncertainty_and_remembered_member_permissions_prevent_settlement(monkeypatch):
    handle, _ = make_handle()
    fake_psutil = SimpleNamespace(
        pids=Mock(side_effect=PermissionError("snapshot")),
        AccessDenied=subject.psutil.AccessDenied,
        NoSuchProcess=subject.psutil.NoSuchProcess,
    )
    monkeypatch.setattr(subject, "psutil", fake_psutil)
    assert handle._dead_group_snapshot() is None
    remembered = Mock()
    remembered.is_running.side_effect = fake_psutil.AccessDenied(123)
    handle._group_members[123] = remembered
    assert handle._group_exists()


def test_termination_lock_deadline_failure_does_not_signal_any_process():
    handle, _ = make_handle()
    handle._termination_lock = Mock()
    handle._termination_lock.acquire.return_value = False
    handle._signal_group = Mock()
    with pytest.raises(TimeoutError, match="deadline"):
        handle._terminate_group(deadline=0)
    handle._signal_group.assert_not_called()
    handle._termination_lock.release.assert_not_called()


def test_cached_settlement_is_final_and_cancel_has_no_side_effects():
    handle, adapter = make_handle()
    settled = TaskExecutionStatus(phase=TaskExecutionPhase.TERMINAL, exit_code=0, settled=True)
    handle._settled_status = settled
    handle._signal_group = Mock()
    assert handle.poll() is settled
    assert handle.cancel() is settled
    adapter.poll.assert_not_called()
    handle._signal_group.assert_not_called()
    assert handle.process_group_id == 123
    assert handle.execution_id == "process:123"
    assert handle.request is not None
