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

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from nvflare.apis.task_launcher_spec import TaskExecutionPhase, TaskExecutionStatus, TaskLaunchRequest
from nvflare.app_opt.task_launcher import k8s_task_service as module


@pytest.mark.parametrize("launch_error", [ValueError("bad request"), RuntimeError("admission failed")])
def test_rejected_task_launch_settles_without_a_pod_and_retains_failure_diagnostic(launch_error):
    backend = SimpleNamespace(launch_task=Mock(), poll_interval=0)
    job = module._Job(backend, "job", "site", "token", "origin")
    item = module._Attempt(TaskLaunchRequest("job", "site", "task", "attempt", ("worker",)))
    job.attempts[("task", "attempt")] = item

    def reject_and_release(request):
        job.release()
        raise launch_error

    backend.launch_task.side_effect = reject_and_release
    job.on_done = Mock()
    job.wake = Mock()
    job.run()
    assert item.status.settled and item.status.phase == TaskExecutionPhase.TERMINAL
    assert type(launch_error).__name__ in item.status.failure_reason
    assert str(launch_error) in item.status.failure_reason
    assert item.handle is None
    backend.launch_task.assert_called_once_with(item.request)
    job.on_done.assert_called_once_with()
    job.wake.wait.assert_not_called()


def test_owner_contains_observation_failure_then_retries_until_resources_are_confirmed_settled(monkeypatch):
    handle = Mock(execution_id="pod:uid")
    handle.poll.side_effect = [OSError("outage"), TaskExecutionStatus(TaskExecutionPhase.TERMINAL, settled=True)]
    job = module._Job(SimpleNamespace(poll_interval=1), "job", "site", "token", "origin")
    item = module._Attempt(TaskLaunchRequest("job", "site", "task", "attempt", ("worker",)), handle=handle)
    job.attempts[("task", "attempt")] = item
    job.release()
    job.on_done = Mock()
    calls = []

    def bounded_wait(timeout):
        calls.append(timeout)
        assert len(calls) == 1, "owner failed to settle after its second observation"

    job.wake = SimpleNamespace(wait=bounded_wait, clear=Mock())
    logger = Mock()
    monkeypatch.setattr(module, "logger", logger)
    job.run()
    assert item.status.settled and item.execution_id == "pod:uid"
    assert handle.poll.call_count == 2 and calls == [1]
    logger.warning.assert_called_once()
    job.on_done.assert_called_once_with()
