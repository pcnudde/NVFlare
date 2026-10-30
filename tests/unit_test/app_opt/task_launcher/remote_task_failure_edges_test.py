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

import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from nvflare.apis.task_launcher_spec import TaskLauncherError, TaskLaunchRequest, TaskSettlementError
from nvflare.app_opt.task_launcher import k8s_task_service as module
from nvflare.fuel.f3.cellnet.defs import MessageHeaderKey, ReturnCode
from nvflare.fuel.f3.cellnet.utils import new_cell_message


@pytest.fixture
def service():
    cell = Mock()
    owner = module.K8sTaskService.get_or_create(cell)
    assert module.K8sTaskService.get_or_create(cell) is owner
    cell.register_request_cb.assert_called_once()
    job = module._Job(SimpleNamespace(max_attempts=5), "job", "site", "token", "server.job")
    owner.jobs["job"] = job
    return owner, job


def request(action="poll", **changes):
    data = dict(job_id="job", token="token", action=action, task_id="task", attempt_id="attempt")
    data.update(changes)
    return new_cell_message({MessageHeaderKey.ORIGIN: "server.job"}, json.dumps(data))


@pytest.mark.parametrize("payload", [b"bytes", "[]", "{}", "x" * (module._MAX_MESSAGE + 1)])
def test_control_rejects_unbounded_or_untyped_wire_values_without_creating_attempts(service, payload):
    owner, job = service
    reply = owner._request(new_cell_message({}, payload))
    assert reply.get_header(MessageHeaderKey.RETURN_CODE) == ReturnCode.INVALID_REQUEST
    assert not job.attempts and not job.wake.is_set()


@pytest.mark.parametrize("changes", [{"task_id": ""}, {"attempt_id": "x" * 257}, {"action": "unknown"}])
def test_authenticated_invalid_request_does_not_modify_job_admission(service, changes):
    owner, job = service
    reply = owner._request(request(**changes))
    assert reply.get_header(MessageHeaderKey.RETURN_CODE) == ReturnCode.INVALID_REQUEST
    assert not job.attempts and not job.closed


def test_unknown_poll_and_closed_job_cancel_never_allocate_or_fence_an_attempt(service):
    owner, job = service
    assert owner._request(request()).get_header(MessageHeaderKey.RETURN_CODE) == ReturnCode.INVALID_REQUEST
    job.closed = True
    assert owner._request(request("cancel")).get_header(MessageHeaderKey.RETURN_CODE) == ReturnCode.INVALID_REQUEST
    assert not job.attempts and job.cancel_fences == 0


def test_created_cj_identity_requires_uid_and_wakes_owner_without_starting_worker(service):
    _, job = service
    with pytest.raises(ValueError, match="created CJ UID"):
        job.attach_cj("pod", "")
    assert job.cj_name is None and not job.wake.is_set()
    job.attach_cj("pod", "uid")
    assert (job.cj_name, job.cj_uid, job.next_cj_observation) == ("pod", "uid", 0)
    assert job.wake.is_set()


def test_client_requires_capability_and_rejects_closed_or_cross_job_launches(monkeypatch):
    monkeypatch.delenv(module.ENV_WORKSPACE_OWNER_FQCN, raising=False)
    monkeypatch.delenv(module.ENV_WORKSPACE_TRANSFER_TOKEN, raising=False)
    cell = Mock()
    with pytest.raises(RuntimeError, match="CP-provisioned"):
        module.K8sTaskClient(cell, "job", "site", 1, 0.01, 0)
    monkeypatch.setenv(module.ENV_WORKSPACE_OWNER_FQCN, "server")
    monkeypatch.setenv(module.ENV_WORKSPACE_TRANSFER_TOKEN, "token")
    client = module.K8sTaskClient(cell, "job", "site", 1, 0.01, 0)
    other = TaskLaunchRequest("other", "site", "task", "attempt", ("worker",))
    with pytest.raises(ValueError, match="different CJ"):
        client.launch(other)
    client.closed = True
    with pytest.raises(TaskLauncherError, match="proxy is closed"):
        client.launch(other)
    cell.send_request.assert_not_called()


def test_cancel_timeout_preserves_cp_cleanup_ownership_without_sleeping(monkeypatch):
    client = SimpleNamespace(call=Mock(side_effect=OSError("connection lost")), cleanup_timeout=0, poll_interval=1)
    item = module.K8sRemoteTaskHandle(client, TaskLaunchRequest("job", "site", "task", "attempt", ("worker",)))
    clock = SimpleNamespace(monotonic=lambda: 10, sleep=Mock(side_effect=AssertionError("must not sleep")))
    monkeypatch.setattr(module, "time", clock)
    with pytest.raises(TaskSettlementError, match="cleanup remains owned by CP") as error:
        item.cancel()
    assert isinstance(error.value.__cause__, TimeoutError)
    assert not error.value.status.settled
    client.call.assert_called_once_with("cancel", task_id="task", attempt_id="attempt")
    clock.sleep.assert_not_called()
