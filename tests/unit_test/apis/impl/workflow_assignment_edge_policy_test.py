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

from contextlib import nullcontext
from unittest.mock import Mock

import pytest

from nvflare.apis.client import Client
from nvflare.apis.controller_spec import ClientTask, Task, TaskCompletionStatus
from nvflare.apis.fl_context import FLContext
from nvflare.apis.impl import wf_comm_server as subject
from nvflare.apis.shareable import Shareable


@pytest.fixture
def runtime(monkeypatch):
    server, ctx, engine = subject.WFCommServer(), FLContext(), Mock()
    engine.all_clients = [Client("one", "token")]
    engine.get_clients.return_value = engine.all_clients
    engine.new_context.side_effect = lambda: nullcontext(ctx)
    server._engine = engine
    monkeypatch.setattr(subject, "delete_msg_root", Mock())
    for name in ("log_info", "log_error", "log_warning", "system_panic"):
        monkeypatch.setattr(server, name, Mock())
    return server, ctx


@pytest.mark.parametrize("targets, expected", [("*", []), ("one", ["one"]), (None, ["one"])])
def test_assignment_target_normalization_and_one_use_task_identity(runtime, targets, expected):
    server, ctx = runtime
    task = Task("train", Shareable())
    server._schedule_task(task, ctx, None, targets)
    assert task.targets == expected and task.is_standing and server._tasks == [task]
    with pytest.raises(ValueError, match="already used"):
        server._schedule_task(task, ctx, None, targets)
    assert server._tasks == [task]


@pytest.mark.parametrize("targets", [42, [object()]])
def test_invalid_assignment_target_shape_fails_without_scheduling(runtime, targets):
    server, ctx = runtime
    with pytest.raises(ValueError, match="targets|element"):
        server._schedule_task(Task("train", Shareable()), ctx, None, targets)
    assert server._tasks == []


def test_upper_layer_failure_cancels_known_assignment_and_ignores_stale_identity(runtime):
    server, ctx = runtime
    task = Task("train", Shareable())
    client_task = ClientTask(Client("one", "token"), task)
    server._client_task_map[client_task.id] = client_task
    server.handle_exception("unknown", ctx)
    assert task.completion_status is None
    server.handle_exception(client_task.id, ctx)
    assert task.completion_status is TaskCompletionStatus.CANCELLED
    server.log_error.assert_called_once()


def test_task_monitor_rejects_invalid_manager_without_removing_task(runtime):
    server, ctx = runtime
    task = Task("train", Shareable())
    server._schedule_task(task, ctx, object(), ["one"])
    with pytest.raises(TypeError, match="TaskManager"):
        server.check_tasks()
    assert server._tasks == [task] and task.completion_status is None


def test_missing_engine_and_malformed_task_requests_fail_before_starting_monitor(runtime):
    server, ctx = runtime
    server.initialize_run(ctx)
    server.system_panic.assert_called_once()
    assert not server._task_monitor.is_alive()
    with pytest.raises(TypeError, match="client must"):
        server.process_task_request(object(), ctx)
    with pytest.raises(TypeError, match="fl_ctx must"):
        server.process_task_request(Client("one", "token"), object())
