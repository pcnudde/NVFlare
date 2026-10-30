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
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from nvflare.apis.client import Client
from nvflare.apis.controller_spec import ClientTask, Task, TaskCompletionStatus
from nvflare.apis.fl_context import FLContext
from nvflare.apis.impl import wf_comm_server as subject
from nvflare.apis.shareable import Shareable
from nvflare.widgets.info_collector import GroupInfoCollector, InfoCollector


@pytest.fixture
def runtime(monkeypatch):
    server, ctx = subject.WFCommServer(), FLContext()
    for name in ("log_exception", "log_error", "log_info", "log_debug"):
        monkeypatch.setattr(server, name, Mock())
    server._engine = Mock()
    server._engine.new_context.side_effect = lambda: nullcontext(ctx)
    return server, ctx


@pytest.mark.parametrize("stage", ["before", "after", "copy"])
def test_failed_send_callback_never_assigns_task_and_always_runs_cleanup_callback(runtime, stage):
    server, ctx = runtime
    before, after = Mock(), Mock()
    error = RuntimeError("send preparation failed")
    if stage == "before":
        before.side_effect = error
    elif stage == "after":
        after.side_effect = error
    data = Shareable()
    if stage == "copy":

        class Uncopyable:
            def __deepcopy__(self, memo):
                raise error

        data["invalid"] = Uncopyable()
    task = Task("train", data, before_task_sent_cb=before, after_task_sent_cb=after)
    manager = Mock(spec=subject.BcastTaskManager)
    manager.check_task_send.return_value = subject.TaskCheckStatus.SEND
    task.props[subject._TASK_KEY_MANAGER] = manager
    server._tasks = [task]
    assert server.process_task_request(Client("a", "token"), ctx) == ("", "", None)
    before.assert_called_once()
    after.assert_called_once()
    assert task.completion_status is TaskCompletionStatus.ERROR and task.exception is error
    assert server._client_task_map == {} and task.client_tasks == []


def test_task_exit_survives_download_cleanup_and_done_callback_failure(runtime, monkeypatch):
    server, ctx = runtime
    error = RuntimeError("completion callback failed")
    done = Mock(side_effect=error)
    task = Task("train", Shareable(), task_done_cb=done)
    task.completion_status = TaskCompletionStatus.OK
    task.msg_root_id = "root"
    task._broadcast_data = Shareable({"cached": True})
    client_task = ClientTask(Client("a", "token"), task)
    task.client_tasks = [client_task]
    server._client_task_map[client_task.id] = client_task
    server._tasks = [task]
    cleanup = Mock(side_effect=RuntimeError("download cleanup failed"))
    monkeypatch.setattr(subject, "delete_msg_root", cleanup)
    server.check_tasks()
    cleanup.assert_called_once_with("root")
    done.assert_called_once_with(task=task, fl_ctx=ctx)
    assert task.exception is error and task.completion_status is TaskCompletionStatus.ERROR
    assert not task.is_standing and not hasattr(task, "_broadcast_data")
    assert server._tasks == [] and server._client_task_map == {}
    assert client_task.id in server._completed_client_task_map
    assert server.log_exception.call_count == 2


def test_statistics_show_assigned_clients_and_validate_collector_type(runtime):
    server, ctx = runtime
    task = Task("train", Shareable())
    task.client_tasks = [ClientTask(Client("a", "token"), task)]
    server._tasks = [task]
    server.controller = SimpleNamespace(name="workflow")
    collector = GroupInfoCollector()
    ctx.set_prop(InfoCollector.CTX_KEY_STATS_COLLECTOR, collector)
    server.handle_event(InfoCollector.EVENT_TYPE_GET_STATS, ctx)
    assert collector.info == {"workflow": {"tasks": {"train": ["a"]}}}
    ctx.set_prop(InfoCollector.CTX_KEY_STATS_COLLECTOR, object())
    with pytest.raises(TypeError, match="GroupInfoCollector"):
        server.handle_event(InfoCollector.EVENT_TYPE_GET_STATS, ctx)
