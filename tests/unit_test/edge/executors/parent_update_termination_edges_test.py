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

from nvflare.apis.fl_constant import ReservedKey
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import ReturnCode, Shareable, make_reply
from nvflare.apis.signal import Signal
from nvflare.edge.constants import EdgeTaskHeaderKey
from nvflare.edge.executors import hug as module
from nvflare.edge.updater import Updater


@pytest.fixture
def runtime(monkeypatch):
    executor = module.HierarchicalUpdateGatherer("", "updater", 1)
    data = Shareable()
    for key, value in [
        (ReservedKey.TASK_ID, "task"),
        (ReservedKey.TASK_NAME, "train"),
        (EdgeTaskHeaderKey.TASK_SEQ, 2),
    ]:
        data.set_header(key, value)
    executor._pending_task = module.TaskInfo(data)
    executor._task_start_time = 9
    executor._parent_name = "parent"
    executor._updater = Mock(spec=Updater)
    report = Shareable({"update": 3})
    monkeypatch.setattr(executor, "_make_update_report", Mock(return_value=report))
    for name in ("log_info", "log_error", "log_debug", "log_exception"):
        monkeypatch.setattr(executor, name, Mock())
    monkeypatch.setattr(module, "time", SimpleNamespace(time=lambda: 10))
    ctx = FLContext()
    ctx.set_prop(ReservedKey.ENGINE, Mock())
    return executor, ctx, report


@pytest.mark.parametrize("kind", ["parent-abort", "stale-sequence", "children-complete", "timeout-after-task"])
def test_parent_authority_or_completed_children_end_round_without_sleep_or_extra_send(runtime, kind):
    executor, ctx, report = runtime
    reply = make_reply(ReturnCode.TASK_ABORTED if kind == "parent-abort" else ReturnCode.OK)
    if kind == "stale-sequence":
        reply.set_header(EdgeTaskHeaderKey.TASK_SEQ, 3)
    elif kind == "children-complete":
        executor._num_children = 1
        executor._pending_clients["site"] = 10
    elif kind == "timeout-after-task":
        reply = make_reply(ReturnCode.TIMEOUT)

    def send(**kwargs):
        if kind == "timeout-after-task":
            executor._task_done = True
        return {"parent": reply}

    ctx.get_engine().send_aux_request.side_effect = send
    result = executor._do_task(ctx, Signal())
    assert result is report and not executor._process_error
    ctx.get_engine().send_aux_request.assert_called_once()
    executor._updater.process_parent_update_reply.assert_not_called()


def test_current_task_state_failure_preserves_existing_task_and_logs_error(runtime):
    executor, ctx, _ = runtime
    original = executor._pending_task
    executor._updater.get_current_state.side_effect = OSError("state unavailable")
    assert executor.get_current_task(ctx) is original
    executor.log_exception.assert_called_once()
    assert original.id == "task" and original.seq == 2
