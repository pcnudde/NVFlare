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

from nvflare.apis.fl_constant import FLContextKey, ReservedKey
from nvflare.apis.fl_context import FLContext
from nvflare.apis.job_def import JobMetaKey
from nvflare.edge.constants import EdgeApiStatus, EdgeContextKey
from nvflare.edge.executors.ete import EdgeTaskExecutor
from nvflare.edge.web.models.result_response import ResultResponse
from nvflare.edge.web.models.selection_response import SelectionResponse
from nvflare.edge.web.models.task_response import TaskResponse


class RecordingExecutor(EdgeTaskExecutor):
    def __init__(self):
        super().__init__(updater_id="updater", update_timeout=1)
        self.received = []

    def process_edge_task_request(self, request, current_task, fl_ctx):
        self.received.append((request, current_task, fl_ctx))
        return TaskResponse(EdgeApiStatus.OK)

    def process_edge_selection_request(self, request, current_task, fl_ctx):
        self.received.append((request, current_task, fl_ctx))
        return SelectionResponse(EdgeApiStatus.OK)

    def process_edge_result_report(self, request, current_task, fl_ctx):
        self.received.append((request, current_task, fl_ctx))
        return ResultResponse(EdgeApiStatus.OK)


@pytest.mark.parametrize(
    "method,kind",
    [
        ("process_edge_task_request", TaskResponse),
        ("process_edge_selection_request", SelectionResponse),
        ("process_edge_result_report", ResultResponse),
    ],
)
@pytest.mark.parametrize("has_task", [False, True])
def test_edge_dispatch_returns_idle_response_or_forwards_request_to_selected_task(method, kind, has_task, monkeypatch):
    executor = RecordingExecutor()
    executor.log_debug = Mock()
    current = SimpleNamespace(id="task") if has_task else None
    monkeypatch.setattr(executor, "get_current_task", lambda ctx: current)
    ctx = FLContext()
    payload = {"device": "device-1"}
    ctx.set_prop(EdgeContextKey.REQUEST_FROM_EDGE, payload)
    idle = kind(EdgeApiStatus.RETRY)
    executor._handle_edge_request("event", ctx, getattr(executor, method), idle)
    result = ctx.get_prop(EdgeContextKey.REPLY_TO_EDGE)
    assert isinstance(result, kind)
    if has_task:
        assert result.status == EdgeApiStatus.OK
        assert executor.received == [(payload, current, ctx)]
    else:
        assert result is idle and not executor.received
    assert EdgeContextKey.REPLY_TO_EDGE not in ctx.get_all_public_props()


def test_embedded_job_reply_includes_identity_and_omits_device_configuration():
    executor = RecordingExecutor()
    executor.log_debug = Mock()
    ctx = FLContext()
    ctx.set_prop(FLContextKey.JOB_META, {JobMetaKey.JOB_NAME: "edge-training"})
    ctx.set_prop(ReservedKey.RUN_NUM, "job-1")
    executor._handle_edge_job_request("job", ctx)
    reply = ctx.get_prop(EdgeContextKey.REPLY_TO_EDGE)
    assert reply.status == EdgeApiStatus.OK and reply.job_id == "job-1" and reply.job_name == "edge-training"
    assert reply.job_data == {} and not executor.received
