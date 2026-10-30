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

import threading
from unittest.mock import Mock

import pytest

from nvflare.apis.fl_constant import FLContextKey, ReservedKey, ReturnCode, TaskResultReceipt
from nvflare.apis.fl_context import FLContext
from nvflare.apis.fl_exception import UnsafeJobError
from nvflare.apis.shareable import ReservedHeaderKey, Shareable
from nvflare.apis.signal import Signal
from nvflare.private.fed.client import client_runner as module
from nvflare.private.fed.client.client_engine_executor_spec import TaskAssignment
from nvflare.private.privacy_manager import Scope
from nvflare.widgets.info_collector import InfoCollector


@pytest.fixture
def runner(monkeypatch):
    subject = module.ClientRunner.__new__(module.ClientRunner)
    subject.task_data_filters = subject.task_result_filters = {}
    subject.run_abort_signal = Signal()
    subject.task_lock = threading.Lock()
    subject.running_tasks = {}
    subject.job_id = "job"
    for method in ["log_debug", "log_info", "log_error", "log_exception", "fire_event", "fire_event_with_data"]:
        setattr(subject, method, Mock())
    subject._reply_and_audit = Mock(side_effect=lambda reply, **kwargs: reply)
    monkeypatch.setattr(module, "add_job_audit_event", Mock())
    return subject


@pytest.mark.parametrize(
    "mode, expected",
    [
        ("input_unsafe", ReturnCode.UNSAFE_JOB),
        ("input_error", ReturnCode.TASK_DATA_FILTER_ERROR),
        ("input_type", ReturnCode.TASK_DATA_FILTER_ERROR),
        ("executor_unsafe", ReturnCode.UNSAFE_JOB),
        ("executor_type", ReturnCode.EXECUTION_RESULT_ERROR),
        ("executor_abort", ReturnCode.TASK_ABORTED),
        ("result_error", ReturnCode.TASK_RESULT_FILTER_ERROR),
        ("result_type", ReturnCode.TASK_RESULT_FILTER_ERROR),
        ("result_unsafe", ReturnCode.UNSAFE_JOB),
    ],
)
def test_task_execution_faults_keep_protocol_return_codes_and_mark_unsafe_jobs(monkeypatch, runner, mode, expected):
    context, abort = FLContext(), Signal()
    task = TaskAssignment("train", "task", Shareable())
    executor = Mock()
    executor.unsafe = False

    def execute(*args):
        if mode == "executor_unsafe":
            raise UnsafeJobError("untrusted executor")
        if mode == "executor_abort":
            abort.trigger(True)
        return [] if mode == "executor_type" else Shareable({"result": 1})

    executor.execute.side_effect = execute

    def filtering(name, data, *args, **kwargs):
        phase = "input" if name == Scope.TASK_DATA_FILTERS_NAME else "result"
        if mode == phase + "_unsafe":
            raise UnsafeJobError("untrusted filter")
        if mode == phase + "_error":
            raise ValueError("filter failed")
        return [] if mode == phase + "_type" else data

    monkeypatch.setattr(module, "apply_filters", filtering)
    reply = runner._execute_task(task, context, abort, executor, "audit")
    assert reply.get_return_code() == expected
    assert executor.execute.call_count == int(not mode.startswith("input"))
    unsafe = mode.endswith("unsafe")
    assert context.is_job_unsafe() is unsafe and executor.unsafe is unsafe
    assert runner.run_abort_signal.triggered is (mode == "input_unsafe")
    if not mode.startswith("input"):
        assert context.get_prop(FLContextKey.TASK_DATA) is None


def test_child_result_receipt_rejects_malformed_attempt_and_header_cookie_disagreement(runner):
    request = Shareable()
    request.set_header(ReservedHeaderKey.TASK_ID, "task")
    request.set_header(ReservedHeaderKey.TASK_ATTEMPT_ID, "attempt")
    request.add_cookie(ReservedHeaderKey.WORKFLOW, "workflow")
    task = TaskAssignment("train", "task", request)
    task.child_result_receipts = {"child": False}
    runner.running_tasks["task"] = task
    context, peer = FLContext(), FLContext()
    peer.set_prop(ReservedKey.RUN_NUM, "job")
    peer.set_prop(ReservedKey.IDENTITY_NAME, "child")
    context.set_peer_context(peer)
    request.set_header(ReservedHeaderKey.TASK_ATTEMPT_ID, 7)
    assert runner._handle_task_check("check", request, context).get_return_code() == ReturnCode.BAD_REQUEST_DATA
    assert runner._child_result_receipt(request, context) == TaskResultReceipt.TASK_CLOSED
    assert runner.claim_child_result(request, context) == (False, TaskResultReceipt.TASK_CLOSED)
    request.set_header(ReservedHeaderKey.TASK_ATTEMPT_ID, "attempt")
    request.set_header(ReservedHeaderKey.TASK_NAME, "tampered")
    request.add_cookie(ReservedHeaderKey.TASK_ID, "task")
    request.add_cookie(ReservedHeaderKey.TASK_NAME, "train")
    assert runner._child_result_receipt(request, context) == TaskResultReceipt.TASK_CLOSED


def test_statistics_event_rejects_noncollector_without_mutating_tasks(runner):
    context = FLContext()
    context.set_prop(InfoCollector.CTX_KEY_STATS_COLLECTOR, object())
    with pytest.raises(TypeError, match="GroupInfoCollector"):
        runner.handle_event(InfoCollector.EVENT_TYPE_GET_STATS, context)
    assert runner.running_tasks == {}
