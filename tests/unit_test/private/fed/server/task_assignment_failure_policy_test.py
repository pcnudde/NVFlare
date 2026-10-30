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

from nvflare.apis.client import Client
from nvflare.apis.event_type import EventType
from nvflare.apis.fl_constant import FLContextKey, ReservedKey, ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import ReservedHeaderKey, Shareable
from nvflare.private.defs import SpecialTaskName, TaskConstant
from nvflare.private.fed.server import server_runner as subject
from nvflare.private.fed.server.server_engine import ServerEngine
from nvflare.widgets.info_collector import GroupInfoCollector, InfoCollector
from tests.timing_utils import isolate_time


@pytest.fixture
def env(monkeypatch):
    engine = Mock(spec=ServerEngine)
    engine.get_participating_clients.return_value = {}
    ctx = FLContext()
    ctx.set_prop(ReservedKey.ENGINE, engine)
    peer = FLContext()
    peer.set_prop(ReservedKey.RUN_NUM, "job")
    peer.set_prop(ReservedKey.IDENTITY_NAME, "site-a")
    ctx.set_peer_context(peer)
    workflow = SimpleNamespace(id="workflow", controller=Mock())
    config = subject.ServerRunnerConfig(5, 1, [workflow], {}, {}, handlers=[], components={})
    runner = subject.ServerRunner(config, "job", engine)
    runner.status = "started"
    runner.current_wf = workflow
    for name in ("log_debug", "log_info", "log_error", "log_warning", "log_exception", "fire_event"):
        setattr(runner, name, Mock())
    monkeypatch.setattr(subject, "add_job_audit_event", Mock(return_value="audit"))
    return SimpleNamespace(runner=runner, ctx=ctx, peer=peer, workflow=workflow, client=Client("site-a", "token"))


@pytest.mark.parametrize("state", ["init", "done", "no-peer", "wrong-job", "already-processing"])
def test_task_assignment_rejects_unready_or_invalid_client_state(env, state):
    if state in ("init", "done"):
        env.runner.status = state
    elif state == "no-peer":
        env.ctx.set_peer_context(None)
    elif state == "wrong-job":
        env.peer.set_prop(ReservedKey.RUN_NUM, "other")
    else:
        env.runner._processing_tasks[env.client.name] = "pending"
    name, task_id, data = env.runner.process_task_request(env.client, env.ctx)
    assert name == (SpecialTaskName.END_RUN if state in ("done", "wrong-job") else SpecialTaskName.TRY_AGAIN)
    assert task_id == ""
    if name == SpecialTaskName.TRY_AGAIN:
        assert data.get_header(TaskConstant.WAIT_TIME) == 1
    env.workflow.controller.communicator.process_task_request.assert_not_called()


@pytest.mark.parametrize("data", [None, {"bad": 1}, Shareable()])
def test_workflow_task_request_normalizes_empty_payload_but_rejects_non_shareable(env, data):
    env.workflow.controller.communicator.process_task_request.return_value = ("train", "task", data)
    name, task_id, result = env.runner._try_to_get_task(env.client, env.ctx)
    if data is not None and not isinstance(data, Shareable):
        assert (name, task_id, result) == ("", "", None)
    else:
        assert (name, task_id) == ("train", "task")
        assert isinstance(result, Shareable)
        assert result.get_header(ReservedHeaderKey.TASK_ID) == "task"
        assert result.get_cookie(ReservedHeaderKey.WORKFLOW) == "workflow"


def test_workflow_task_request_timeout_is_bounded_and_no_workflow_returns_immediately(env, monkeypatch):
    env.workflow.controller.communicator.process_task_request.return_value = (SpecialTaskName.TRY_AGAIN, "", None)
    clock = isolate_time(monkeypatch, subject)
    monkeypatch.setattr(clock, "time", Mock(side_effect=[0, 3]))
    assert env.runner._try_to_get_task(env.client, env.ctx, timeout=2) == ("", "", None)
    env.runner.current_wf = None
    monkeypatch.setattr(clock, "time", lambda: 4)
    assert env.runner._try_to_get_task(env.client, env.ctx) == ("", "", None)


@pytest.mark.parametrize("failure", ["filter", "workflow", "audit"])
def test_task_assignment_faults_clear_processing_marker_and_request_retry(env, monkeypatch, failure):
    data = Shareable()
    data.add_cookie(ReservedHeaderKey.WORKFLOW, "workflow")
    task = Mock(
        return_value=("train", "task", data),
        side_effect=RuntimeError("workflow failed") if failure == "workflow" else None,
    )
    monkeypatch.setattr(env.runner, "_try_to_get_task", task)
    monkeypatch.setattr(
        subject,
        "apply_filters",
        Mock(return_value=data, side_effect=ValueError("filter failed") if failure == "filter" else None),
    )
    if failure == "audit":
        subject.add_job_audit_event.side_effect = OSError("audit failed")
    name, task_id, reply = env.runner.process_task_request(env.client, env.ctx)
    assert name == SpecialTaskName.TRY_AGAIN and task_id == ""
    assert reply.get_header(TaskConstant.WAIT_TIME) == 1
    assert env.runner._processing_tasks == {}
    assert env.workflow.controller.communicator.handle_exception.call_count == int(failure == "filter")


@pytest.mark.parametrize("failure", ["not-shareable", "wrong-job", "wrong-identity", "bad-attempt"])
def test_result_submission_rejects_invalid_envelope_before_claiming_result(env, failure):
    result = Shareable()
    if failure == "not-shareable":
        result = {}
    elif failure == "wrong-job":
        env.peer.set_prop(ReservedKey.RUN_NUM, "other")
    elif failure == "wrong-identity":
        env.peer.set_prop(ReservedKey.IDENTITY_NAME, "other")
    else:
        result.set_header(ReservedHeaderKey.TASK_ATTEMPT_ID, False)
    env.runner._process_submission(env.client, "train", "task", result, env.ctx)
    env.workflow.controller.communicator.claim_submission.assert_not_called()
    env.workflow.controller.communicator.process_submission.assert_not_called()


def test_result_filter_failure_still_finishes_claimed_submission(env, monkeypatch):
    communicator = env.workflow.controller.communicator
    communicator.check_submission.return_value = True
    communicator.claim_submission.return_value = True
    monkeypatch.setattr(subject, "apply_filters", Mock(side_effect=ValueError("filter failed")))
    env.runner._process_submission(env.client, "train", "task", Shareable(), env.ctx)
    result = communicator.process_submission.call_args.kwargs["result"]
    assert result.get_return_code() == ReturnCode.TASK_RESULT_FILTER_ERROR
    communicator.finish_submission.assert_called_once_with("task", env.ctx)


def test_stats_fatal_events_abort_and_checkpoint_restore(env):
    collector = Mock(spec=GroupInfoCollector)
    env.ctx.set_prop(InfoCollector.CTX_KEY_STATS_COLLECTOR, collector)
    env.runner.handle_event(InfoCollector.EVENT_TYPE_GET_STATS, env.ctx)
    collector.set_info.assert_called_once_with(
        group_name="ServerRunner", info={"job_id": "job", "status": "started", "workflow": "workflow"}
    )
    env.ctx.set_prop(FLContextKey.EVENT_DATA, "fatal")
    env.runner.handle_event(EventType.FATAL_SYSTEM_ERROR, env.ctx)
    assert env.ctx.get_prop(FLContextKey.FATAL_SYSTEM_ERROR)
    assert env.runner.abort_signal.triggered and env.runner.status == "done"
    env.runner.restore({"job_id": "restored", "current_wf_index": "1"}, env.ctx)
    assert env.runner.get_persist_state(env.ctx) == {"job_id": "restored", "current_wf_index": 1}
