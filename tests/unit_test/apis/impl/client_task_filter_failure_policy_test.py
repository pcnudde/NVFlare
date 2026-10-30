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
from nvflare.apis.controller_spec import Task, TaskCompletionStatus
from nvflare.apis.event_type import EventType
from nvflare.apis.fl_constant import FLContextKey, ReservedTopic, ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.impl import task_controller as subject
from nvflare.apis.shareable import ReservedHeaderKey, Shareable, make_reply


@pytest.fixture
def runtime(monkeypatch):
    controller, ctx, engine = subject.TaskController(), FLContext(), Mock()
    monkeypatch.setattr(ctx, "get_engine", lambda: engine)
    engine.all_clients = [Client("one", "token")]
    engine.validate_targets.return_value = (["one"], [])
    engine.send_aux_request.return_value = {"one": make_reply(ReturnCode.OK)}
    for name in ("log_debug", "log_exception", "fire_event", "fire_event_with_data"):
        monkeypatch.setattr(controller, name, Mock())
    filters = Mock(side_effect=lambda name, data, *args, **kwargs: data)
    cleanup = Mock()
    monkeypatch.setattr(subject, "apply_filters", filters)
    monkeypatch.setattr(subject, "delete_msg_root", cleanup)
    return controller, ctx, engine, filters, cleanup


def test_successful_broadcast_preserves_filtered_payload_ttl_and_callback_identity(runtime):
    controller, ctx, engine, filters, cleanup = runtime
    outgoing, incoming = Shareable({"filtered": "task"}), make_reply(ReturnCode.OK)
    incoming["value"] = 4
    outgoing.set_header(ReservedHeaderKey.MSG_ROOT_TTL, 90)
    filters.side_effect = [outgoing, incoming]
    result_cb, done_cb = Mock(), Mock()
    task = Task("train", Shareable(), timeout=5, result_received_cb=result_cb, task_done_cb=done_cb)
    assert controller.broadcast(task, ctx, ["one"]) == {"one": incoming}
    sent = engine.send_aux_request.call_args.kwargs
    assert sent["request"] is outgoing and sent["timeout"] == 5 and sent["topic"] == ReservedTopic.DO_TASK
    assert outgoing.get_header(ReservedHeaderKey.MSG_ROOT_TTL) == 90
    cleanup.assert_called_once_with(task.msg_root_id)
    client_task = result_cb.call_args.kwargs["client_task"]
    assert client_task.client is engine.all_clients[0] and client_task.result is incoming
    done_cb.assert_called_once_with(task=task, fl_ctx=ctx)


@pytest.mark.parametrize(
    "failure,expected",
    [
        ("data-filter", ReturnCode.TASK_DATA_FILTER_ERROR),
        ("result-filter", ReturnCode.TASK_RESULT_FILTER_ERROR),
        ("result-callback", ReturnCode.ERROR),
        ("done-callback", ReturnCode.ERROR),
        ("remote", ReturnCode.ERROR),
    ],
)
def test_filter_callback_and_remote_failures_return_explicit_error_replies(runtime, failure, expected):
    controller, ctx, engine, filters, _ = runtime
    error = RuntimeError("policy failure")
    result_cb, done_cb = Mock(), Mock()
    task = Task("train", Shareable(), timeout=5, result_received_cb=result_cb, task_done_cb=done_cb)
    if failure == "data-filter":
        filters.side_effect = error
    elif failure == "result-filter":
        filters.side_effect = [task.data, error]
    elif failure == "result-callback":
        result_cb.side_effect = error
    elif failure == "done-callback":
        done_cb.side_effect = error
    else:
        engine.send_aux_request.return_value = {"one": make_reply(ReturnCode.EXECUTION_EXCEPTION)}
    replies = controller.broadcast_and_wait(task, ctx, ["one"])
    assert replies["one"].get_return_code() == expected
    if failure == "data-filter":
        engine.send_aux_request.assert_not_called()
    if failure in ("result-callback", "done-callback"):
        assert task.completion_status is TaskCompletionStatus.ERROR and task.exception is error


@pytest.mark.parametrize(
    "targets,message", [([], "Must provide"), (["one", "two"], "single target"), (["missing"], "invalid target")]
)
def test_send_rejects_invalid_target_before_transport(runtime, targets, message):
    controller, ctx, engine, _, _ = runtime
    if targets == ["missing"]:
        engine.validate_targets.return_value = ([], ["missing"])
    with pytest.raises(ValueError, match=message):
        controller.send(Task("train", Shareable(), timeout=5), ctx, targets)
    engine.send_aux_request.assert_not_called()


def test_start_run_uses_runner_filters_and_empty_values_fall_back_to_dict(runtime):
    controller, ctx, _, _, _ = runtime
    ctx.set_prop(FLContextKey.RUNNER, SimpleNamespace(task_data_filters={"train": "out"}, task_result_filters=None))
    controller.handle_event(EventType.START_RUN, ctx)
    assert controller.task_data_filters == {"train": "out"} and controller.task_result_filters == {}
