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

from nvflare.apis.fl_constant import FLContextKey, ReservedKey, ServerCommandKey
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.fuel.f3.cellnet.defs import MessageHeaderKey, ReturnCode
from nvflare.fuel.f3.message import Message
from nvflare.private.fed.client import communicator as module


@pytest.fixture
def subject():
    value = module.Communicator({"client_name": "site"}, cell=Mock())
    value.logger = Mock()
    return value


@pytest.mark.parametrize("mode", ["no-engine", "no-task", "same-task", "not-ready", "unhandled-event"])
def test_child_get_task_retries_until_new_task_is_ready_and_warns_on_unhandled_delivery(subject, mode):
    context = FLContext()
    engine = Mock()
    engine.new_context.side_effect = lambda: nullcontext(context)
    subject.engine = None if mode == "no-engine" else engine
    subject._is_valid_child_identity = Mock(return_value=True)
    task = Shareable()
    task.set_header(FLContextKey.TASK_ID, "task")
    task.set_header(ReservedKey.TASK_IS_READY, mode == "unhandled-event")
    subject.pending_task = None if mode == "no-task" else task
    request = Shareable()
    request.set_header(ServerCommandKey.LAST_TASK_ID, "task" if mode == "same-task" else "older")
    response = subject._process_get_task(Message(payload=request))
    assert response.get_header(MessageHeaderKey.RETURN_CODE) == ReturnCode.OK
    if mode == "unhandled-event":
        assert response.payload.get_header(FLContextKey.TASK_ID) == "task"
        assert response.payload.get_header(ReservedKey.TASK_IS_READY) is None
        assert "not processed" in subject.logger.warning.call_args.args[0]
        engine.fire_event.assert_called_once()
    else:
        assert response.payload.get_header(ServerCommandKey.TASK_NAME) == module.SpecialTaskName.TRY_AGAIN
        assert response.payload.get_header(FLContextKey.TASK_ID) == ""
        engine.fire_event.assert_not_called()


@pytest.mark.parametrize("operation", ["get", "submit"])
def test_child_protocol_rejects_non_shareable_payload(subject, operation):
    response = (subject._process_get_task if operation == "get" else subject._process_submit_result)(
        Message(payload={})
    )
    assert response.get_header(MessageHeaderKey.RETURN_CODE) == ReturnCode.INVALID_REQUEST


def test_child_submit_rejects_malformed_attempt_and_acknowledges_closed_engine(subject):
    result = Shareable()
    result.set_header(module.ReservedHeaderKey.TASK_ATTEMPT_ID, 42)
    response = subject._process_submit_result(Message(payload=result))
    assert response.get_header(MessageHeaderKey.RETURN_CODE) == ReturnCode.INVALID_REQUEST
    result = Shareable()
    result.set_header(module.ReservedHeaderKey.TASK_ATTEMPT_ID, "attempt")
    response = subject._process_submit_result(Message(payload=result))
    assert response.get_header(MessageHeaderKey.RETURN_CODE) == ReturnCode.OK
    assert (
        response.payload.get_header(module.ReservedHeaderKey.TASK_RESULT_RECEIPT)
        == module.TaskResultReceipt.TASK_CLOSED
    )
    assert "no engine" in subject.logger.error.call_args.args[0]


def test_submit_update_session_mismatch_never_sends_and_ttl_propagates_on_valid_session(subject, monkeypatch):
    context = FLContext()
    context.set_prop(FLContextKey.SSID, "session")
    context.set_prop(ReservedKey.RUN_NUM, "job")
    data = Shareable()
    assert (
        subject.submit_update("project", "token", "other", context, "site", data, "train") == ReturnCode.INVALID_SESSION
    )
    subject.cell.send_request.assert_not_called()
    data.set_header(module.ReservedHeaderKey.MSG_ROOT_TTL, 5)

    def send(**kwargs):
        kwargs["request"].set_header(MessageHeaderKey.PAYLOAD_LEN, 0)
        return Message(headers={MessageHeaderKey.RETURN_CODE: ReturnCode.OK})

    subject.cell.send_request.side_effect = send
    monkeypatch.setattr(module, "determine_parent_fqcn", lambda config, ctx: "parent")
    assert subject.submit_update("project", "token", "session", context, "site", data, "train") == ReturnCode.OK
    sent = subject.cell.send_request.call_args.kwargs["request"]
    assert sent.get_header(MessageHeaderKey.MSG_ROOT_TTL) == 5
    assert subject.pending_task is None
