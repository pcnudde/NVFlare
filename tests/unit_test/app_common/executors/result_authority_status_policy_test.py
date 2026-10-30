# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

from unittest.mock import Mock

import pytest

from nvflare.apis.shareable import Shareable
from nvflare.app_common.executors.client_api import cell_backend
from nvflare.app_common.executors.client_api.attach_backend import AttachBackend, _AttachedTrainerSession
from nvflare.client.cell.defs import MsgKey, ResultState, Topic
from nvflare.fuel.f3.cellnet.defs import MessageHeaderKey, ReturnCode
from nvflare.fuel.f3.message import Message


@pytest.fixture
def backend():
    value = AttachBackend()
    value._session = _AttachedTrainerSession("site.trainer")
    value._session.session_id = "session"
    value._session.ready.set()
    value._cell = Mock()
    return value


def request(**fields):
    return Message({MessageHeaderKey.ORIGIN: "site.trainer"}, {MsgKey.SESSION_ID: "session", **fields})


@pytest.mark.parametrize(
    "state", ["accepted", "unknown", "other-result", "stale-session", "wrong-origin", "invalid-payload"]
)
def test_result_status_reports_only_the_canonical_result_to_the_authenticated_session(backend, state):
    backend._result_authority["task"] = ("canonical", "attempt")
    message = request(**{MsgKey.TASK_ID: "task", MsgKey.RESULT_ID: "canonical"})
    if state == "unknown":
        message.payload[MsgKey.TASK_ID] = "missing"
    elif state == "other-result":
        message.payload[MsgKey.RESULT_ID] = "other"
    elif state == "stale-session":
        message.payload[MsgKey.SESSION_ID] = "old"
    elif state == "wrong-origin":
        message.set_header(MessageHeaderKey.ORIGIN, "other.trainer")
    elif state == "invalid-payload":
        message.payload = []
    reply = backend._handle_result_status(message)
    if state == "accepted":
        assert reply.payload[MsgKey.RESULT_STATE] == ResultState.ACCEPTED
        assert reply.payload[MsgKey.ACCEPTED_ATTEMPT_ID] == "attempt"
    elif state in ("unknown", "other-result"):
        assert reply.payload[MsgKey.RESULT_STATE] == ResultState.UNKNOWN
    elif state == "invalid-payload":
        assert reply.get_header(MessageHeaderKey.RETURN_CODE) == ReturnCode.INVALID_REQUEST
    else:
        assert reply.payload[MsgKey.REPLY_TOPIC] == Topic.ERROR
    assert backend._result_authority == {"task": ("canonical", "attempt")}


@pytest.mark.parametrize("same_result", [False, True])
def test_replayed_ready_result_cannot_replace_canonical_attempt(backend, same_result):
    backend._result_authority["task"] = ("canonical", "first-attempt")
    reply = backend._handle_result_ready(
        request(
            **{
                MsgKey.TASK_ID: "task",
                MsgKey.RESULT_ID: "canonical" if same_result else "new",
                MsgKey.ATTEMPT_ID: "retry",
                MsgKey.RESULT: Shareable(),
            }
        )
    )
    assert reply.payload[MsgKey.REPLY_TOPIC] == (Topic.RESULT_ACCEPTED if same_result else Topic.RESULT_REJECTED)
    if same_result:
        assert reply.payload[MsgKey.ACCEPTED_ATTEMPT_ID] == "first-attempt"
    assert backend._current_task is None
    assert backend._result_authority == {"task": ("canonical", "first-attempt")}


def test_live_result_source_defers_authority_eviction_until_authenticated_settlement(backend, monkeypatch):
    monkeypatch.setattr(cell_backend, "_RESULT_AUTHORITY_LIMIT", 2)
    session = backend._session
    session.result_source_live.set()
    session.result_source_task_id = "task"
    for index in range(4):
        backend._remember_result_authority(str(index), "result", "attempt")
    assert len(backend._result_authority) == 4
    reply = backend._handle_result_source_settled(request(**{MsgKey.TASK_ID: "task"}))
    assert reply.payload[MsgKey.REPLY_TOPIC] == Topic.RESULT_SOURCE_SETTLED
    assert list(backend._result_authority) == ["2", "3"]
    assert not session.result_source_live.is_set() and session.result_source_task_id is None


@pytest.mark.parametrize("method", ["_handle_result_ready", "_handle_result_source_settled", "_handle_heartbeat"])
def test_protocol_callbacks_reject_nonmapping_payloads_without_mutating_session(backend, method):
    reply = getattr(backend, method)(Message({}, []))
    assert reply.get_header(MessageHeaderKey.RETURN_CODE) == ReturnCode.INVALID_REQUEST
    assert not backend._session.result_source_live.is_set()
    assert not backend._result_authority


def test_heartbeat_marks_result_source_live_and_closed_backend_rejects_new_activity(backend):
    reply = backend._handle_heartbeat(request(**{MsgKey.RESULT_SOURCE_LIVE: True}))
    assert reply.payload[MsgKey.REPLY_TOPIC] == Topic.HEARTBEAT
    assert backend._session.result_source_live.is_set()
    backend._closed = True
    assert backend._handle_heartbeat(request()).payload[MsgKey.REPLY_TOPIC] == Topic.ERROR


def test_abort_delivery_failure_is_logged_without_overwriting_shared_logger_method(backend, monkeypatch):
    backend._cell.fire_and_forget.side_effect = OSError("offline")
    error = Mock()
    monkeypatch.setattr(backend.logger, "error", error)
    backend._send_abort(backend._session, "job stopped")
    assert "offline" in error.call_args.args[0]
