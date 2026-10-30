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
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from nvflare.client.cell import attach_session as subject
from nvflare.client.cell.bootstrap import BootstrapKey
from nvflare.client.cell.defs import MsgKey, ResultState, TaskState, Topic
from nvflare.fuel.f3.cellnet.defs import MessageHeaderKey, ReturnCode
from nvflare.fuel.f3.cellnet.utils import new_cell_message
from nvflare.fuel.f3.streaming.download_service import OBJ_DOWNLOADER_CHANNEL


def reply(topic, **body):
    return new_cell_message({MessageHeaderKey.RETURN_CODE: ReturnCode.OK}, {MsgKey.REPLY_TOPIC: topic, **body})


@pytest.fixture
def session():
    api = SimpleNamespace(
        _config={
            BootstrapKey.ATTACH_ID: "trainer",
            BootstrapKey.SITE_NAME: "site",
            BootstrapKey.CONNECT_URL: "grpc://127.0.0.1:9000",
            BootstrapKey.CONNECTION_SECURITY: "clear",
        },
        _lock=threading.RLock(),
        _cj_fqcn="site.job",
        _session_id="session",
        _protocol_secure=False,
        _reply=reply,
        _cell=Mock(),
        _validate_cj_control=Mock(return_value="bad origin"),
    )
    return subject.AttachTrainerSession(api)


@pytest.mark.parametrize(
    "task,attempt,seq,reason",
    [
        (None, "a", 1, "task_id"),
        ("t", None, 1, "attempt_id"),
        ("t", "a", True, "positive integer"),
        ("t", "a", 0, "positive integer"),
    ],
)
def test_task_reservation_rejects_invalid_identity_without_advancing_watermark(session, task, attempt, seq, reason):
    result = session.reserve_task(task, attempt, seq)
    assert result.payload[MsgKey.REPLY_TOPIC] == Topic.TASK_FAILED
    assert reason in result.payload[MsgKey.REASON]
    assert session._highest_task_sequence == 0 and not session._task_states


def test_reservation_retry_and_mismatched_sequence_do_not_requeue_or_overwrite_owner(session):
    assert session.reserve_task("task", "a", 1) is None
    session.forget_reserved_task("task", "a")
    assert session._retryable_task == ("task", 1)
    assert session.reserve_task("task", "b", 1) is None
    assert session._retryable_task is None
    mismatch = session.reserve_task("task", "c", 2)
    assert "task_seq mismatch" in mismatch.payload[MsgKey.REASON]
    assert session._task_attempts["task"] == "b"
    with pytest.raises(subject.TrainerSessionError, match="reservation changed"):
        session.commit_reserved_task_locked("task", "a")
    session.commit_reserved_task_locked("task", "b")
    assert session._task_states["task"] == TaskState.QUEUED


@pytest.mark.parametrize(
    "kind,match",
    [
        ("none", "no reply"),
        ("cell-error", "cell-level failure"),
        ("payload", "invalid RESULT_READY reply payload"),
        ("rejected", "rejected by the CJ"),
        ("topic", "invalid RESULT_READY reply topic"),
        ("result-id", "result id mismatch"),
        ("attempt-id", "no accepted attempt id"),
    ],
)
def test_result_acknowledgment_requires_authoritative_matching_result_and_attempt(kind, match):
    response = reply(Topic.RESULT_ACCEPTED, **{MsgKey.RESULT_ID: "result", MsgKey.ACCEPTED_ATTEMPT_ID: "attempt"})
    if kind == "none":
        response = None
    elif kind == "cell-error":
        response.set_header(MessageHeaderKey.RETURN_CODE, ReturnCode.TIMEOUT)
    elif kind == "payload":
        response.payload = "wrong"
    elif kind == "rejected":
        response.payload[MsgKey.REPLY_TOPIC] = Topic.RESULT_REJECTED
    elif kind == "topic":
        response.payload[MsgKey.REPLY_TOPIC] = Topic.TASK_STATUS
    elif kind == "result-id":
        response.payload[MsgKey.RESULT_ID] = "other"
    else:
        response.payload.pop(MsgKey.ACCEPTED_ATTEMPT_ID)
    with pytest.raises(RuntimeError, match=match):
        subject.AttachTrainerSession._accepted_result_attempt(response, "result")


@pytest.mark.parametrize("kind", ["transport", "bad-payload", "unknown", "rejected", "no-attempt"])
def test_result_status_recovers_only_a_valid_canonical_attempt(session, kind):
    response = reply(
        Topic.RESULT_STATUS, **{MsgKey.RESULT_STATE: ResultState.ACCEPTED, MsgKey.ACCEPTED_ATTEMPT_ID: "a"}
    )
    if kind == "transport":
        session._api._cell.send_request.side_effect = OSError("disconnected")
    elif kind == "bad-payload":
        response.payload = []
    elif kind == "unknown":
        response.payload[MsgKey.RESULT_STATE] = ResultState.UNKNOWN
    elif kind == "rejected":
        response.payload[MsgKey.RESULT_STATE] = ResultState.REJECTED
    else:
        response.payload.pop(MsgKey.ACCEPTED_ATTEMPT_ID)
    session._api._cell.send_request.return_value = response
    if kind in ("rejected", "no-attempt"):
        with pytest.raises(subject.TrainerSessionError, match="rejected|no accepted attempt"):
            session._accepted_result_attempt_from_status("task", "result")
    else:
        assert session._accepted_result_attempt_from_status("task", "result") is None
    request = session._api._cell.send_request.call_args.kwargs["request"]
    assert request.payload[MsgKey.SESSION_ID] == "session"
    assert request.payload[MsgKey.TASK_ID] == "task"


def test_session_security_rejects_unresolved_transport_and_unauthorized_downloads(session):
    session.connection_security = None
    with pytest.raises(RuntimeError, match="not resolved"):
        session.cell_security()
    with pytest.raises(RuntimeError, match="not resolved"):
        session.connection_resources()
    session.connection_security = "mtls"
    with pytest.raises(RuntimeError, match="requires 'ca_cert'"):
        session.cell_security()
    with pytest.raises(RuntimeError, match="message interceptor"):
        session.install_pre_decode_guard(SimpleNamespace(core_cell=None))
    message = new_cell_message({MessageHeaderKey.CHANNEL: "unrelated"}, {})
    assert session._pre_decode_guard(message) is None
    message.set_header(MessageHeaderKey.CHANNEL, OBJ_DOWNLOADER_CHANNEL)
    message.set_header(MessageHeaderKey.ORIGIN, "site.job")
    assert session._pre_decode_guard(message) is None
    message.set_header(MessageHeaderKey.ORIGIN, "other.job")
    assert (
        session._pre_decode_guard(message).get_header(MessageHeaderKey.RETURN_CODE) == ReturnCode.AUTHENTICATION_ERROR
    )
    malformed = session._handle_task_status(new_cell_message({}, []))
    assert malformed.get_header(MessageHeaderKey.RETURN_CODE) == ReturnCode.INVALID_REQUEST
    rejected = session._handle_task_status(new_cell_message({}, {}))
    assert rejected.payload[MsgKey.REPLY_TOPIC] == Topic.ERROR
    assert rejected.payload[MsgKey.REASON] == "bad origin"


def test_open_wait_uses_expired_budget_and_cancellation_reports_interrupted_session(session):
    session._wait_deadline = 0
    with pytest.raises(subject.TrainerSessionError, match="no SESSION_OPEN"):
        session.wait_for_open()
    session._api._session_id = None
    session.close()
    with pytest.raises(subject.TrainerSessionError, match="interrupted"):
        session.wait_for_open()
    with pytest.raises(subject.TrainerSessionError, match="unknown result attempt"):
        session._keep_canonical_attempt("unknown", {})
    assert session._resolve_ambiguous_result("task", "result", {}) is None
