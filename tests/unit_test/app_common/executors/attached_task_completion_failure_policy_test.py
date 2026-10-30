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

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from nvflare.apis.fl_constant import ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.apis.signal import Signal
from nvflare.app_common.app_constant import AppConstants
from nvflare.app_common.executors.client_api import attach_backend as subject
from nvflare.app_common.executors.client_api.backend_spec import ClientAPIBackendContext
from nvflare.client.cell.defs import MsgKey, TaskState, Topic
from nvflare.fuel.f3.cellnet.defs import ReturnCode as CellReturnCode
from nvflare.fuel.f3.cellnet.utils import make_reply
from tests.timing_utils import isolate_time


@pytest.fixture
def backend():
    backend = subject.AttachBackend()
    backend._context = ClientAPIBackendContext(executor=Mock(), task_wait_timeout=1, result_wait_timeout=1)
    backend._session = subject._AttachedTrainerSession("site.trainer")
    backend._session.session_id = "session"
    backend._session.ready.set()
    backend._cell = Mock()
    return backend


@pytest.mark.parametrize("case", ["delivery-abort", "waiting-abort", "timeout", "invalid-result", "complete"])
def test_task_completion_restores_active_task_and_preserves_round(backend, monkeypatch, case):
    abort, request = Signal(), Shareable()
    request.set_header(AppConstants.CURRENT_ROUND, 7)
    event = Mock()
    event.is_set.side_effect = [False]
    event.wait.side_effect = [case in ("invalid-result", "complete")]
    task = SimpleNamespace(task_id="task", result_ready=event, result=Shareable() if case == "complete" else [])
    monkeypatch.setattr(subject, "CellTask", Mock(return_value=task))

    def deliver(*args):
        if case in ("delivery-abort", "waiting-abort"):
            abort.trigger(True)
        return (case != "delivery-abort", "transport failed")

    monkeypatch.setattr(backend, "_deliver_task", Mock(side_effect=deliver))
    send_abort = Mock()
    monkeypatch.setattr(backend, "_send_abort", send_abort)
    clock = isolate_time(monkeypatch, subject)
    clock.monotonic = Mock(side_effect=[0, 2])
    result = backend._run_task(backend._session, "train", request, FLContext(), abort)
    assert backend._current_task is None
    if case == "complete":
        assert result is task.result and result.get_header(AppConstants.CURRENT_ROUND) == 7
        send_abort.assert_not_called()
    else:
        expected = ReturnCode.TASK_ABORTED if "abort" in case else ReturnCode.EXECUTION_EXCEPTION
        assert result.get_return_code() == expected
        assert send_abort.call_count == int(case != "invalid-result")


@pytest.mark.parametrize("case", ["already-complete", "status-recovered", "cancelled"])
def test_task_delivery_recovery_is_bounded_by_status_and_cancellation(backend, monkeypatch, case):
    clock = isolate_time(monkeypatch, subject)
    clock.monotonic = Mock(side_effect=[0, 0.1, 0.2, 0.3, 2])
    ready = Mock()
    ready.is_set.side_effect = [True] if case == "already-complete" else [False] * 5
    task = SimpleNamespace(task_id="task", result_ready=ready)
    backend._cell.send_request.return_value = None
    status = Mock(return_value=TaskState.DELIVERED if case == "status-recovered" else TaskState.UNKNOWN)
    monkeypatch.setattr(backend, "_query_task_status", status)
    matches = Mock(return_value=False if case == "cancelled" else True)
    monkeypatch.setattr(backend, "_session_matches", matches)
    accepted, reason = backend._deliver_task(backend._session, task, 1, "train", Shareable(), Signal())
    assert accepted == (case != "cancelled")
    if case == "already-complete":
        backend._cell.send_request.assert_not_called()
        status.assert_not_called()
    else:
        backend._cell.send_request.assert_called_once()
        status.assert_called_once()
    if case == "cancelled":
        assert "no reply" in reason


@pytest.mark.parametrize(
    "reply,terminal,fragment",
    [
        (None, False, "no reply"),
        (make_reply(CellReturnCode.COMM_ERROR), False, "cell-level failure"),
        (make_reply(CellReturnCode.OK, body=[]), True, "invalid TASK_READY reply payload"),
        (
            make_reply(
                CellReturnCode.OK, body={MsgKey.REPLY_TOPIC: Topic.TASK_STATUS, MsgKey.TASK_STATE: TaskState.UNKNOWN}
            ),
            False,
            "still preparing",
        ),
        (
            make_reply(CellReturnCode.OK, body={MsgKey.REPLY_TOPIC: "unexpected"}),
            True,
            "invalid TASK_READY reply topic",
        ),
    ],
)
def test_task_acknowledgement_classifies_retryable_transport_and_terminal_protocol_errors(reply, terminal, fragment):
    reason, actual_terminal = subject.AttachBackend._check_task_accepted(reply)
    assert fragment in reason and actual_terminal is terminal


def test_shutdown_transport_failure_does_not_clear_live_result_source(backend, monkeypatch):
    backend._session.result_source_live.set()
    backend._cell.send_request.side_effect = OSError("offline")
    debug = Mock()
    monkeypatch.setattr(backend.logger, "debug", debug)
    backend._request_shutdown(backend._session)
    assert backend._session.result_source_live.is_set()
    debug.assert_called_once_with("attached trainer did not acknowledge SHUTDOWN")
