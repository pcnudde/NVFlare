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

from nvflare.apis.shareable import Shareable
from nvflare.apis.signal import Signal
from nvflare.app_common.executors.client_api import external_process_backend as subject
from nvflare.app_common.executors.client_api.backend_spec import ClientAPIBackendContext
from nvflare.client.cell.defs import MsgKey, Topic
from nvflare.fuel.f3.cellnet.defs import MessageHeaderKey, ReturnCode
from nvflare.fuel.f3.message import Message


@pytest.fixture
def backend():
    value = subject.ExternalProcessBackend()
    value._context = ClientAPIBackendContext(executor=Mock())
    value._site_name = "site"
    value._site_auth_token = "token"
    value._site_auth_token_signature = "signature"
    return value


@pytest.mark.parametrize("case", ["closed", "invalid-payload", "absent-launch", "unknown-session"])
def test_readiness_rejects_invalid_state_without_publishing_ready(backend, case):
    trainer = subject._TrainerSession("token", "site.trainer")
    trainer.session_id = "session"
    backend._active_launch = trainer
    request = Message({MessageHeaderKey.ORIGIN: "site.trainer"}, {MsgKey.SESSION_ID: "session"})
    if case == "closed":
        backend._closed = True
    elif case == "invalid-payload":
        request.payload = []
    elif case == "absent-launch":
        backend._active_launch = None
    else:
        request.payload[MsgKey.SESSION_ID] = "stale"
    reply = backend._handle_session_ready(request)
    if case == "invalid-payload":
        assert reply.get_header(MessageHeaderKey.RETURN_CODE) == ReturnCode.INVALID_REQUEST
    else:
        assert reply.payload[MsgKey.REPLY_TOPIC] == Topic.ERROR
    assert not trainer.ready.is_set()


def test_hello_without_active_launch_cannot_create_session(backend):
    response = backend._handle_hello(Message({}, {}))
    assert response.payload[MsgKey.REPLY_TOPIC] == Topic.HELLO_REJECTED
    assert "no active trainer launch" in response.payload[MsgKey.REASON]
    assert backend._active_launch is None


def test_abort_intent_wins_over_concurrent_result_source_failure(backend):
    backend._run_abort_signal = Signal()
    backend._run_abort_signal.trigger("user cancelled")
    assert not backend._claim_result_source_failure()
    assert backend._terminal_intent == subject._TERMINAL_INTENT_ABORT
    assert not backend._failure_panic_sent


@pytest.mark.parametrize("failure", ["collect", "thread-start"])
def test_result_source_monitor_start_failure_does_not_leave_phantom_thread(backend, monkeypatch, failure):
    trainer = subject._TrainerSession("token", "site.trainer")
    task = SimpleNamespace(task_id="task", result_receiver_ids=("server.job",))
    collect = Mock(side_effect=ValueError("bad result") if failure == "collect" else None, return_value={"reference"})
    monkeypatch.setattr(backend, "_collect_result_source_refs", collect)
    error = Mock()
    monkeypatch.setattr(backend.logger, "error", error)
    thread = Mock()
    thread.start.side_effect = RuntimeError("thread failed")
    monkeypatch.setattr(subject, "threading", SimpleNamespace(Thread=Mock(return_value=thread)))
    backend._on_result_accepted(trainer, task, Shareable())
    assert trainer.source_monitor_thread is None
    assert thread.start.call_count == int(failure == "thread-start")
    error.assert_called_once()
