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

import threading
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from nvflare.apis.fl_constant import ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.apis.signal import Signal
from nvflare.app_common.executors.client_api import attach_backend as subject
from nvflare.app_common.executors.client_api.backend_spec import ClientAPIBackendContext
from nvflare.client.cell.defs import MsgKey, Topic
from nvflare.fuel.f3.cellnet.defs import MessageHeaderKey
from nvflare.fuel.f3.cellnet.defs import ReturnCode as CellReturnCode
from nvflare.fuel.f3.cellnet.utils import make_reply
from nvflare.fuel.f3.drivers.driver_params import DriverParams
from nvflare.fuel.f3.streaming.stream_const import STREAM_CHANNEL, STREAM_DATA_TOPIC, StreamHeaderKey
from tests.timing_utils import isolate_time


@pytest.fixture
def backend():
    backend = subject.AttachBackend()
    backend._context = ClientAPIBackendContext(executor=Mock(), attach_id="trainer", attach_timeout=2)
    backend._cell = Mock()
    backend._cj_fqcn, backend._job_id, backend._site_name = "site.job", "job", "site"
    backend._trainer_fqcn = "site.trainer"
    backend._session = subject._AttachedTrainerSession("site.trainer")
    return backend


@pytest.mark.parametrize(
    "config",
    [
        [],
        {subject.ATTACH_COMM_CONFIG: []},
        {subject.ATTACH_COMM_CONFIG: {"scheme": "tcp://bad", "resources": {}}},
        {subject.ATTACH_COMM_CONFIG: {"scheme": "file", "resources": None}},
        {subject.ATTACH_COMM_CONFIG: {"scheme": "tcp", "resources": {}}},
    ],
)
def test_invalid_site_listener_configuration_fails_before_starting_transport(backend, config):
    backend._cell.core_cell.comm_configurator.get_config.return_value = config
    with pytest.raises(ValueError):
        backend._start_shared_file_listener()
    backend._cell.core_cell.communicator.start_listener.assert_not_called()


def test_shared_file_listener_requires_communicator_and_cleans_partial_publish_state(backend, monkeypatch):
    core = backend._cell.core_cell
    core.comm_configurator.get_config.return_value = {
        subject.ATTACH_COMM_CONFIG: {"scheme": subject.SHARED_FILE_SCHEME, "resources": {}}
    }
    core.communicator = None
    with pytest.raises(RuntimeError, match="communicator"):
        backend._start_shared_file_listener()
    core.communicator = Mock()
    core.communicator.start_listener.side_effect = OSError("driver failed")
    with pytest.raises(RuntimeError, match="cannot start"):
        backend._start_shared_file_listener()
    backend._attach_listener_params = {}
    with pytest.raises(ValueError, match="requires"):
        backend._publish_shared_file_endpoint()
    assert not backend._shared_file_listener_is_protected({})
    params = {DriverParams.URL.value: "file://x", subject.SHARED_FILE_ROOT_DIR: "/root"}
    monkeypatch.setattr(subject, "validate_shared_file_listener", Mock(side_effect=ValueError("unsafe")))
    assert not backend._shared_file_listener_is_protected(params)
    with pytest.raises(ValueError, match="shared-file"):
        backend._shared_file_route_kind({DriverParams.SCHEME.value: "tcp"})


def test_protocol_guard_filter_installation_is_atomic_and_guard_reuse_is_checked(backend):
    core = SimpleNamespace(add_incoming_filter=Mock(side_effect=RuntimeError("install")))
    backend._cell.core_cell = core
    with pytest.raises(RuntimeError, match="install"):
        backend._install_secure_protocol_guard()
    assert not hasattr(core, subject._ATTACH_PROTOCOL_GUARD_ATTR)
    core.add_incoming_filter = Mock()
    backend._install_secure_protocol_guard()
    guard = backend._protocol_guard
    backend._install_secure_protocol_guard()
    assert core.add_incoming_filter.call_count == 3
    backend._remove_secure_protocol_guard()
    assert not guard._claims
    setattr(core, subject._ATTACH_PROTOCOL_GUARD_ATTR, object())
    with pytest.raises(RuntimeError, match="incompatible"):
        backend._install_secure_protocol_guard()
    assert backend._secure_protocol_guard(make_reply(CellReturnCode.OK)) is None


def test_guard_ignores_other_destinations_and_resolves_stream_return_only_route():
    guard = subject._AttachProtocolGuard("site.job", Mock())
    message = make_reply(CellReturnCode.OK)
    message.set_header(MessageHeaderKey.DESTINATION, "other")
    assert guard(message) is None
    message.set_header(MessageHeaderKey.DESTINATION, "site.job")
    message.set_header(MessageHeaderKey.CHANNEL, STREAM_CHANNEL)
    message.set_header(MessageHeaderKey.TOPIC, STREAM_DATA_TOPIC)
    message.set_header(StreamHeaderKey.CHANNEL, subject.CellChannel.RETURN_ONLY)
    message.set_header(StreamHeaderKey.TOPIC, f"{subject.CHANNEL}:task")
    assert guard._application_route(message) == (subject.CHANNEL, "task")
    assert guard(message) is None


@pytest.mark.parametrize("reason", ["complete", "aborted", "closed", "lost"])
def test_attach_cancel_signal_latches_each_terminal_condition(backend, reason):
    abort = Signal()
    completion = threading.Event()
    backend._session.ready.set()
    cancel = subject._AttachCancelSignal(abort, backend, backend._session.session_id, completion)
    assert not cancel.triggered
    if reason == "complete":
        completion.set()
    elif reason == "aborted":
        abort.trigger(True)
    elif reason == "closed":
        backend._closed = True
    else:
        backend._session.error = "lost"
    assert cancel.triggered
    assert cancel.triggered


@pytest.mark.parametrize(
    "reply,error",
    [
        (None, None),
        (make_reply(CellReturnCode.COMM_ERROR), None),
        (make_reply(CellReturnCode.OK, body=[]), "invalid SESSION_OPEN"),
        (make_reply(CellReturnCode.OK, body={MsgKey.REPLY_TOPIC: Topic.SESSION_REJECTED}), "rejected"),
        (
            make_reply(
                CellReturnCode.OK, body={MsgKey.REPLY_TOPIC: Topic.SESSION_ACCEPTED, MsgKey.SESSION_ID: "wrong"}
            ),
            "topic/session",
        ),
    ],
)
def test_session_open_rejects_invalid_control_replies(backend, reply, error):
    backend._cell.core_cell.send_request.return_value = reply
    backend._try_session_open(backend._session)
    assert not backend._session.ready.is_set()
    if error:
        assert error in backend._session.error
    else:
        assert backend._session.error is None


def test_session_open_and_status_transport_failures_are_retryable(backend):
    backend._cell.core_cell.send_request.side_effect = RuntimeError("route")
    backend._try_session_open(backend._session)
    assert backend._session.error is None
    backend._cell.send_request.side_effect = RuntimeError("status")
    assert backend._query_task_status(backend._session, "task") is None
    backend._cell.send_request.assert_called_once()
    assert backend._query_task_status(backend._session, "task", timeout=0) is None
    assert backend._cell.send_request.call_count == 1
    backend._cell.send_request.side_effect = None
    backend._cell.send_request.return_value = make_reply(CellReturnCode.OK, body=[])
    assert backend._query_task_status(backend._session, "task") is None


def test_inspection_failure_is_not_evidence_of_source_disconnect(backend):
    backend._session.result_source_disconnect_since = 10
    backend._cell.is_cell_connected.side_effect = RuntimeError("inspect")
    assert not backend._result_source_confirmed_disconnected(backend._session)
    assert backend._session.result_source_disconnect_since is None


@pytest.mark.parametrize("outcome", ["closed", "busy-abort", "busy", "no-session", "exception"])
def test_execute_rejects_unavailable_admission_and_always_releases_owned_gate(backend, outcome, monkeypatch):
    abort = Signal()
    if outcome == "closed":
        backend._closed = True
    elif outcome.startswith("busy"):
        backend._execute_gate.acquire()
        if outcome == "busy-abort":
            abort.trigger(True)
    elif outcome == "no-session":
        monkeypatch.setattr(backend, "_wait_for_session", Mock(return_value=None))
    else:
        monkeypatch.setattr(backend, "_wait_for_session", Mock(side_effect=RuntimeError("session")))
    result = backend.execute("train", Shareable(), FLContext(), abort)
    assert result.get_return_code() == (
        ReturnCode.TASK_ABORTED if outcome == "busy-abort" else ReturnCode.EXECUTION_EXCEPTION
    )
    if outcome.startswith("busy"):
        backend._execute_gate.release()
    assert backend._execute_gate.acquire(blocking=False)
    backend._execute_gate.release()


def test_listener_close_contains_cleanup_errors_and_is_idempotent(backend):
    publisher = Mock()
    publisher.close.side_effect = RuntimeError("publisher")
    backend._endpoint_publisher = publisher
    backend._attach_listener_handle = "listener"
    backend._cell.core_cell.communicator.remove_connector.side_effect = RuntimeError("connector")
    backend._close_attach_listener()
    backend._close_attach_listener()
    publisher.close.assert_called_once()
    backend._cell.core_cell.communicator.remove_connector.assert_called_once_with("listener")
    assert backend._endpoint_publisher is None
    assert backend._attach_listener_handle is None


def test_source_release_wait_has_finite_clock_budget(backend, monkeypatch):
    backend._session.result_source_live.set()
    monkeypatch.setattr(backend, "_result_source_confirmed_disconnected", Mock(return_value=False))
    clock = isolate_time(monkeypatch, subject)
    clock.monotonic = Mock(side_effect=[0, 1000])
    backend.logger.warning = Mock()
    backend._wait_for_result_source_release(backend._session)
    backend.logger.warning.assert_called_once()
