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

from nvflare.apis.fl_context import FLContext
from nvflare.apis.signal import Signal
from nvflare.app_common.executors.client_api import attach_backend as subject
from nvflare.app_common.executors.client_api.backend_spec import ClientAPIBackendContext


@pytest.fixture
def backend():
    value = subject.AttachBackend()
    value._context = ClientAPIBackendContext(executor=Mock(), attach_id="trainer", attach_timeout=2)
    value._session = subject._AttachedTrainerSession("site.trainer")
    value._session.session_id = "session"
    value._cell = Mock()
    return value


@pytest.mark.parametrize("case", ["no-session", "rejected", "expired", "healthy"])
def test_session_monitor_stops_on_terminal_admission_state_and_bounds_healthy_poll(backend, monkeypatch, case):
    session = backend._session
    if case == "no-session":
        backend._session = None
    elif case == "rejected":
        session.error = "rejected"
    elif case == "healthy":
        session.ready.set()
        monkeypatch.setattr(backend, "_liveness_error", Mock(return_value=None))
    monkeypatch.setattr(backend, "_deadline_expired", Mock(return_value=True))
    stop = Mock()
    stop.is_set.side_effect = [False, True]
    stop.wait.side_effect = [None]
    backend._session_stop = stop
    backend._session_loop()
    assert stop.wait.call_count == int(case == "healthy")
    if case == "expired":
        assert "attach_timeout=2" in session.error
        assert not session.ready.is_set()
    elif case == "healthy":
        assert session.ready.is_set() and session.error is None


def test_attach_admission_requires_id_and_predecode_guard_without_opening_transport(backend):
    with pytest.raises(ValueError, match="requires attach_id"):
        backend.initialize(ClientAPIBackendContext(executor=Mock()), FLContext())
    backend._cell.core_cell = SimpleNamespace()
    with pytest.raises(RuntimeError, match="pre-decode"):
        backend._install_secure_protocol_guard()
    with pytest.raises(RuntimeError, match="communication configuration"):
        backend._start_shared_file_listener()
    backend._cell.core_cell.comm_configurator = SimpleNamespace(get_config=lambda: None)
    assert backend._start_shared_file_listener() is None


def test_session_wait_aborts_or_expires_without_returning_unready_session(backend, monkeypatch):
    abort = Signal()
    abort.trigger(True)
    assert backend._wait_for_session(abort) is None
    session = backend._session
    session.ready = Mock()
    session.ready.wait.side_effect = [False]
    monkeypatch.setattr(backend, "_deadline_expired", Mock(return_value=True))
    assert backend._wait_for_session(Signal()) is None
    session.ready.wait.assert_called_once()
    assert backend._query_task_status(session, "task", timeout=0) is None
    backend._cell.send_request.assert_not_called()


def test_transfer_cleanup_contains_failure_for_each_owned_transaction(monkeypatch):
    delete = Mock(side_effect=[OSError("first unavailable"), None])
    monkeypatch.setattr(subject.DownloadService, "delete_transaction", delete)
    subject.AttachBackend._delete_transfers(
        [SimpleNamespace(transaction_id="one"), SimpleNamespace(transaction_id="two")]
    )
    assert [item.args[0] for item in delete.call_args_list] == ["one", "two"]
