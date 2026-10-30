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
from unittest.mock import MagicMock, Mock

import pytest

from nvflare.apis.fl_constant import FLMetaKey
from nvflare.apis.signal import Signal
from nvflare.app_common.executors.client_api import external_process_backend as subject
from nvflare.app_common.executors.client_api.backend_spec import ClientAPIBackendContext
from nvflare.client.cell.defs import MsgKey, Topic
from nvflare.fuel.f3.cellnet.defs import ReturnCode as CellReturnCode
from nvflare.fuel.f3.cellnet.utils import make_reply
from tests.timing_utils import isolate_time


@pytest.fixture
def runtime(monkeypatch):
    backend = subject.ExternalProcessBackend()
    backend._context = ClientAPIBackendContext(executor=Mock(), task_wait_timeout=1)
    backend._cell = Mock()
    trainer = subject._TrainerSession("token", "site.trainer")
    trainer.process = Mock()
    trainer.process.poll.return_value = None
    trainer.pgid = 12345
    monkeypatch.setattr(backend, "_process_group_alive", Mock(return_value=True))
    return backend, trainer


def test_task_ready_cancel_signal_latches_callback_exception_once():
    callback = Mock(side_effect=RuntimeError("inspect"))
    cancel = subject._TaskReadyCancelSignal(callback)
    assert cancel.triggered
    assert cancel.triggered
    assert str(cancel.error) == "inspect"
    callback.assert_called_once()


@pytest.mark.parametrize("cause", ["abort", "closed", "process-dead", "timeout", "inspection-error"])
def test_task_ready_delivery_cancels_and_discards_payload_for_terminal_condition(runtime, monkeypatch, cause):
    backend, trainer = runtime
    signal = Signal()
    if cause == "abort":
        signal.trigger(True)
    elif cause == "closed":
        backend._closed = True
    elif cause == "process-dead":
        backend._process_group_alive.return_value = False
    elif cause == "inspection-error":
        backend._process_group_alive.side_effect = RuntimeError("inspect")
    clock = isolate_time(monkeypatch, subject)
    clock.monotonic = Mock(side_effect=[0, 2])
    delete = Mock()
    monkeypatch.setattr(backend, "_delete_task_transfers", delete)

    def send(**kwargs):
        assert kwargs["abort_signal"].triggered
        raise RuntimeError("cancelled transport")

    backend._cell.send_request.side_effect = send
    if cause == "inspection-error":
        with pytest.raises(RuntimeError, match="inspect"):
            backend._send_task_ready(trainer, {}, signal)
    else:
        status, _ = backend._send_task_ready(trainer, {}, signal)
        assert (
            status
            == {
                "abort": subject._SEND_ABORTED,
                "closed": subject._SEND_CLOSED,
                "process-dead": subject._SEND_PROCESS_DEAD,
                "timeout": subject._SEND_SESSION_DEAD,
            }[cause]
        )
    delete.assert_called_once_with([])


def test_rejected_task_ready_payload_and_failed_transfer_cleanup_preserve_original_result(runtime, monkeypatch):
    backend, trainer = runtime
    backend._context = ClientAPIBackendContext(executor=Mock())
    backend._cell.send_request.return_value = make_reply(
        CellReturnCode.OK, body={MsgKey.REPLY_TOPIC: Topic.TASK_FAILED, MsgKey.REASON: "unsupported"}
    )
    delete = Mock()
    monkeypatch.setattr(backend, "_delete_task_transfers", delete)
    status, reply = backend._send_task_ready(trainer, {}, Signal())
    assert status == subject._SEND_OK
    assert "unsupported" in backend._check_task_accepted(reply)
    delete.assert_called_once_with([])
    transaction_delete = Mock(side_effect=RuntimeError("delete"))
    monkeypatch.setattr(subject.DownloadService, "delete_transaction", transaction_delete)
    subject.ExternalProcessBackend._delete_task_transfers([SimpleNamespace(transaction_id="tx")])
    transaction_delete.assert_called_once_with("tx")


def test_stop_trainer_attempts_all_cleanup_phases_even_when_shutdown_and_termination_fail(runtime, monkeypatch):
    backend, trainer = runtime
    backend._request_trainer_shutdown = Mock(side_effect=RuntimeError("shutdown"))
    backend._terminate_process_tree = Mock(side_effect=RuntimeError("signal"))
    backend._cleanup_trainer = Mock()
    clock = isolate_time(monkeypatch, subject)
    clock.monotonic = Mock(side_effect=range(10))
    backend._stop_trainer(trainer, natural_exit_wait=0)
    backend._request_trainer_shutdown.assert_called_once()
    backend._terminate_process_tree.assert_called_once()
    backend._cleanup_trainer.assert_called_once_with(trainer)
    trainer._cleaned = True
    backend._stop_trainer(trainer, natural_exit_wait=0)
    assert backend._cleanup_trainer.call_count == 1


def test_cleanup_trainer_contains_log_and_bootstrap_cleanup_failures_and_releases_protocol_state(
    runtime, monkeypatch, tmp_path
):
    backend, trainer = runtime
    backend._active_launch = trainer
    backend._protocol_sessions[trainer.trainer_fqcn] = trainer
    trainer.log_thread = Mock()
    trainer.log_thread.is_alive.side_effect = RuntimeError("log thread")
    trainer.bootstrap_path = str(tmp_path / "bootstrap")
    (tmp_path / "bootstrap").write_text("{}")
    backend._process_group_alive.return_value = False
    monkeypatch.setattr(
        subject, "os", SimpleNamespace(path=subject.os.path, remove=Mock(side_effect=OSError("unlink")))
    )
    backend._cleanup_trainer(trainer)
    backend._cleanup_trainer(trainer)
    assert trainer._cleaned
    assert trainer.token == ""
    assert trainer.session_id is None
    assert backend._active_launch is None
    assert trainer.trainer_fqcn not in backend._protocol_sessions
    assert (tmp_path / "bootstrap").exists()


def test_reaper_start_failure_is_unregistered_and_retryable(runtime, monkeypatch):
    backend, trainer = runtime
    thread = Mock()
    thread.start.side_effect = RuntimeError("thread start")
    monkeypatch.setattr(subject, "threading", SimpleNamespace(Thread=Mock(return_value=thread)))
    with pytest.raises(RuntimeError, match="thread start"):
        backend._reap_trainer_after_result(trainer)
    assert trainer not in backend._result_reapers
    assert trainer.reaper_thread is None


def test_proc_stat_parser_uses_last_comm_delimiter_and_rejects_malformed_input(tmp_path):
    path = tmp_path / "stat"
    path.write_bytes(b"123 (trainer (rank) name)) Z 1 123 123 0 0")
    assert subject.ExternalProcessBackend._read_linux_proc_stat_fields(str(path))[:4] == [b"Z", b"1", b"123", b"123"]
    path.write_bytes(b"malformed")
    with pytest.raises(ValueError, match="delimiter"):
        subject.ExternalProcessBackend._read_linux_proc_stat_fields(str(path))


@pytest.mark.parametrize("fault", ["malformed", "unreadable", "vanished"])
def test_procfs_thread_inspection_treats_uncertainty_as_unknown(runtime, monkeypatch, fault):
    backend, _ = runtime
    entries = MagicMock()
    entries.__enter__.return_value = [SimpleNamespace(name="123"), SimpleNamespace(name="name")]
    monkeypatch.setattr(subject, "os", SimpleNamespace(scandir=Mock(return_value=entries)))
    reader = Mock(return_value=[])
    if fault != "malformed":
        reader.side_effect = PermissionError("denied") if fault == "unreadable" else FileNotFoundError("gone")
    monkeypatch.setattr(subject.ExternalProcessBackend, "_read_linux_proc_stat_fields", reader)
    assert backend._linux_process_has_live_threads("123") is None


def test_exit_code_write_reports_missing_or_unwritable_run_directory(runtime, tmp_path, monkeypatch):
    backend, _ = runtime
    monkeypatch.setattr(backend.logger, "error", Mock())
    backend._run_dir = None
    backend._write_process_exit_code(3)
    backend._run_dir = str(tmp_path / "missing")
    backend._write_process_exit_code(3)
    assert backend.logger.error.call_count == 2
    backend._run_dir = str(tmp_path)
    backend._write_process_exit_code(3)
    assert (tmp_path / FLMetaKey.PROCESS_RC_FILE).read_text() == "3"
