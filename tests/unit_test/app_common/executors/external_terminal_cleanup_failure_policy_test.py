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
from nvflare.app_common.executors.client_api import external_process_backend as subject
from nvflare.app_common.executors.client_api.backend_spec import ClientAPIBackendContext
from tests.timing_utils import isolate_time


@pytest.fixture
def runtime(monkeypatch):
    backend = subject.ExternalProcessBackend()
    backend._context = ClientAPIBackendContext(executor=Mock())
    trainer = subject._TrainerSession("token", "site.trainer")
    trainer.session_id = "session"
    backend._active_launch, backend._cell = trainer, Mock()
    error = Mock()
    monkeypatch.setattr(backend.logger, "error", error)
    return backend, trainer, error


@pytest.mark.parametrize("state", ["not-ready", "dead"])
def test_persistent_task_admission_requires_ready_live_trainer(runtime, monkeypatch, state):
    backend, trainer, _ = runtime
    finish, execute = Mock(), Mock()
    monkeypatch.setattr(backend, "_finish_task_trainer", finish)
    monkeypatch.setattr(backend, "_run_task", execute)
    if state == "dead":
        trainer.ready.set()
        monkeypatch.setattr(backend, "_trainer_liveness_error", Mock(return_value="lost heartbeat"))
    result = backend._execute_admitted_task("train", Shareable(), FLContext(), Signal())
    assert result.get_return_code() == ReturnCode.EXECUTION_EXCEPTION
    execute.assert_not_called()
    finish.assert_called_once_with(trainer, True, task_aborted=False)
    assert backend._abort is (state == "dead")


def test_closed_launch_and_closed_hello_wait_fail_without_spawning_or_polling_process(runtime, monkeypatch):
    backend, trainer, _ = runtime
    backend._closed = True
    with pytest.raises(RuntimeError, match="not launching"):
        backend._launch_trainer(1)
    trainer.ready = Mock()
    trainer.ready.wait.side_effect = [False]
    clock = isolate_time(monkeypatch, subject)
    clock.monotonic = Mock(side_effect=[0])
    with pytest.raises(RuntimeError, match="closed while waiting"):
        backend._wait_for_hello(trainer, 1)
    trainer.ready.wait.assert_called_once()


def test_failed_shutdown_delivery_does_not_mark_shutdown_acknowledged(runtime, monkeypatch):
    backend, trainer, _ = runtime
    monkeypatch.setattr(backend, "_process_group_alive", Mock(return_value=True))
    backend._cell.fire_and_forget.return_value = {trainer.trainer_fqcn: "offline"}
    warning = Mock()
    monkeypatch.setattr(backend.logger, "warning", warning)
    backend._request_trainer_shutdown(trainer, wait_timeout=0)
    assert not trainer.shutdown_requested.is_set()
    assert "offline" in warning.call_args.args[0]


def test_unwind_failure_still_closes_backend_and_disables_pass_through(runtime, monkeypatch):
    backend, trainer, error = runtime
    stop = Mock(side_effect=RuntimeError("stop failed"))
    disable = Mock()
    monkeypatch.setattr(backend, "_stop_trainer", stop)
    monkeypatch.setattr(backend, "_disable_task_pass_through", disable)
    backend._unwind()
    assert backend._closed and backend._terminal_intent == subject._TERMINAL_INTENT_SHUTDOWN
    stop.assert_called_once_with(trainer, natural_exit_wait=0.0)
    disable.assert_called_once_with()
    assert "stop failed" in error.call_args.args[0]


def test_retiring_single_task_trainer_contains_stop_failure(runtime, monkeypatch):
    backend, trainer, error = runtime
    monkeypatch.setattr(backend, "_stop_trainer", Mock(side_effect=RuntimeError("stop failed")))
    backend._finish_task_trainer(trainer, launch_once=False)
    error.assert_called_once()
    assert "stop failed" in error.call_args.args[0]


@pytest.mark.parametrize("hard", [False, True])
def test_platform_process_signal_error_is_contained_without_any_real_signal(runtime, monkeypatch, hard):
    backend, trainer, _ = runtime
    monkeypatch.setattr(subject, "os", SimpleNamespace(name="nt"))
    trainer.process = Mock()
    trainer.process.poll.return_value = None
    trainer.process.kill.side_effect = OSError("denied")
    trainer.process.terminate.side_effect = OSError("denied")
    debug = Mock()
    monkeypatch.setattr(backend.logger, "debug", debug)
    backend._signal_process_tree(trainer, hard)
    assert trainer.process.kill.call_count == int(hard)
    assert trainer.process.terminate.call_count == int(not hard)
    assert "denied" in debug.call_args.args[0]


def test_source_failure_notification_error_always_releases_delivery_barrier(runtime, monkeypatch):
    backend, trainer, error = runtime
    trainer.result_source_live.set()
    trainer.result_source_task_id = "task"
    trainer.result_source_refs, trainer.result_receiver_ids = ("reference",), ("server.job",)
    notify = Mock(side_effect=OSError("receiver unreachable"))
    monkeypatch.setattr(subject.DownloadService, "notify_source_failure", notify)
    assert backend._fail_accepted_result_source(trainer, "task", "trainer exited")
    assert trainer.result_failure_delivery_done.is_set() and trainer.result_failure_notified
    assert not trainer.result_source_live.is_set() and trainer.result_source_task_id is None
    assert notify.call_args.kwargs["ref_ids"] == ("reference",)
    assert "receiver unreachable" in error.call_args.args[0]


def test_fatal_source_failure_keeps_abort_and_exit_code_when_panic_publication_fails(runtime, monkeypatch):
    backend, _, error = runtime
    backend._terminal_intent = subject._TERMINAL_INTENT_FAILURE
    backend._run_abort_signal = Signal()
    backend._engine = Mock()
    backend._engine.new_context.side_effect = RuntimeError("context unavailable")
    write = Mock()
    monkeypatch.setattr(backend, "_write_process_exit_code", write)
    backend._fail_job_for_lost_result_source("result lost")
    assert backend._abort and backend._run_abort_signal.triggered and backend._failure_panic_sent
    write.assert_called_once_with(subject.ProcessExitCode.EXCEPTION)
    assert "failed to publish fatal" in error.call_args.args[0]
    backend._fail_job_for_lost_result_source("repeated")
    assert write.call_count == 1
