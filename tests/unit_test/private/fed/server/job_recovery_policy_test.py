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
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from nvflare.apis.client import Client
from nvflare.apis.event_type import EventType
from nvflare.apis.fl_constant import RunProcessKey, SystemComponents
from nvflare.apis.fl_context import FLContext
from nvflare.apis.job_def import JobMetaKey, RunStatus
from nvflare.private.defs import RequestHeader, TrainingTopic
from nvflare.private.fed.server import job_runner as subject


@pytest.fixture
def env(tmp_path):
    runner = subject.JobRunner(str(tmp_path))
    runner.scheduler = Mock()
    runner.log_error = Mock()
    runner.log_info = Mock()
    runner.log_debug = Mock()
    engine = Mock()
    engine.run_processes = {}
    engine.exception_run_processes = {}
    engine.lock = threading.Lock()
    engine.abort_app_on_server.return_value = ""
    engine.delete_job_id.return_value = ""
    ctx = Mock()
    ctx.get_engine.return_value = engine
    return SimpleNamespace(runner=runner, engine=engine, ctx=ctx, manager=engine.get_component.return_value)


def test_client_send_resolves_tokens_and_uses_admin_default_timeout():
    engine = Mock()
    client = Client("site-a", "token")
    engine.validate_targets.return_value = ([client], [])
    admin = Mock()
    admin.timeout = 13
    ctx = FLContext()
    admin.sai.new_context.return_value = nullcontext(ctx)
    message = Mock()
    replies = subject._send_to_clients(admin, ["site-a"], engine, message, optional=True)
    admin.send_requests.assert_called_once_with({"token": message}, ctx, timeout_secs=13, optional=True)
    assert replies is admin.send_requests.return_value
    engine.validate_targets.return_value = ([], ["unknown"])
    with pytest.raises(RuntimeError, match="unknown clients"):
        subject._send_to_clients(admin, ["unknown"], engine, message)
    assert admin.send_requests.call_count == 1


def test_lifecycle_events_initialize_scheduler_save_workspace_and_stop(env, monkeypatch):
    env.runner.scheduler = None
    save = Mock()
    monkeypatch.setattr(env.runner, "_save_workspace", save)
    env.runner.handle_event(EventType.SYSTEM_START, env.ctx)
    env.engine.get_component.assert_called_once_with(SystemComponents.JOB_SCHEDULER)
    assert env.runner.scheduler is env.engine.get_component.return_value
    env.runner.handle_event(EventType.END_RUN, env.ctx)
    save.assert_called_once_with(env.ctx)
    env.runner.handle_event(EventType.SYSTEM_END, env.ctx)
    assert env.runner.ask_to_stop


def test_stop_run_aborts_only_connected_original_participants_then_server(env, monkeypatch):
    connected = Client("active", "token-a")
    disconnected = Client("gone", "token-b")
    env.engine.client_manager.clients = {connected.token: connected}
    env.engine.run_processes["job"] = {
        RunProcessKey.PARTICIPANTS: {connected.token: connected, disconnected.token: disconnected}
    }
    abort = Mock()
    monkeypatch.setattr(env.runner, "abort_client_run", abort)
    env.engine.abort_app_on_server.return_value = "server unreachable"
    env.runner._stop_run("job", env.ctx)
    abort.assert_called_once_with("job", ["active"], env.ctx)
    env.engine.abort_app_on_server.assert_called_once_with("job")
    assert "server unreachable" in env.runner.log_error.call_args.args[1]
    env.runner._stop_run("unknown", env.ctx)
    assert abort.call_count == 1


@pytest.mark.parametrize("failure", [False, True])
def test_abort_client_run_is_optional_bounded_and_contains_unknown_client_error(env, monkeypatch, failure):
    send = Mock(side_effect=RuntimeError("unknown client") if failure else None)
    monkeypatch.setattr(subject, "_send_to_clients", send)
    env.runner.abort_client_run("job", ["site-a"], env.ctx)
    call = send.call_args
    assert call.args[:3] == (env.engine.server.admin_server, ["site-a"], env.engine)
    message = call.args[3]
    assert message.topic == TrainingTopic.ABORT
    assert message.get_header(RequestHeader.JOB_ID) == "job"
    assert call.kwargs == {"timeout": 2.0, "optional": True}
    assert env.runner.log_error.call_count == int(failure)


@pytest.mark.parametrize("failure", [False, True])
def test_delete_run_attempts_server_cleanup_even_when_client_delivery_fails(env, monkeypatch, failure):
    send = Mock(side_effect=RuntimeError("delivery failed") if failure else None, return_value=[])
    check = Mock()
    monkeypatch.setattr(subject, "_send_to_clients", send)
    monkeypatch.setattr(subject, "check_client_replies", check)
    env.engine.delete_job_id.return_value = "cannot remove"
    env.runner._delete_run("job", ["site-a"], env.ctx)
    assert send.call_args.args[3].topic == TrainingTopic.DELETE_RUN
    env.engine.delete_job_id.assert_called_once_with("job")
    assert check.call_count == int(not failure)
    assert env.runner.log_error.call_count == (2 if failure else 1)


@pytest.mark.parametrize("failure", [None, "server", "scheduler"])
def test_restore_running_job_records_participants_and_reports_start_failures(env, failure):
    client = Client("site-a", "token")
    clients = {client.token: client}
    job = SimpleNamespace(job_id="job", meta={})
    env.manager.get_job.return_value = job
    env.engine.start_app_on_server.return_value = "cannot start" if failure == "server" else ""
    if failure == "scheduler":
        env.runner.scheduler.restore_scheduled_job.side_effect = RuntimeError("scheduler unavailable")
    env.runner.restore_running_job("job", clients, "snapshot", env.ctx)
    env.engine.start_app_on_server.assert_called_once_with(env.ctx, job=job, job_clients=clients, snapshot="snapshot")
    if failure == "server":
        assert env.runner.running_jobs == {}
        env.runner.scheduler.restore_scheduled_job.assert_not_called()
    else:
        assert env.runner.running_jobs == {"job": job}
        assert env.runner._pending_client_outcomes == {"job": {"site-a"}}
        env.runner.scheduler.restore_scheduled_job.assert_called_once_with("job")
    assert env.runner.log_error.call_count == int(failure is not None)


@pytest.mark.parametrize(
    "method,status",
    [("update_abnormal_finished_jobs", RunStatus.FINISHED_ABNORMAL), ("update_unfinished_jobs", RunStatus.ABANDONED)],
)
def test_recovery_updates_each_orphan_and_continues_after_persistence_failure(env, method, status):
    jobs = [SimpleNamespace(job_id=name) for name in ("failed", "orphan", "active")]
    env.manager.get_jobs_by_status.return_value = jobs
    env.manager.set_status.side_effect = [OSError("disk error"), None, None]
    if method == "update_abnormal_finished_jobs":
        getattr(env.runner, method)(["active"], env.ctx)
    else:
        getattr(env.runner, method)(env.ctx)
    expected = jobs[:2] if method == "update_abnormal_finished_jobs" else jobs
    assert [c.args[:2] for c in env.manager.set_status.call_args_list] == [(j.job_id, status) for j in expected]
    env.runner.log_error.assert_called_once()
    env.manager.get_jobs_by_status.assert_called_once_with([RunStatus.RUNNING, RunStatus.DISPATCHED], env.ctx)


def test_stop_all_runs_stops_runner_after_dispatching_each_job(env, monkeypatch):
    env.engine.run_processes = {"a": {}, "b": {}}
    stop = Mock()
    monkeypatch.setattr(env.runner, "stop_run", stop)
    env.runner.stop_all_runs(env.ctx)
    assert [c.args for c in stop.call_args_list] == [("a", env.ctx), ("b", env.ctx)]
    assert env.runner.ask_to_stop


def test_running_job_removal_clears_outcome_bookkeeping_and_scheduler(env):
    env.runner.running_jobs = {"job": object()}
    env.runner._pending_client_outcomes = {"job": {"site-a"}}
    env.runner._client_outcome_deadlines = {"job": 1}
    env.runner.remove_running_job("job")
    assert env.runner.running_jobs == env.runner._pending_client_outcomes == env.runner._client_outcome_deadlines == {}
    env.runner.scheduler.remove_scheduled_job.assert_called_once_with("job")


def test_status_change_check_reloads_persistent_job(env):
    env.manager.get_job.return_value = SimpleNamespace(meta={JobMetaKey.STATUS: RunStatus.RUNNING})
    assert not env.runner._check_job_status(env.manager, "job", RunStatus.RUNNING, env.ctx)
    assert env.runner._check_job_status(env.manager, "job", RunStatus.FINISHED_ABORTED, env.ctx)
