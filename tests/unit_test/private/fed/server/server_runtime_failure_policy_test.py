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
from nvflare.apis.fl_component import FLComponent
from nvflare.apis.fl_constant import MachineStatus, RunProcessKey, SnapshotKey
from nvflare.apis.fl_context import FLContext
from nvflare.fuel.f3.cellnet.utils import new_cell_message
from nvflare.private.fed.server import fed_server, server_engine
from tests.timing_utils import isolate_time


@pytest.fixture
def server():
    value = object.__new__(fed_server.FederatedServer)
    value.logger = Mock()
    value.client_manager = Mock()
    value.client_manager.clients = {}
    value.engine = Mock()
    value.engine.run_processes = {}
    value.engine.exception_run_processes = {}
    value.engine.new_context.side_effect = lambda: nullcontext(FLContext())
    value.cell = Mock()
    value.server_runner = Mock()
    value.lock = threading.Lock()
    value.shutdown = False
    value.heart_beat_timeout = 10
    value.shutdown_period = 2
    value.executor = Mock()
    return value


def test_dead_client_cleanup_removes_only_heartbeats_past_deadline(server, monkeypatch):
    expired = SimpleNamespace(token="old", name="expired", last_connect_time=89)
    live = SimpleNamespace(token="live", name="live", last_connect_time=90)
    server.client_manager.get_clients.return_value = {"old": expired, "live": live}
    logout = Mock(return_value=expired)
    monkeypatch.setattr(server, "logout_client", logout)
    clock = isolate_time(monkeypatch, fed_server)
    monkeypatch.setattr(clock, "time", lambda: 100)
    server.remove_dead_clients()
    logout.assert_called_once_with("old")


def test_cleanup_loop_checks_shutdown_frequently_and_removal_on_interval(server, monkeypatch):
    clock = isolate_time(monkeypatch, fed_server)
    monkeypatch.setattr(clock, "time", Mock(side_effect=[10, 11]))
    sleeps = []

    def pause(interval):
        sleeps.append(interval)
        assert len(sleeps) <= 2, "cleanup ignored the shutdown flag"
        if len(sleeps) == 2:
            server.shutdown = True

    monkeypatch.setattr(clock, "sleep", pause)
    cleanup = Mock()
    monkeypatch.setattr(server, "remove_dead_clients", cleanup)
    server.client_cleanup()
    assert sleeps == [0.2, 0.2]
    cleanup.assert_called_once_with()


def test_server_shutdown_times_out_remaining_clients_then_closes_executor(server, monkeypatch):
    server.client_manager.clients = {"token": object()}
    clock = isolate_time(monkeypatch, fed_server)
    monkeypatch.setattr(clock, "time", Mock(side_effect=[0, 3]))
    monkeypatch.setattr(clock, "sleep", Mock(side_effect=[None]))
    close = Mock()
    monkeypatch.setattr(server, "close", close)
    fed_server.BaseServer.fl_shutdown(server)
    assert server.shutdown
    close.assert_called_once_with()
    server.executor.shutdown.assert_called_once_with()


def test_close_contains_unlocked_sync_lock_error(server):
    assert fed_server.BaseServer.close(server) == 0
    assert any("canceling sync locks" in c.args[0] for c in server.logger.info.call_args_list)


def test_server_auth_headers_are_signed_once_and_token_checks_use_verifier(server, monkeypatch):
    server.my_own_token_signature = None
    server.my_own_auth_client_name = "server-auth"
    server.my_own_token = "token"
    sign = Mock(return_value="signature")
    monkeypatch.setattr(server, "sign_auth_token", sign)
    headers = Mock()
    monkeypatch.setattr(fed_server, "add_authentication_headers", headers)
    message = new_cell_message({}, None)
    server._add_auth_headers(message)
    server._add_auth_headers(message)
    sign.assert_called_once_with("server-auth", "token")
    assert headers.call_count == 2
    headers.assert_called_with(message, "server-auth", "token", "signature")


def test_signature_and_verification_use_credential_asserter_and_preserve_no_pki_policy(server, monkeypatch):
    asserter = Mock()
    monkeypatch.setattr(server, "_get_id_asserter", Mock(return_value=asserter))
    verifier = Mock()
    monkeypatch.setattr(fed_server, "TokenVerifier", Mock(return_value=verifier))
    assert server.sign_auth_token("site", "token") is asserter.sign.return_value
    asserter.sign.assert_called_once_with("sitetoken", return_str=True)
    assert server.verify_auth_token("site", "token", "signature") is verifier.verify.return_value
    verifier.verify.assert_called_once_with("site", "token", "signature")
    server._get_id_asserter.return_value = None
    assert server.sign_auth_token("site", "token") == "NA"
    assert server.verify_auth_token("site", "token", "signature") is True


def test_registration_session_expiration_drops_stale_entries_only(server, monkeypatch):
    server.reg_lock = threading.Lock()
    server.max_reg_duration = 5
    server.name_to_reg = {"old": SimpleNamespace(reg_start_time=0), "new": SimpleNamespace(reg_start_time=7)}
    clock = isolate_time(monkeypatch, fed_server)
    monkeypatch.setattr(clock, "time", lambda: 10)

    class StopPolling(Exception):
        pass

    monkeypatch.setattr(clock, "sleep", Mock(side_effect=StopPolling))
    with pytest.raises(StopPolling):
        server._check_regs()
    assert set(server.name_to_reg) == {"new"}


def test_terminal_outcome_failure_is_published_then_resolved_for_disconnected_client(server):
    client = Client("site-a", "token")
    server.engine.job_runner.is_client_outcome_pending.return_value = True
    server._resolve_missing_client_outcome(client, "job", "lost")
    assert server.engine.job_runner.fail_run.call_args.args[:2] == (
        "job",
        fed_server.ProcessExitCode.INFRASTRUCTURE_ERROR,
    )
    server.engine.job_runner.resolve_client_outcome.assert_called_once_with("job", "site-a")
    server.engine.job_runner.is_client_outcome_pending.return_value = False
    server._resolve_missing_client_outcome(client, "other", "lost")
    assert server.engine.job_runner.resolve_client_outcome.call_count == 1


@pytest.fixture
def engine(tmp_path):
    value = object.__new__(server_engine.ServerEngine)
    value.logger = Mock()
    value.lock = threading.Lock()
    value.server = Mock()
    value.run_processes = {}
    value.exception_run_processes = {}
    value.run_manager = Mock()
    value.client_manager = Mock()
    value.engine_info = SimpleNamespace(status=MachineStatus.STARTED)
    value.args = SimpleNamespace(workspace=str(tmp_path))
    value.executor = Mock()
    return value


def test_wait_for_complete_bounds_missing_status_update_and_records_exit_failure(engine, monkeypatch):
    process = Mock()
    engine.run_processes = {"job": {RunProcessKey.PARTICIPANTS: {}}}
    clock = isolate_time(monkeypatch, server_engine)
    monkeypatch.setattr(clock, "time", Mock(side_effect=[0, 3]))
    monkeypatch.setattr(clock, "sleep", Mock(side_effect=AssertionError("should already time out")))
    monkeypatch.setattr(server_engine, "get_return_code", Mock(return_value=7))
    engine.wait_for_complete("workspace", "job", process)
    process.wait.assert_called_once_with()
    assert not engine.run_processes
    assert engine.exception_run_processes["job"][RunProcessKey.PROCESS_RETURN_CODE] == 7
    assert engine.engine_info.status == MachineStatus.STOPPED


@pytest.mark.parametrize("method", ["show_stats", "get_errors", "reset_errors", "configure_job_log"])
def test_engine_admin_queries_contain_transport_failures(engine, monkeypatch, method):
    monkeypatch.setattr(engine, "send_command_to_child_runner_process", Mock(side_effect=OSError("offline")))
    result = (
        getattr(engine, method)("job", {"level": "INFO"})
        if method == "configure_job_log"
        else getattr(engine, method)("job")
    )
    if method in ("show_stats", "get_errors"):
        assert result == {}
    elif method == "reset_errors":
        assert result == "reset the server error stats for job: job"
    else:
        assert "offline" in result
    engine.logger.error.assert_called_once()


def test_component_restore_uses_component_snapshots_and_restores_context(engine):
    component = Mock(spec=FLComponent)
    engine.run_manager.components = {"model": component, "plain": object()}
    snapshot = Mock()
    snapshot.get_component_snapshot.side_effect = lambda component_id: (
        {"ctx": {"value": 7}} if component_id == SnapshotKey.FL_CONTEXT else {"weights": [1]}
    )
    ctx = FLContext()
    engine.restore_components(snapshot, ctx)
    component.restore.assert_called_once_with({"weights": [1]}, ctx)
    assert ctx.props["ctx"] == {"value": 7}


@pytest.mark.parametrize("method", ["disable_clients", "enable_clients"])
def test_bulk_client_policy_reports_per_client_errors_but_single_failure_propagates(engine, method):
    manager = engine.server.client_manager
    if method == "disable_clients":
        manager.is_client_disabled.return_value = False
        manager.disable_client.side_effect = [OSError("disk full"), []]
    else:
        manager.enable_client.side_effect = [OSError("disk full"), True]
    result = getattr(engine, method)(["bad", "good"])
    assert result["clients"][0] == {"client_name": "bad", "state": "error", "error": "disk full"}
    assert result["clients"][1]["client_name"] == "good"
    operation = manager.disable_client if method == "disable_clients" else manager.enable_client
    operation.side_effect = OSError("disk full")
    with pytest.raises(OSError, match="disk full"):
        getattr(engine, method)(["bad"])
