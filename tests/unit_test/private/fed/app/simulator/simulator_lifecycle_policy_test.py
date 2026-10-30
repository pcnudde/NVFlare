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
from argparse import Namespace
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from nvflare.apis.fl_constant import MachineStatus, RunnerTask
from nvflare.apis.job_def import JobMetaKey
from nvflare.fuel.common.exit_codes import ProcessExitCode
from nvflare.private.fed.app.simulator import simulator_runner as subject
from nvflare.private.fed.client.client_status import ClientStatus
from tests.timing_utils import isolate_time


@pytest.fixture
def runner(tmp_path):
    return subject.SimulatorRunner("job", str(tmp_path / "workspace"))


@pytest.fixture
def client_runner(tmp_path):
    args = Namespace(workspace=str(tmp_path), set=[], threads=2, end_run_for_all=True)
    clients = [SimpleNamespace(client_name=name, simulate_running=False) for name in ("site-a", "site-b")]
    return subject.SimulatorClientRunner(None, args, clients, {}, None, {})


@pytest.mark.parametrize(
    "reported,file_value,exitcode,expected",
    [
        (7, None, 1, 7),
        (None, "4\n", 1, 4),
        (0, None, 0, 0),
        (None, "broken\n", 3, 3),
        (None, None, -9, -9),
    ],
)
def test_return_code_prefers_reported_status_then_rc_file_then_exitcode(
    runner, tmp_path, reported, file_value, exitcode, expected
):
    rc_file = tmp_path / subject.FLMetaKey.PROCESS_RC_FILE
    if file_value is not None:
        rc_file.write_text(file_value)
    assert (
        runner._get_return_code({"run_status": reported}, SimpleNamespace(pid=42, exitcode=exitcode), str(tmp_path))
        == expected
    )
    if file_value == "4\n":
        assert not rc_file.exists()


def test_run_uses_controlled_manager_and_process_boundaries(runner, monkeypatch):
    statuses = {"run_status": 7}
    manager = Mock()
    manager.dict.return_value = statuses
    process = Mock()
    factory = Mock(return_value=process)
    monkeypatch.setattr(subject, "Manager", Mock(return_value=manager))
    monkeypatch.setattr(subject, "Process", factory)
    assert runner.run() == 7
    factory.assert_called_once_with(target=runner.run_process, args=(statuses,))
    process.start.assert_called_once_with()
    process.join.assert_called_once_with()


def test_run_keyboard_interrupt_kills_owned_children(runner, monkeypatch):
    monkeypatch.setattr(subject, "Manager", Mock(side_effect=KeyboardInterrupt))
    kill = Mock()
    monkeypatch.setattr(subject, "kill_child_processes", kill)
    assert runner.run() == -9
    kill.assert_called_once_with(subject.os.getpid())


@pytest.mark.parametrize("failure", [False, True])
def test_run_process_records_monitor_status_and_exceptions(runner, monkeypatch, failure):
    monitor = Mock(side_effect=ValueError("failed") if failure else None, return_value=4)
    monkeypatch.setattr(subject.mpm, "run", monitor)
    statuses = {}
    runner.run_process(statuses)
    assert statuses == {"run_status": ProcessExitCode.EXCEPTION if failure else 4}
    monitor.assert_called_once_with(
        main_func=runner.simulator_run_main, run_dir=runner.workspace, shutdown_grace_time=3, cleanup_grace_time=6
    )


def test_simulation_stops_after_invalid_setup(runner, monkeypatch):
    monkeypatch.setattr(runner, "setup", Mock(return_value=False))
    runner.deployer = Mock()
    assert runner.simulator_run_main() == 1
    runner.deployer.close.assert_not_called()


def test_simulation_closes_deployer_when_client_creation_fails(runner, monkeypatch):
    monkeypatch.setattr(runner, "setup", Mock(return_value=True))
    monkeypatch.setattr(runner, "create_clients", Mock(side_effect=RuntimeError("no clients")))
    runner.deployer = Mock()
    assert runner.simulator_run_main() == 2
    runner.deployer.close.assert_called_once_with()


def test_simulation_dispatches_gpu_groups_then_aborts_server_and_closes(runner, monkeypatch):
    monkeypatch.setattr(runner, "setup", Mock(return_value=True))
    monkeypatch.setattr(runner, "create_clients", Mock())
    monkeypatch.setattr(runner, "_get_server_run_status", Mock(return_value=0))
    runner.args = Namespace(gpu="0,1")
    runner.server_custom_folder = "/custom"
    runner.server = Mock()
    runner.server.engine.engine_info.status = MachineStatus.STARTED
    runner.server.engine.run_processes = {}
    runner.federated_clients = ["a", "b", "c"]
    runner.deployer = Mock()
    thread = Mock()
    monkeypatch.setattr(subject, "threading", SimpleNamespace(Thread=Mock(return_value=thread)))
    executor = Mock()
    executor_factory = Mock(return_value=executor)
    monkeypatch.setattr(subject, "ThreadPoolExecutor", executor_factory)
    assert runner.simulator_run_main() == 0
    executor_factory.assert_called_once_with(max_workers=2)
    assert [call.args[1] for call in executor.submit.call_args_list] == [
        ["/custom", ["a", "c"], "0"],
        ["/custom", ["b"], "1"],
    ]
    thread.start.assert_called_once_with()
    thread.join.assert_called_once_with()
    runner.server.abort_run.assert_called_once_with()
    runner.deployer.close.assert_called_once_with()


def test_create_clients_assigns_deployment_and_ready_status(runner):
    clients = [SimpleNamespace(client_name=name) for name in ("site-a", "site-b")]
    runner.client_names = [client.client_name for client in clients]
    runner.args = Namespace()
    runner.simulator_root = runner.workspace
    runner.deployer = Mock()
    runner.deployer.create_fl_client.side_effect = [
        (client, {"client": i}, "deploy", "ctx") for i, client in enumerate(clients)
    ]
    runner.create_clients()
    assert runner.federated_clients == clients
    assert runner.client_config == {"client": 1}
    for client in clients:
        assert client.status == ClientStatus.STARTED
        assert client.simulate_running is False
        assert client.args is runner.args
        assert client.client_name in client.app_client_root


@pytest.mark.parametrize(
    "gpu,threads,clients,host,expected_threads,valid",
    [
        (None, None, 2, [], 1, True),
        ("0,1", 5, 2, [0, 1], 1, True),
        ("0", None, 2, [0], 1, True),
        ("0,1", 1, 1, [0, 1], 1, False),
        ("2", 1, 2, [0], 1, False),
        ("[0", 1, 2, [0], 1, False),
        (None, 4, 2, [], 4, False),
        (None, 0, 2, [], 0, False),
    ],
)
def test_setup_validates_gpu_and_thread_capacity_without_starting_runtime(
    runner, monkeypatch, tmp_path, gpu, threads, clients, host, expected_threads, valid
):
    monkeypatch.chdir(tmp_path)
    runner.gpu, runner.threads, runner.n_clients = gpu, threads, clients
    for name in (
        "nvflare_fobs_initialize",
        "init_security_content_service",
        "dynamic_log_config",
        "custom_fobs_initialize",
        "register_ext_decomposers",
    ):
        monkeypatch.setattr(subject, name, Mock())
    monkeypatch.setattr(subject.AuthorizationService, "initialize", Mock())
    monkeypatch.setattr(subject.AuditService, "the_auditor", None)
    monkeypatch.setattr(subject, "get_host_gpu_ids", Mock(return_value=host))
    monkeypatch.setattr(subject, "secure_log_traceback", Mock())
    monkeypatch.setattr(subject.ConfigService, "get_str_var", Mock(return_value=None))
    monkeypatch.setattr(
        runner, "validate_job_data", Mock(return_value=(b"job", "job", {JobMetaKey.DEPLOY_MAP: {"app": ["@ALL"]}}))
    )
    monkeypatch.setattr(runner, "_deploy_apps", Mock())
    monkeypatch.setattr(subject, "Workspace", Mock())
    runner.deployer = Mock()
    server = Mock()
    server.get_cell.return_value.get_root_url_for_child.return_value = "tcp://localhost:1234"
    runner.deployer.create_fl_server.return_value = ({}, server)
    assert runner.setup() is valid
    assert runner.args.threads == expected_threads
    assert runner.deployer.create_fl_server.call_count == int(valid)
    if valid:
        assert runner.args.sp_target == "localhost:1234"
        assert runner.args.sp_scheme == "tcp"
        assert runner.client_names == [f"site-{i + 1}" for i in range(clients)]


def test_shutdown_client_stops_heartbeat_transport_and_marks_stopped(client_runner):
    client = Mock()
    client_runner._shutdown_client(client)
    assert client.communicator.heartbeat_done is True
    client.terminate.assert_called_once_with()
    client.communicator.cell.stop.assert_called_once_with()
    assert client.status == ClientStatus.STOPPED


def test_shutdown_client_contains_cleanup_exceptions(client_runner):
    client = Mock()
    client.terminate.side_effect = OSError("already gone")
    client_runner.logger = Mock()
    client_runner._shutdown_client(client)
    client_runner.logger.warning.assert_called_once()
    client.communicator.cell.stop.assert_not_called()


def test_end_run_clients_picks_only_unfinished_idle_clients(client_runner):
    first, second = client_runner.federated_clients
    client_runner.clients_finished_end_run = [first.client_name]
    task = Mock()
    client_runner.do_one_task = task
    lock = threading.Lock()
    client_runner._end_run_clients("0", lock, 2, 13)
    task.assert_called_once_with(second, 2, "0", lock, timeout=13, task_name=RunnerTask.END_RUN)
    assert client_runner.clients_finished_end_run == [first.client_name, second.client_name]
    assert not second.simulate_running
    assert client_runner._pick_next_client() is None


def test_client_thread_reuses_selected_client_and_records_end_run(client_runner, monkeypatch):
    clock = isolate_time(monkeypatch, subject)
    # A finite budget rejects accidental extra polling rather than hanging the suite.
    monkeypatch.setattr(clock, "sleep", Mock(side_effect=[None, None]))
    first, second = client_runner.federated_clients
    client_runner.do_one_task = Mock(side_effect=[(False, second, None), (True, None, second.client_name)])
    client_runner._end_run_clients = Mock()
    lock = threading.Lock()
    client_runner.run_client_thread(2, "0", lock, True, timeout=8)
    assert [c.args[0] for c in client_runner.do_one_task.call_args_list] == [first, second]
    assert client_runner.clients_finished_end_run == [second.client_name]
    assert not first.simulate_running and not second.simulate_running
    client_runner._end_run_clients.assert_called_once_with("0", lock, 2, 8)


def test_connection_retry_timeout_uses_isolated_advancing_clock(client_runner, monkeypatch):
    clock = isolate_time(monkeypatch, subject)
    monkeypatch.setattr(clock, "time", Mock(side_effect=[0, 1, 3]))
    monkeypatch.setattr(clock, "sleep", Mock(side_effect=[None]))
    connect = Mock(side_effect=ConnectionRefusedError)
    monkeypatch.setattr(subject, "Client", connect)
    with pytest.raises(RuntimeError, match="timeout: 2"):
        client_runner._create_connection(1234, timeout=2)
    assert connect.call_count == 2
