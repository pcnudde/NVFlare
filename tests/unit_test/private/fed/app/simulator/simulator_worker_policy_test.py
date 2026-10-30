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

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from nvflare.apis.event_type import EventType
from nvflare.apis.fl_constant import FLContextKey
from nvflare.apis.fl_context import FLContext
from nvflare.private.fed.app.simulator import simulator_worker as worker_module
from nvflare.private.fed.client.client_status import ClientStatus


def _client(runner):
    context = FLContext()
    context.set_prop(FLContextKey.RUNNER, runner)
    client = Mock(client_name="site-1")
    client.run_manager.new_context.side_effect = lambda: context
    return client, context


@pytest.mark.parametrize(
    "processed,aborted,task_name,expected_calls",
    [
        (True, False, "train", 1),
        (False, False, "train", 3),
        (False, True, "train", 1),
        (False, False, worker_module.RunnerTask.END_RUN, 0),
    ],
)
def test_task_iteration_stops_on_result_abort_end_run_or_retry_budget(
    monkeypatch, processed, aborted, task_name, expected_calls
):
    runner = Mock()
    runner.fetch_and_run_one_task.return_value = (2.0, processed)
    runner.run_abort_signal.triggered = aborted
    client, context = _client(runner)
    worker = worker_module.ClientTaskWorker()
    worker.fire_event = Mock()
    sleep = Mock()
    monkeypatch.setattr(worker_module, "time", SimpleNamespace(sleep=sleep))
    interval, stop, ended_client = worker.do_one_task(client, SimpleNamespace(task_name=task_name))
    worker.fire_event.assert_called_once_with(EventType.SWAP_IN, context)
    assert runner.fetch_and_run_one_task.call_count == expected_calls
    assert stop is (aborted or task_name == worker_module.RunnerTask.END_RUN)
    assert ended_client == ("site-1" if aborted else None)
    assert interval == (1.0 if expected_calls == 0 else 2.0)
    assert runner.end_run_events_sequence.call_count == int(stop)
    assert sleep.call_count == (2 if not processed and not aborted and expected_calls else 0)


def test_task_creation_and_execution_errors_stop_run_without_escaping(monkeypatch):
    client = Mock(client_name="site-1", run_manager=None)
    worker = worker_module.ClientTaskWorker()
    worker.create_client_runner = Mock(side_effect=RuntimeError("configuration rejected"))
    monkeypatch.setattr(worker_module, "secure_log_traceback", Mock())
    assert worker.do_one_task(client, SimpleNamespace(task_name="train")) == (1.0, True, None)
    worker.create_client_runner.assert_called_once_with(client)


def test_release_removes_runner_from_context_after_swap_out():
    runner = Mock()
    client, context = _client(runner)
    worker = worker_module.ClientTaskWorker()
    events = []
    worker.fire_event = lambda event, ctx: events.append((event, ctx.get_prop(FLContextKey.RUNNER)))
    worker.release_resources(client)
    assert events == [(EventType.SWAP_OUT, runner)]
    assert context.get_prop(FLContextKey.RUNNER) is None
    client.run_manager = None
    worker.release_resources(client)
    assert len(events) == 1


def test_client_engine_initialization_associates_engine_and_fires_system_start(monkeypatch):
    client = Mock()
    context = FLContext()
    engine = Mock()
    engine.new_context.return_value = context
    factory = Mock(return_value=engine)
    monkeypatch.setattr(worker_module, "ClientEngine", factory)
    args = object()
    worker_module.ClientTaskWorker().create_client_engine(client, args, rank=4)
    factory.assert_called_once_with(client, args, 4)
    client.set_client_engine.assert_called_once_with(engine)
    engine.fire_event.assert_called_once_with(EventType.SYSTEM_START, context)
    assert client.run_manager is None


def test_client_runner_inherits_task_identity_and_initializes_job_cell(monkeypatch):
    context = FLContext()
    runner = Mock()
    runner.engine.new_context.return_value = context
    app_runner = Mock()
    app_runner.create_client_runner.return_value = runner
    monkeypatch.setattr(worker_module, "SimulatorClientAppRunner", lambda: app_runner)
    args = SimpleNamespace(config_folder="config")
    client = SimpleNamespace(
        app_client_root="/workspace/app", args=args, client_name="site-1", token="token", cell=object()
    )
    worker_module.ClientTaskWorker().create_client_runner(client)
    assert args.client_name == "site-1" and args.token == "token"
    app_runner.create_client_runner.assert_called_once_with("/workspace/app", args, "config", client, False)
    app_runner.start_command_agent.assert_called_once_with(args, client, context)
    app_runner.sync_up_parents_process.assert_called_once_with(client)
    assert runner.engine.cell is client.cell
    runner.init_run.assert_called_once_with("/workspace/app", args)


@pytest.mark.parametrize("stop,continue_run", [(False, False), (True, False)])
def test_worker_protocol_reports_completion_and_always_stops_owned_cell(monkeypatch, tmp_path, stop, continue_run):
    worker = worker_module.ClientTaskWorker()
    client = Mock(client_name="site-1")
    worker._create_client = Mock(return_value=client)
    worker.create_client_engine = Mock()
    worker.do_one_task = Mock(return_value=(0.5, stop, "site-1" if stop else None))
    worker.release_resources = Mock()
    args = SimpleNamespace(simulator_root=str(tmp_path))
    conn = Mock()
    conn.recv.side_effect = [
        {
            worker_module.SimulatorConstants.CLIENT_CONFIG: {},
            worker_module.SimulatorConstants.DEPLOY_ARGS: "args",
            worker_module.SimulatorConstants.BUILD_CTX: "build",
        },
        continue_run,
    ]
    monkeypatch.setattr(worker_module, "get_simulator_app_root", lambda *args: str(tmp_path / "app"))
    worker.run(args, conn)
    assert [c.args[0] for c in conn.send.call_args_list] == ([True, "site-1"] if stop else [False])
    worker.release_resources.assert_called_once_with(client)
    client.cell.stop.assert_called_once_with()


def test_worker_receive_error_is_contained_before_client_creation():
    worker = worker_module.ClientTaskWorker()
    worker._create_client = Mock()
    conn = Mock()
    conn.recv.side_effect = EOFError("parent disconnected")
    worker.run(SimpleNamespace(), conn)
    worker._create_client.assert_not_called()
    conn.send.assert_not_called()


def test_client_deployment_sets_identity_started_state_and_connection(monkeypatch, tmp_path):
    client = Mock(client_name="site-1")
    deployer = Mock()
    deployer.create_fed_client.return_value = client
    monkeypatch.setattr(worker_module, "BaseClientDeployer", lambda: deployer)
    monkeypatch.setattr(worker_module, "get_simulator_app_root", lambda root, name: str(tmp_path / name))
    worker = worker_module.ClientTaskWorker()
    worker._create_client_cell = Mock()
    args = SimpleNamespace(
        token="token", simulator_root=str(tmp_path), root_url="tcp://root", parent_url="tcp://parent"
    )
    deploy_args = object()
    assert worker._create_client(args, {"configuration": True}, deploy_args) is client
    assert client.token == "token" and client.args is deploy_args
    assert client.app_client_root == str(tmp_path / "site-1")
    assert client.simulate_running is False and client.status == ClientStatus.STARTED
    deployer.build.assert_called_once_with({"configuration": True})
    worker._create_client_cell.assert_called_once_with(client, "tcp://root", "tcp://parent")


def test_cell_creation_registers_cleanup_and_auth_before_checking_readiness(monkeypatch):
    cell = Mock()
    cell.is_cell_connected.return_value = True
    factory = Mock(return_value=cell)
    cleanup = Mock()
    monkeypatch.setattr(worker_module, "Cell", factory)
    monkeypatch.setattr(worker_module.mpm, "add_cleanup_cb", cleanup)
    client = Mock(client_name="site-1", token="token")
    worker_module.ClientTaskWorker()._create_client_cell(client, "tcp://root", "ignored-parent")
    assert factory.call_args.kwargs["parent_url"] is None
    assert factory.call_args.kwargs["secure"] is False
    cell.start.assert_called_once_with()
    cleanup.assert_called_once_with(cell.stop)
    assert client.cell is cell
    client.communicator.set_cell.assert_called_once_with(cell)
    client.communicator.set_auth.assert_called_once_with(
        client_name="site-1", token="token", token_signature="NA", ssid="NA"
    )


def test_cell_readiness_has_bounded_timeout_using_isolated_clock(monkeypatch):
    cell = Mock()
    cell.is_cell_connected.return_value = False
    monkeypatch.setattr(worker_module, "Cell", lambda **kwargs: cell)
    monkeypatch.setattr(worker_module.mpm, "add_cleanup_cb", Mock())
    timestamps = iter([0.0, worker_module.CELL_CONNECT_CHECK_TIMEOUT + 1])
    sleep = Mock()
    monkeypatch.setattr(worker_module, "time", SimpleNamespace(time=lambda: next(timestamps), sleep=sleep))
    with pytest.raises(RuntimeError, match="Could not connect"):
        worker_module.ClientTaskWorker()._create_client_cell(Mock(client_name="site-1"), "root", None)
    sleep.assert_called_once_with(0.1)


def test_connection_helper_accepts_on_requested_local_port_without_network(monkeypatch):
    listener = Mock()
    listener.accept.return_value = "connection"
    factory = Mock(return_value=listener)
    monkeypatch.setattr(worker_module, "Listener", factory)
    assert worker_module._create_connection("8412") == "connection"
    factory.assert_called_once_with(("localhost", 8412))
    listener.accept.assert_called_once_with()


def test_worker_arguments_require_identity_and_parse_optional_gpu(monkeypatch):
    argv = [
        "worker",
        "--workspace",
        "/workspace",
        "--logging_config",
        "log.json",
        "--client",
        "site-1",
        "--token",
        "token",
        "--port",
        "8000",
        "--parent_pid",
        "123",
        "--decomposer_module",
        "custom",
        "--gpu",
        "0",
        "--task_name",
        "train",
    ]
    monkeypatch.setattr(worker_module.sys, "argv", argv)
    args = worker_module.parse_arguments()
    assert args.workspace == "/workspace" and args.client == "site-1"
    assert args.parent_pid == 123 and args.gpu == "0" and args.task_name == "train"
    monkeypatch.setattr(worker_module.sys, "argv", ["worker", "--workspace", "/workspace"])
    with pytest.raises(SystemExit):
        worker_module.parse_arguments()
