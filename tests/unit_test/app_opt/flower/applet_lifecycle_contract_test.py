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

import shlex
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from nvflare.apis.fl_constant import ReservedKey
from nvflare.apis.fl_context import FLContext
from nvflare.apis.workspace import Workspace
from nvflare.app_opt.flower import applet as subject
from nvflare.app_opt.flower.defs import Constant


@pytest.fixture
def context(tmp_path):
    ctx = FLContext()
    ctx.set_prop(ReservedKey.IDENTITY_NAME, "site-b")
    ctx.set_prop(ReservedKey.RUN_NUM, "job")
    engine = MagicMock()
    engine.get_clients.return_value = [SimpleNamespace(name="site-b"), SimpleNamespace(name="site-a")]
    workspace = MagicMock(spec=Workspace)
    workspace.get_root_dir.return_value = str(tmp_path)
    workspace.get_app_dir.return_value = str(tmp_path / "app")
    workspace.get_app_config_dir.return_value = str(tmp_path / "config")
    workspace.get_app_custom_dir.return_value = str(tmp_path / "custom")
    workspace.get_run_dir.return_value = str(tmp_path / "run")
    engine.get_workspace.return_value = workspace
    ctx.set_prop(ReservedKey.ENGINE, engine)
    return ctx


def test_client_command_contains_sorted_partition_and_metrics_bootstrap(context, monkeypatch):
    monkeypatch.setattr(subject, "_validate_flower_executable", MagicMock())
    monkeypatch.setenv("PATH", "/usr/bin")
    client = subject.FlowerClientApplet(extra_env={"KEY": "value"})
    desc = client.get_command(
        {
            Constant.APP_CTX_SUPERLINK_ADDR: "127.0.0.1:9000",
            Constant.APP_CTX_CLIENTAPP_API_ADDR: "127.0.0.1:9001",
            Constant.APP_CTX_FL_CONTEXT: context,
        }
    )
    argv = shlex.split(desc.cmd)
    assert argv[argv.index("--superlink") + 1] == "127.0.0.1:9000"
    assert "partition-id=1" in argv[argv.index("--node-config") + 1]
    assert "num-partitions=2" in argv[argv.index("--node-config") + 1]
    assert desc.cwd.endswith("app") and desc.env["KEY"] == "value"
    assert desc.env[subject.ANALYTICS_BOOTSTRAP_ENV].endswith("config/analytics_bootstrap.json")
    context.set_prop(ReservedKey.IDENTITY_NAME, "not-registered")
    assert subject.get_partition_id(context) == -1
    assert "partition-id=" not in client._get_node_config(context)


def test_flower_executable_preflight_distinguishes_missing_and_non_executable(tmp_path):
    path = tmp_path / "flwr"
    with pytest.raises(RuntimeError, match="not found"):
        subject._validate_flower_executable("flwr", str(path))
    path.write_text("#!/bin/sh")
    path.chmod(0o600)
    with pytest.raises(RuntimeError, match="not executable"):
        subject._validate_flower_executable("flwr", str(path))
    path.chmod(0o700)
    subject._validate_flower_executable("flwr", str(path))


@pytest.mark.parametrize("database,predeployed", [(":memory:", False), ("database file.db", True)])
def test_server_start_prepares_home_and_sequences_process_readiness_then_submission(
    context, monkeypatch, tmp_path, database, predeployed
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(subject, "_validate_flower_executable", MagicMock())
    order = []
    process = MagicMock()

    def start(desc, ctx, stop_method):
        assert stop_method == "term"
        order.append(("process", desc))
        return process

    monkeypatch.setattr(subject, "start_process", start)
    monkeypatch.setattr(subject, "create_channel", lambda **kw: order.append(("ready", kw)))
    server = subject.FlowerServerApplet(database, 4, flower_app_path="local/custom/app" if predeployed else None)
    if predeployed:
        (tmp_path / "local/custom/app").mkdir(parents=True)
    server._run_flower_command = MagicMock(
        side_effect=lambda *args, **kw: order.append(("run", (args, kw))) or {"run-id": "run-123"}
    )
    server.start(
        {
            Constant.APP_CTX_FL_CONTEXT: context,
            Constant.APP_CTX_SERVERAPP_API_ADDR: "localhost:1",
            Constant.APP_CTX_FLEET_API_ADDR: "localhost:2",
            Constant.APP_CTX_EXEC_API_ADDR: "localhost:3",
        }
    )
    assert [kind for kind, _ in order] == ["process", "ready", "run"]
    assert server.run_id == "run-123" and server._superlink_process_mgr is process
    argv = shlex.split(order[0][1].cmd)
    assert argv[argv.index("--database") + 1] == (database if database == ":memory:" else str(tmp_path / database))
    assert order[1][1] == {"server_addr": "localhost:2", "grpc_options": None, "ready_timeout": 4, "test_only": True}
    assert (tmp_path / "run/flwr_home/config.toml").read_text().find('address = "localhost:3"') >= 0
    assert server._run_flower_command.call_args.kwargs["cwd"].endswith("local/custom/app" if predeployed else "custom")


@pytest.mark.parametrize("reply", [None, "invalid json", "[]", '{"success":false}'])
def test_server_command_requires_successful_json_object(monkeypatch, reply):
    server = subject.FlowerServerApplet("", 1)
    monkeypatch.setattr(subject, "run_command", MagicMock(return_value=reply))
    with pytest.raises(RuntimeError):
        server._run_flower_command("flwr list")
    monkeypatch.setattr(subject, "run_command", MagicMock(return_value='{"success":true,"runs":[]}'))
    assert server._run_flower_command("flwr list") == {"success": True, "runs": []}


@pytest.mark.parametrize(
    "info,expected",
    [
        ({}, (True, subject.TieConstant.EXIT_CODE_FATAL_ERROR)),
        ({"runs": "bad"}, (True, subject.TieConstant.EXIT_CODE_FATAL_ERROR)),
        ({"runs": [3]}, (True, subject.TieConstant.EXIT_CODE_FATAL_ERROR)),
        ({"runs": [{}]}, (True, subject.TieConstant.EXIT_CODE_FATAL_ERROR)),
        ({"runs": [{"status": 3}]}, (True, subject.TieConstant.EXIT_CODE_FATAL_ERROR)),
        ({"runs": [{"status": "running"}]}, (False, 0)),
        ({"runs": [{"status": "finished completed"}]}, (True, 0)),
        ({"runs": [{"status": "finished failed"}]}, (True, subject.TieConstant.EXIT_CODE_FAILED)),
    ],
)
def test_run_status_validates_protocol_and_distinguishes_completion_from_failure(monkeypatch, info, expected):
    monkeypatch.setattr(subject, "_validate_flower_executable", MagicMock())
    server = subject.FlowerServerApplet("", 1)
    server._run_flower_command = MagicMock(return_value=info)
    assert server._query_for_run_status() == expected
    if (
        info.get("runs")
        and isinstance(info["runs"], list)
        and isinstance(info["runs"][0], dict)
        and str(info["runs"][0].get("status", "")).startswith("finished")
    ):
        assert server.flower_run_finished and server.flower_run_rc == expected[1]


def test_status_query_caches_recent_results_and_process_exit_is_authoritative(monkeypatch):
    server = subject.FlowerServerApplet("", 1, superlink_min_query_interval=10)
    now = [100]
    monkeypatch.setattr(subject, "time", SimpleNamespace(time=lambda: now[0]))
    server._query_for_run_status = MagicMock(side_effect=[(False, 0), (True, 2)])
    process = MagicMock()
    process.poll.return_value = None
    server._superlink_process_mgr = process
    assert server.is_stopped() == (False, 0)
    now[0] += 1
    assert server.is_stopped() == (False, 0)
    assert server._query_for_run_status.call_count == 1
    now[0] += 11
    assert server.is_stopped() == (True, 2)
    assert server.is_stopped() == (True, 2)
    assert server._query_for_run_status.call_count == 2
    process.poll.return_value = 5
    assert server.is_stopped() == (True, 5)
    assert server._superlink_process_mgr is None


def test_stop_sends_cli_stop_only_once_and_always_releases_superlink(monkeypatch):
    monkeypatch.setattr(subject, "_validate_flower_executable", MagicMock())
    sleep = MagicMock()
    monkeypatch.setattr(subject, "time", SimpleNamespace(sleep=sleep))
    server = subject.FlowerServerApplet("", 1)
    server.run_id = "run-id"
    process = MagicMock()
    server._superlink_process_mgr = process
    server._run_flower_command = MagicMock(side_effect=RuntimeError("already ended"))
    assert server.stop() == server.stop() == 0
    process.stop.assert_called_once_with()
    assert server._run_flower_command.call_count == 1
    assert shlex.split(server._run_flower_command.call_args.args[0])[1:4] == ["stop", "run-id", "nvflare"]
    sleep.assert_called_once_with(1.0)
    assert server._superlink_process_mgr is None and server.flwr_stop_called


def test_start_boundary_failure_is_recorded_as_cant_start(monkeypatch, context):
    server = subject.FlowerServerApplet("", 1)
    monkeypatch.setattr(subject, "start_process", MagicMock(side_effect=OSError("missing executable")))
    assert server._start_process("superlink", subject.CommandDescriptor("flwr"), context) is None
    assert server.is_stopped() == (True, subject.TieConstant.EXIT_CODE_CANT_START)
    with pytest.raises(RuntimeError, match="invalid FLContext"):
        server.start({})
    with pytest.raises(RuntimeError, match="invalid FLContext"):
        subject.FlowerClientApplet().get_command({})
    context.get_engine().get_workspace.return_value = object()
    with pytest.raises(RuntimeError, match="invalid workspace"):
        server.start({Constant.APP_CTX_FL_CONTEXT: context})
