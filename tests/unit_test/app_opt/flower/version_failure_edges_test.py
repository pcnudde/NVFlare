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
from unittest.mock import Mock

import pytest

flwr = pytest.importorskip("flwr")

from nvflare.apis.fl_constant import ReservedKey  # noqa: E402
from nvflare.apis.fl_context import FLContext  # noqa: E402
from nvflare.apis.workspace import Workspace  # noqa: E402
from nvflare.app_opt.flower import applet as subject  # noqa: E402
from nvflare.app_opt.flower.defs import Constant  # noqa: E402


@pytest.fixture
def context(tmp_path):
    (tmp_path / "startup").mkdir()
    (tmp_path / "local").mkdir()
    workspace = Workspace(str(tmp_path))
    ctx = FLContext()
    ctx.set_prop(ReservedKey.IDENTITY_NAME, "site")
    ctx.set_prop(ReservedKey.RUN_NUM, "job")
    engine = Mock()
    engine.get_workspace.return_value = workspace
    engine.get_clients.return_value = [SimpleNamespace(name="site")]
    ctx.set_prop(ReservedKey.ENGINE, engine)
    return ctx


@pytest.mark.parametrize("version,expected", [("1.28.0", False), ("1.29.0", True), (None, False)])
def test_runtime_dependency_flag_requires_supported_real_flower_version(monkeypatch, version, expected):
    if version is None:
        monkeypatch.delattr(flwr, "__version__")
    else:
        monkeypatch.setattr(flwr, "__version__", version)
    assert subject._check_runtime_dependency_installation_support(Mock()) is expected


def test_flower_commands_quote_run_arguments_and_require_control_endpoint(monkeypatch):
    monkeypatch.setattr(subject, "_validate_flower_executable", Mock())
    server = subject.FlowerServerApplet(":memory:", 1)
    with pytest.raises(RuntimeError, match="control API address"):
        server._build_flower_config()
    with pytest.raises(ValueError, match="requires a run ID"):
        server._flower_command("stop")
    command = shlex.split(server._flower_command("list", "run id with spaces"))
    assert command[1:4] == ["list", "run id with spaces", subject.FLOWER_SUPERLINK_CONNECTION]
    assert shlex.split(server._flower_command("custom", "quoted value"))[1:3] == ["custom", "quoted value"]
    server._flower_command = Mock(return_value="flwr list")
    server._run_flower_command = Mock(side_effect=OSError("CLI unavailable"))
    assert server._query_for_run_status() == (True, subject.TieConstant.EXIT_CODE_FATAL_ERROR)
    assert server._is_process_stopped(None) == (True, 0)
    server._superlink_process_mgr = Mock()
    server._superlink_process_mgr.poll.return_value = None
    server.flower_run_finished, server.flower_run_rc = True, 12
    assert server.is_stopped() == (True, 12)


def test_client_command_enables_supported_runtime_installation_and_rejects_invalid_workspace(context, monkeypatch):
    monkeypatch.setattr(subject, "_validate_flower_executable", Mock())
    monkeypatch.setattr(subject, "_check_runtime_dependency_installation_support", lambda logger: True)
    client = subject.FlowerClientApplet(allow_runtime_dependency_installation=True)
    description = client.get_command(
        {
            Constant.APP_CTX_FL_CONTEXT: context,
            Constant.APP_CTX_SUPERLINK_ADDR: "127.0.0.1:9000",
            Constant.APP_CTX_CLIENTAPP_API_ADDR: "127.0.0.1:9001",
        }
    )
    assert "--allow-runtime-dependency-installation" in shlex.split(description.cmd)
    context.get_engine().get_workspace.return_value = object()
    with pytest.raises(RuntimeError, match="invalid workspace"):
        client.get_command({Constant.APP_CTX_FL_CONTEXT: context})
    broken = FLContext()
    broken.get_engine = Mock(side_effect=RuntimeError("missing engine"))
    client.log_error = Mock()
    assert client._get_node_config(broken) is None
    client.log_error.assert_called_once()


@pytest.mark.parametrize("failure", ["missing-app", "start-process", "missing-run-id"])
def test_server_start_fails_before_claiming_running_when_process_or_submission_is_unavailable(
    context, monkeypatch, failure
):
    monkeypatch.setattr(subject, "_validate_flower_executable", Mock())
    monkeypatch.setattr(subject, "create_channel", Mock())
    server = subject.FlowerServerApplet(
        ":memory:",
        1,
        flower_app_path="local/custom/missing-app" if failure == "missing-app" else None,
        allow_runtime_dependency_installation=True,
    )
    monkeypatch.setattr(subject, "_check_runtime_dependency_installation_support", lambda logger: True)
    server._start_process = Mock(return_value=None if failure == "start-process" else Mock())
    server._run_flower_command = Mock(return_value={})
    app_context = {Constant.APP_CTX_FL_CONTEXT: context, Constant.APP_CTX_EXEC_API_ADDR: "127.0.0.1:9002"}
    with pytest.raises(RuntimeError, match="does not exist|cannot start superlink|missing run-id"):
        server.start(app_context)
    assert server.run_id is None
    if failure == "missing-app":
        server._start_process.assert_not_called()
    else:
        assert "--allow-runtime-dependency-installation" in server._start_process.call_args.kwargs["cmd_desc"].cmd
