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

import io
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from nvflare.apis.fl_constant import FLContextKey, ReservedKey
from nvflare.apis.fl_context import FLContext
from nvflare.apis.workspace import Workspace
from nvflare.app_common.tie import cli_applet as cli
from nvflare.app_common.tie import process_mgr as process
from nvflare.app_common.tie import py_applet as py
from nvflare.app_common.tie.defs import Constant
from tests.timing_utils import isolate_time


class PythonApplet(py.PyApplet):
    def get_runner(self, app_ctx):
        return self.test_runner


class CommandApplet(cli.CLIApplet):
    def get_command(self, app_ctx):
        return self.test_command


def context(engine=None):
    ctx = FLContext()
    ctx.set_prop(ReservedKey.ENGINE, engine or Mock(), private=True, sticky=True)
    ctx.set_prop(ReservedKey.RUN_NUM, "job", private=True, sticky=True)
    ctx.set_prop(ReservedKey.IDENTITY_NAME, "site", private=True, sticky=True)
    return ctx


@pytest.mark.parametrize("in_process", [False, True])
@pytest.mark.parametrize("fails", [False, True])
def test_python_starter_records_completion_and_start_errors(monkeypatch, in_process, fails):
    runner = Mock(spec=py.PyRunner)
    if fails:
        runner.start.side_effect = RuntimeError("invalid runner")
    configure = Mock()
    monkeypatch.setattr(py, "configure_logging", configure)
    starter = py._PyStarter(runner, in_process, "workspace", "job")
    if fails and not in_process:
        with pytest.raises(SystemExit) as exc:
            starter.start({"data": 1})
        assert exc.value.code == Constant.EXIT_CODE_CANT_START
    else:
        starter.start({"data": 1})
    assert starter.stopped
    assert starter.started is not fails
    assert starter.exit_code == (Constant.EXIT_CODE_CANT_START if fails else 0)
    assert bool(starter.error) is fails
    assert configure.call_count == (0 if in_process else 1)
    runner.start.assert_called_once_with({"data": 1})


@pytest.mark.parametrize("in_process", [False, True])
def test_python_applet_selects_configured_launch_boundary(monkeypatch, in_process):
    app = PythonApplet(in_process)
    app.test_runner = Mock(spec=py.PyRunner)
    ctx = context()
    app_ctx = {Constant.APP_CTX_FL_CONTEXT: ctx}
    thread_start, process_start = Mock(), Mock()
    monkeypatch.setattr(app, "_start_in_thread", thread_start)
    monkeypatch.setattr(app, "_start_in_process", process_start)
    app.start(app_ctx)
    chosen = thread_start if in_process else process_start
    chosen.assert_called_once_with(app.starter, app_ctx)
    assert app.runner is app.test_runner
    assert app.starter.job_id == "job"
    assert app.starter.workspace is ctx.get_engine().get_workspace.return_value


def test_python_applet_rejects_invalid_runner_and_unstarted_control_operations():
    app = PythonApplet(True)
    app.test_runner = object()
    with pytest.raises(RuntimeError, match="PyRunner"):
        app.start({Constant.APP_CTX_FL_CONTEXT: context()})
    with pytest.raises(RuntimeError, match="PyRunner is not set"):
        app.stop()
    with pytest.raises(RuntimeError, match="PyRunner is not set"):
        app.is_stopped()


def test_thread_launch_has_explicit_error_and_completion_semantics(monkeypatch):
    app = PythonApplet(True)
    app.runner = Mock(spec=py.PyRunner)
    app.starter = SimpleNamespace(start=Mock(), started=False, error="start failed", stopped=True, exit_code=3)
    fake_thread = Mock()
    thread_factory = Mock(return_value=fake_thread)
    monkeypatch.setattr(py, "threading", SimpleNamespace(Thread=thread_factory))
    with pytest.raises(RuntimeError, match="start failed"):
        app._start_in_thread(app.starter, {"x": 2})
    fake_thread.start.assert_called_once()
    assert thread_factory.call_args.kwargs["target"] is app.starter.start
    assert thread_factory.call_args.kwargs["args"] == ({"x": 2},)
    assert app.is_stopped() == (True, 3)
    app.starter.stopped = False
    app.runner.is_stopped.return_value = (False, 0)
    assert app.is_stopped() == (False, 0)
    assert app.stop(4) == 0
    app.runner.stop.assert_called_once_with(4)


def test_process_launch_removes_unpickleable_context_and_stops_only_live_process(monkeypatch):
    class FakeProcess:
        def __init__(self, **kwargs):
            self.options = kwargs
            self.exitcode = None
            self.start = Mock()
            self.kill = Mock()

    monkeypatch.setattr(py, "multiprocessing", SimpleNamespace(Process=FakeProcess))
    app = PythonApplet(False)
    app.runner = Mock(spec=py.PyRunner)
    starter = SimpleNamespace(start=Mock())
    app_ctx = {Constant.APP_CTX_FL_CONTEXT: context(), "weights": [1]}
    app._start_in_process(starter, app_ctx)
    child = app.process
    assert app_ctx == {"weights": [1]}
    assert child.options["target"] is starter.start
    child.start.assert_called_once()
    assert app.is_stopped() == (False, 0)
    child.exitcode = 7
    assert app.is_stopped() == (True, 7)
    child.exitcode = None
    assert app.stop() == -9
    child.kill.assert_called_once()
    assert app.process is None and app.is_stopped() == (True, 0)


@pytest.mark.parametrize("value,expected", [(None, (False, 0)), (4, (True, 4))])
def test_command_applet_reports_poll_state_and_handles_graceful_stop(monkeypatch, value, expected):
    app = CommandApplet()
    app.test_command = process.CommandDescriptor("worker --job one")
    manager = Mock()
    manager.poll.return_value = value
    manager.stop.return_value = -15
    start = Mock(return_value=manager)
    monkeypatch.setattr(cli, "start_process", start)
    ctx = context()
    app.start({Constant.APP_CTX_FL_CONTEXT: ctx})
    start.assert_called_once_with(app.test_command, ctx, stop_method="kill")
    assert app.is_stopped() == expected
    clock = isolate_time(monkeypatch, cli)
    clock.time = Mock(side_effect=[0, 0.5, 1, 1.5])
    clock.sleep = Mock()
    assert app.stop(timeout=1) == (4 if value == 4 else -15)
    assert app._proc_mgr is None
    assert app.stop() == 0
    assert app.is_stopped() == (True, 0)


def test_command_applet_exposes_start_failure_and_missing_command(monkeypatch):
    app = CommandApplet()
    app.test_command = None
    with pytest.raises(RuntimeError, match="failed to get cli command"):
        app.start({})
    app.test_command = process.CommandDescriptor("worker")
    monkeypatch.setattr(cli, "start_process", Mock(side_effect=OSError("no executable")))
    app.start({Constant.APP_CTX_FL_CONTEXT: context()})
    assert app.is_stopped() == (True, Constant.EXIT_CODE_CANT_START)
    assert app.stop() == 0


@pytest.mark.parametrize(
    "kwargs,error",
    [
        ({"cmd": 7}, TypeError),
        ({"cmd": "run", "cwd": 5}, TypeError),
        ({"cmd": "run", "env": [1]}, TypeError),
        ({"cmd": "run", "log_file_name": 4}, TypeError),
        ({"cmd": "run", "stdout_msg_prefix": 4}, TypeError),
        ({"cmd": "run", "stop_method": "unknown"}, ValueError),
    ],
)
def test_command_descriptor_rejects_invalid_launch_parameters(kwargs, error):
    with pytest.raises(error):
        process.CommandDescriptor(**kwargs)


def test_process_manager_launch_uses_argv_environment_and_owned_log_file(monkeypatch, tmp_path, capsys):
    desc = process.CommandDescriptor(
        'worker --label "two words"',
        cwd=str(tmp_path),
        env={"TEST_SETTING": "yes"},
        log_file_name="run.log",
        stdout_msg_prefix="child",
    )
    manager = process.ProcessManager(desc)
    child = Mock()
    child.stdout = io.BytesIO(b"first\n\rprogress\n")
    popen = Mock(return_value=child)
    monkeypatch.setattr(process.subprocess, "Popen", popen)
    thread = Mock()
    monkeypatch.setattr(
        process,
        "threading",
        SimpleNamespace(Lock=lambda: __import__("threading").Lock(), Thread=Mock(return_value=thread)),
    )
    ctx = context()
    workspace = Mock(spec=Workspace)
    workspace.get_run_dir.return_value = str(tmp_path)
    ctx.set_prop(FLContextKey.WORKSPACE_OBJECT, workspace, private=True, sticky=True)
    manager.start(ctx)
    assert popen.call_args.args[0] == ["worker", "--label", "two words"]
    assert popen.call_args.kwargs["env"]["TEST_SETTING"] == "yes"
    assert popen.call_args.kwargs["shell"] is False
    thread.start.assert_called_once()
    manager._write_log()
    assert (tmp_path / "run.log").read_bytes() == b"first\n\rprogress\n"
    assert capsys.readouterr().out == "[child@site] first\n\rprogress\n"
    child.poll.return_value = 0
    assert manager.stop() == 0
    assert manager.log_file is None
    child.kill.assert_not_called()


@pytest.mark.parametrize("method,rc", [("kill", -9), ("terminate", -15)])
def test_process_manager_stop_selects_requested_signal(method, rc):
    manager = process.ProcessManager(process.CommandDescriptor("worker", stop_method=method))
    with pytest.raises(RuntimeError, match="no process"):
        manager.poll()
    manager.process = Mock()
    manager.process.poll.return_value = None
    assert manager.stop() == rc
    getattr(manager.process, method).assert_called_once()


def test_process_launch_rejects_bad_workspace_before_starting(monkeypatch):
    popen = Mock()
    monkeypatch.setattr(process.subprocess, "Popen", popen)
    manager = process.ProcessManager(process.CommandDescriptor("worker", log_file_name="app.log"))
    with pytest.raises(RuntimeError, match="bad FLContext"):
        manager.start(context())
    popen.assert_not_called()


def test_run_command_decodes_stdout_and_keeps_stderr_out_of_return_value(monkeypatch):
    child = SimpleNamespace(stdout=io.BytesIO(b"one\ntwo\n"), stderr=io.BytesIO(b"warning\n"))
    popen = Mock(return_value=child)
    monkeypatch.setattr(process.subprocess, "Popen", popen)
    assert (
        process.run_command(process.CommandDescriptor('worker --name "two words"', env={"CHECK": "1"})) == "one\ntwo\n"
    )
    assert popen.call_args.args[0] == ["worker", "--name", "two words"]
    assert popen.call_args.kwargs["env"]["CHECK"] == "1"
    assert popen.call_args.kwargs["shell"] is False
