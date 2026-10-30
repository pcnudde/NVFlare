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

from unittest.mock import Mock

import pytest

from nvflare.apis.fl_constant import (
    FLContextKey,
    MachineStatus,
    ReservedKey,
    ReturnCode,
    ServerCommandKey,
    ServerCommandNames,
)
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.private.fed.server import server_commands
from nvflare.widgets.widget import WidgetID


def _context(engine=None):
    context = FLContext()
    context.set_prop(ReservedKey.ENGINE, engine)
    return context


def test_command_registry_resolves_each_registered_command_and_unknown_names():
    commands = server_commands.ServerCommands
    for processor in commands.commands:
        assert commands.get_command(processor.get_command_name()) is processor
    assert commands.get_command("missing-command") is None
    assert commands.client_request_commands_names == [ServerCommandNames.GET_TASK, ServerCommandNames.SUBMIT_UPDATE]


@pytest.mark.parametrize("run_info", [None, {"run": "active"}])
def test_run_info_requires_engine_and_uses_no_op_reply_for_absent_job(run_info):
    command = server_commands.GetRunInfoCommand()
    assert command.process(Shareable(), _context()) == server_commands.NO_OP_REPLY
    engine = Mock()
    engine.get_run_info.return_value = run_info
    assert command.process(Shareable(), _context(engine)) == (run_info or server_commands.NO_OP_REPLY)
    engine.get_run_info.assert_called_once_with()


@pytest.mark.parametrize(
    "command_type,operation,expected",
    [
        (server_commands.ShowStatsCommand, "get_run_stats", {"round": 2}),
        (server_commands.GetErrorsCommand, "get_errors", ["failed"]),
        (server_commands.GetErrorsCommand, "get_errors", None),
        (server_commands.ResetErrorsCommand, "reset_errors", None),
    ],
)
def test_telemetry_processors_select_info_widget_and_normalize_empty_errors(command_type, operation, expected):
    engine, collector = Mock(), Mock()
    engine.get_widget.return_value = collector
    getattr(collector, operation).return_value = expected
    result = command_type().process(Shareable(), _context(engine))
    engine.get_widget.assert_called_once_with(WidgetID.INFO_COLLECTOR)
    getattr(collector, operation).assert_called_once_with()
    assert result == ("No Error" if operation == "get_errors" and not expected else expected)


def test_server_state_command_replaces_state_and_dead_job_command_preserves_client_identity():
    engine = Mock()
    context = _context(engine)
    state = object()
    assert server_commands.ServerStateCommand().process(state, context) == "Success"
    assert engine.server.server_state is state
    runner = Mock()
    context.set_prop(FLContextKey.RUNNER, runner)
    request = Shareable()
    request.set_header(ServerCommandKey.FL_CLIENT, "site-1")
    request.set_header(ServerCommandKey.REASON, "lost connection")
    assert server_commands.HandleDeadJobCommand().process(request, context) == ""
    runner.handle_dead_job.assert_called_once_with("site-1", context)
    context.set_prop(FLContextKey.RUNNER, None)
    server_commands.HandleDeadJobCommand().process(request, context)
    assert runner.handle_dead_job.call_count == 1


def test_task_and_submission_state_checks_use_distinct_server_state_operations():
    engine, context = Mock(), FLContext()
    context.set_prop(ReservedKey.ENGINE, engine)
    server_commands.GetTaskCommand().get_state_check(context)
    server_commands.SubmitUpdateCommand().get_state_check(context)
    engine.server.server_state.get_task.assert_called_once_with(context)
    engine.server.server_state.submit_result.assert_called_once_with(context)


def test_abort_retains_cold_state_request_and_waits_only_until_stopped():
    context = FLContext()
    engine, runner = Mock(), Mock()
    engine.engine_info.status = MachineStatus.STOPPED
    context.set_prop(ReservedKey.ENGINE, engine)
    context.set_prop(FLContextKey.RUNNER, runner)
    data = Shareable()
    data.set_header(ServerCommandKey.TURN_TO_COLD, True)
    assert server_commands.AbortCommand().process(data, context) == "Aborted the run"
    runner.abort.assert_called_once_with(fl_ctx=context, turn_to_cold=True)


def test_app_command_registry_enforces_unique_callable_topics_and_wildcard_fallback(monkeypatch):
    commands = server_commands.ServerCommands
    monkeypatch.setattr(commands, "app_cmd_registry", {})

    def handler(topic, data, context):
        return {"echo": data}

    with pytest.raises(RuntimeError, match="expect str"):
        commands.register_app_command(17, handler)
    with pytest.raises(RuntimeError, match="not callable"):
        commands.register_app_command("topic", object())
    commands.register_app_command("topic", handler, 1, offset=2)
    with pytest.raises(RuntimeError, match="duplicate"):
        commands.register_app_command("topic", handler)
    commands.register_app_command("*", handler)
    assert commands.get_app_command("topic") == (handler, (1,), {"offset": 2})
    assert commands.get_app_command("other") == (handler, (), {})


@pytest.mark.parametrize(
    "case,expected",
    [
        ("missing", ReturnCode.BAD_REQUEST_DATA),
        ("unknown", ReturnCode.BAD_REQUEST_DATA),
        ("exception", ReturnCode.EXECUTION_EXCEPTION),
        ("invalid_result", ReturnCode.EXECUTION_EXCEPTION),
        ("valid", ReturnCode.OK),
    ],
)
def test_application_command_processor_validates_registration_and_result_contract(monkeypatch, case, expected):
    commands = server_commands.ServerCommands
    monkeypatch.setattr(commands, "app_cmd_registry", {})
    context, data = FLContext(), Shareable()
    if case != "missing":
        data[ServerCommandKey.TOPIC] = "calculate"
    data[ServerCommandKey.DATA] = {"value": 3}
    if case in {"exception", "invalid_result", "valid"}:

        def calculate(topic, request, ctx, offset):
            assert topic == "calculate" and ctx is context
            if case == "exception":
                raise ValueError("cannot calculate")
            return request["value"] + offset if case == "invalid_result" else {"value": request["value"] + offset}

        commands.register_app_command("calculate", calculate, 2)
    reply = server_commands.AppCommandProcessor().process(data, context)
    assert reply.get_return_code() == expected
    if case == "valid":
        assert reply[ServerCommandKey.DATA] == {"value": 5}
    else:
        assert reply.get_header(ServerCommandKey.REASON)


def test_job_log_processor_validates_then_passes_job_specific_paths(monkeypatch):
    engine = Mock()
    engine.get_workspace.return_value.get_run_dir.return_value = "/workspace/job"
    engine.get_workspace.return_value.get_log_config_file_path.return_value = "/workspace/log.json"
    context = _context(engine)
    context.set_prop(ReservedKey.RUN_NUM, "job")
    validate, configure = Mock(return_value={"level": "INFO"}), Mock()
    monkeypatch.setattr(server_commands, "validate_site_log_config", validate)
    monkeypatch.setattr(server_commands, "dynamic_log_config", configure)
    assert server_commands.ConfigureJobLogCommand().process("INFO", context) is None
    configure.assert_called_once_with(
        config={"level": "INFO"}, dir_path="/workspace/job", reload_path="/workspace/log.json"
    )
    validate.side_effect = ValueError("invalid config")
    assert "invalid config" in server_commands.ConfigureJobLogCommand().process("bad", context)


def test_shutdown_and_heartbeat_commands_have_no_job_mutation():
    assert server_commands.ByeCommand().process(Shareable(), FLContext()) is None
    assert server_commands.HeartbeatCommand().process(Shareable(), FLContext()) is None
