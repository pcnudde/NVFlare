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

import cmd
import io
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from nvflare.fuel.hci.client import cli
from nvflare.fuel.hci.client.api import CommandInfo, ResultKey
from nvflare.fuel.hci.client.api_spec import AdminConfigKey, UidSource
from nvflare.fuel.hci.client.api_status import APIStatus
from nvflare.fuel.hci.client.cli import AdminClient
from nvflare.fuel.hci.client.event import EventContext, EventPropKey, EventType
from nvflare.fuel.hci.proto import ProtoKey
from nvflare.fuel.hci.table import Table
from tests.timing_utils import isolate_time


@pytest.fixture
def client(monkeypatch):
    clock = isolate_time(monkeypatch, cli)
    clock.time = Mock(return_value=100)
    clock.sleep = Mock()
    instance = AdminClient.__new__(AdminClient)
    cmd.Cmd.__init__(instance, stdout=io.StringIO())
    instance.api = Mock(user_name="owner", cmd_timeout=None, shutdown_received=False)
    instance.api.check_command.return_value = CommandInfo.OK
    instance.api.do_command.return_value = {ResultKey.STATUS: APIStatus.SUCCESS, ProtoKey.DETAILS: "accepted"}
    instance._study, instance.user_name = "study", "owner"
    instance.debug, instance.stopped, instance.no_stdout = False, False, False
    instance.out_file, instance.login_timeout, instance.idle_timeout = None, 3, 10
    instance.last_active_time = 95
    instance.cli_history_file = "/unused/history"
    yield instance
    instance._close_output_file()


@pytest.mark.parametrize("uid_source", [UidSource.CERT, UidSource.USER_INPUT])
def test_constructor_binds_api_history_and_event_handlers_without_registering_real_signal(
    monkeypatch, tmp_path, uid_source
):
    api = Mock(user_name="authenticated")
    factory = Mock(return_value=api)
    monkeypatch.setattr(cli, "AdminAPI", factory)
    line = Mock()
    monkeypatch.setattr(cli, "readline", line)
    signal = Mock()
    monkeypatch.setattr(cli, "signal", SimpleNamespace(signal=signal, SIGUSR1=10))
    prompt = Mock(return_value="entered")
    monkeypatch.setattr(AdminClient, "_user_input", prompt)
    config = {AdminConfigKey.UID_SOURCE: uid_source, AdminConfigKey.PROMPT: "admin>"}
    history = tmp_path / "history"
    extra_handler = object()
    instance = AdminClient(
        config,
        username="provided",
        handlers=[extra_handler],
        cli_history_dir=str(history),
        cli_history_size=17,
        study="study",
    )
    assert instance.user_name == "authenticated" and instance.prompt == "admin>"
    assert history.is_dir() and instance.cli_history_file.endswith(".admin_cli_history")
    assert factory.call_args.kwargs["user_name"] == ("provided" if uid_source == UidSource.CERT else "entered")
    assert factory.call_args.kwargs["event_handlers"] == [instance, extra_handler]
    assert factory.call_args.kwargs["study"] == "study"
    assert len(factory.call_args.kwargs["cmd_modules"][0].get_spec().cmd_specs) == 4
    signal.assert_called_once_with(10, instance.session_signal_handler)
    line.set_history_length.assert_called_once_with(17)


@pytest.mark.parametrize(
    "modules,error",
    [(None, "missing cli_history_dir"), ({"invalid": True}, "must be a list"), ([object()], "CommandModule")],
)
def test_invalid_constructor_options_fail_before_api_creation(monkeypatch, modules, error):
    factory = Mock()
    monkeypatch.setattr(cli, "AdminAPI", factory)
    with pytest.raises((Exception, TypeError), match=error):
        AdminClient({}, cmd_modules=modules, cli_history_dir="" if modules is None else "/unused")
    factory.assert_not_called()


@pytest.mark.parametrize(
    "command_info,answer,accepted",
    [
        (CommandInfo.UNKNOWN, "", False),
        (CommandInfo.AMBIGUOUS, "", False),
        (CommandInfo.CONFIRM_YN, "no", False),
        (CommandInfo.CONFIRM_YN, "YES", True),
        (CommandInfo.CONFIRM_USER_NAME, "wrong", False),
        (CommandInfo.CONFIRM_USER_NAME, "owner", True),
        (CommandInfo.CONFIRM_AUTH, "owner", True),
        (CommandInfo.OK, "", True),
    ],
)
def test_confirmation_policy_precedes_command_execution(client, monkeypatch, command_info, answer, accepted):
    client.api.check_command.return_value = command_info
    monkeypatch.setattr(client, "_user_input", Mock(return_value=answer))
    client._do_default("delete_job job-1 #custom")
    assert client.api.do_command.called is accepted
    if accepted:
        client.api.do_command.assert_called_once_with("delete_job job-1", props="custom")
        assert "accepted" in client.stdout.getvalue()
    else:
        assert "accepted" not in client.stdout.getvalue()


def test_anonymous_confirmation_requires_yes_and_session_shutdown_terminates_command(client, monkeypatch):
    client.api.user_name = ""
    client.api.check_command.return_value = CommandInfo.CONFIRM_AUTH
    monkeypatch.setattr(client, "_user_input", Mock(return_value="y"))
    client.api.shutdown_received, client.api.shutdown_msg = True, "server stopped"
    assert client._do_default("shutdown") is True
    assert "server stopped" in client.stdout.getvalue()
    client.api.shutdown_received = False
    client.api.do_command.return_value = {"status": APIStatus.ERROR_INACTIVE_SESSION}
    assert client._do_default("check_status") is True


@pytest.mark.parametrize("redirect", [">", ">>"])
def test_command_output_redirection_keeps_payload_out_of_or_in_console_as_selected(client, tmp_path, redirect):
    output = tmp_path / "reply.txt"
    client._do_default(f"check_status {redirect}{output}")
    client.flush()
    assert output.read_text() == "accepted\n"
    assert ("accepted" in client.stdout.getvalue()) is (redirect == ">")
    client._close_output_file()
    assert client.out_file is None and not client.no_stdout
    client.api.check_command.assert_called_once_with("check_status")


@pytest.mark.parametrize("suffix,error", [(" >one >two", "only one output"), (" >>", "must not be empty")])
def test_invalid_redirects_do_not_execute_command(client, suffix, error):
    client._do_default("check_status" + suffix)
    assert error in client.stdout.getvalue()
    client.api.do_command.assert_not_called()


def test_unwritable_output_target_reports_error_before_command_execution(client, tmp_path):
    client._do_default(f"check_status >{tmp_path}")
    assert "cannot open file" in client.stdout.getvalue()
    client.api.do_command.assert_not_called()


@pytest.mark.parametrize("value", ["", "0", "2.5", "bad"])
def test_timeout_command_reports_current_setting_or_changes_only_valid_numbers(client, value):
    client.do_timeout(value)
    if value == "":
        assert "not set" in client.stdout.getvalue()
        client.api.set_command_timeout.assert_not_called()
    elif value == "bad":
        assert "invalid timeout" in client.stdout.getvalue()
        client.api.set_command_timeout.assert_not_called()
    else:
        client.api.set_command_timeout.assert_called_once_with(float(value))
        assert ("unset" if value == "0" else "set to 2.5") in client.stdout.getvalue()
    assert client.do_bye("") is True
    client.api.logout.assert_called_once()


def test_response_formatter_renders_all_supported_item_types_and_ignores_unknown_items(client):
    client.print_resp(
        {
            ProtoKey.DETAILS: "details",
            ProtoKey.DATA: [
                "ignored",
                {ProtoKey.TYPE: ProtoKey.STRING, ProtoKey.DATA: "text"},
                {ProtoKey.TYPE: ProtoKey.ERROR, ProtoKey.DATA: "bad"},
                {ProtoKey.TYPE: ProtoKey.DICT, ProtoKey.DATA: {"count": 3}},
                {ProtoKey.TYPE: ProtoKey.TABLE, ProtoKey.ROWS: [["Name"], ["row"]]},
            ],
        }
    )
    rendered = client.stdout.getvalue()
    assert all(value in rendered for value in ("details", "text", "Error: bad", '"count": 3', "Name", "row"))
    client.print_resp({ProtoKey.DETAILS: Table(["empty"])})
    client.print_resp({})
    assert "Response is not correct" in client.stdout.getvalue()


def test_idle_monitor_only_signals_after_deadline_with_finite_clock_and_no_real_signal(client, monkeypatch):
    clock = isolate_time(monkeypatch, cli)
    clock.time = Mock(side_effect=[100, 106])
    clock.sleep = Mock()
    kill = Mock()
    monkeypatch.setattr(cli, "os", SimpleNamespace(kill=kill, getpid=lambda: 17))
    client._monitor_user()
    clock.sleep.assert_called_once_with(1.0)
    client.api.logout.assert_called_once()
    kill.assert_called_once_with(17, cli.signal.SIGUSR1)


def test_timeout_event_and_signal_close_session_without_signalling_real_process(client, monkeypatch):
    kill = Mock()
    monkeypatch.setattr(cli, "os", SimpleNamespace(kill=kill, getpid=lambda: 17))
    context = EventContext()
    context.set_prop(EventPropKey.MSG, "expired")
    client.handle_event(EventType.SESSION_TIMEOUT, context)
    assert "expired" in client.stdout.getvalue()
    kill.assert_called_once()
    with pytest.raises(cli._SessionClosed, match="Session Closed"):
        client.session_signal_handler(None, None)
    client.api.close.assert_called_once()
    client.stopped = True
    client.session_signal_handler(None, None)
    client.api.close.assert_called_once()


@pytest.mark.parametrize(
    "login",
    [
        None,
        {ResultKey.STATUS: APIStatus.ERROR_AUTHENTICATION, ResultKey.DETAILS: "bad certificate"},
        {ResultKey.STATUS: APIStatus.SUCCESS},
    ],
)
def test_run_enters_prompt_only_after_successful_login_and_always_closes(client, monkeypatch, login):
    client.api.login.return_value = login
    threads = Mock(return_value=Mock())
    monkeypatch.setattr(cli, "threading", SimpleNamespace(Thread=threads))
    client.cmdloop = Mock()
    client.run()
    client.api.connect.assert_called_once_with(3)
    client.api.close.assert_called_once()
    assert client.stopped
    if login and login[ResultKey.STATUS] == APIStatus.SUCCESS:
        client.cmdloop.assert_called_once()
        threads.return_value.start.assert_called_once()
    else:
        client.cmdloop.assert_not_called()
        threads.assert_not_called()
        assert "Login" in client.stdout.getvalue()


def test_command_loop_consumes_queued_commands_and_stops_without_background_runtime(client, monkeypatch):
    client.use_rawinput, client.completekey = False, None
    client.intro, client.cmdqueue = "", ["first", "bye"]
    client.onecmd = Mock(side_effect=[False, True])
    monkeypatch.setattr(cli, "readline", None)
    client.cmdloop()
    assert [call.args[0] for call in client.onecmd.call_args_list] == ["first", "bye"]


def test_completion_and_output_file_ownership_are_explicit(client):
    client.api.all_cmds = ["show", "shutdown", "submit"]
    assert [client.complete("sh", i) for i in range(3)] == ["show ", "shutdown ", None]
    previous = io.StringIO()
    client.out_file = previous
    replacement = io.StringIO()
    client._set_output_file(replacement, True)
    assert previous.closed and client.out_file is replacement and client.no_stdout
    client.write_dict({"value": 3})
    client.flush()
    assert '"value": 3' in replacement.getvalue()
