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
from unittest.mock import Mock

import pytest

from nvflare.fuel.hci.client import cli as subject
from nvflare.fuel.hci.reg import CommandModuleSpec, CommandRegister, CommandSpec


@pytest.fixture
def client(monkeypatch, tmp_path):
    instance = subject.AdminClient.__new__(subject.AdminClient)
    cmd.Cmd.__init__(instance, stdin=io.StringIO(), stdout=io.StringIO())
    instance.api = Mock()
    instance.debug = False
    instance.out_file = None
    instance.no_stdout = False
    instance.cli_history_file = str(tmp_path / "history")
    monkeypatch.setattr(subject, "readline", None)
    return instance


def registry(scope):
    result = CommandRegister(None)
    result.register_module_spec(
        CommandModuleSpec(
            scope,
            [
                CommandSpec("visible", "public description", "visible arg", None),
                CommandSpec("hidden", "secret description", "hidden arg", None, visible=False),
            ],
        )
    )
    result.finalize()
    return result


def test_help_lists_only_visible_commands_but_explicit_server_help_includes_hidden(client):
    client.api.client_cmd_reg = registry("")
    client.api.server_cmd_reg = registry("remote")
    client.do_help("")
    listing = client.stdout.getvalue()
    assert "Client Initiated Commands" in listing and "Server Commands" in listing
    assert "visible" in listing and "hidden" not in listing
    client.stdout = io.StringIO()
    client.do_help("visible hidden missing")
    detail = client.stdout.getvalue()
    assert "Command: visible" in detail and "Command: remote.visible" in detail
    assert "Command: remote.hidden" in detail and "Command: hidden\n" not in detail
    assert "Usage: hidden arg" in detail and "Description: secret description" in detail
    client._show_one_command("missing", client.api.server_cmd_reg)
    assert "Undefined command missing" in client.stdout.getvalue()


def test_history_read_failure_clears_stale_history_and_postcmd_preserves_stop(client, monkeypatch, tmp_path):
    (tmp_path / "history").write_text("broken")
    history = Mock()
    history.read_history_file.side_effect = OSError("broken history")
    monkeypatch.setattr(subject, "readline", history)
    client.preloop()
    history.read_history_file.assert_called_once_with(client.cli_history_file)
    history.clear_history.assert_called_once_with()
    assert "No command history loaded" in client.stdout.getvalue()
    assert client.postcmd(True, "bye") is True
    history.write_history_file.assert_called_once_with(client.cli_history_file)


@pytest.mark.parametrize("input_error", [EOFError(), ConnectionError(), KeyboardInterrupt()])
def test_raw_command_loop_translates_input_interrupt_and_restores_completer(client, monkeypatch, input_error):
    readline = pytest.importorskip("readline")

    old = object()
    monkeypatch.setattr(readline, "get_completer", Mock(return_value=old))
    set_completer = Mock()
    monkeypatch.setattr(readline, "set_completer", set_completer)
    bind = Mock()
    monkeypatch.setattr(readline, "parse_and_bind", bind)
    client.use_rawinput, client.completekey, client.intro = True, "tab", ""
    client._user_input = Mock(side_effect=[input_error])
    client.onecmd = Mock(side_effect=[True])
    client.cmdloop(intro="instructions")
    client.onecmd.assert_called_once_with("\n" if isinstance(input_error, KeyboardInterrupt) else "bye")
    assert set_completer.call_args_list[0].args == (client.complete,)
    assert set_completer.call_args_list[-1].args == (old,)
    bind.assert_called_once_with("tab: complete")
    assert "instructions" in client.stdout.getvalue()


@pytest.mark.parametrize("input_text, command", [("show\r\n", "show"), ("", "EOF")])
def test_noninteractive_loop_reads_one_line_and_stops_without_prompt_worker(client, input_text, command):
    client.use_rawinput, client.completekey, client.intro = False, None, ""
    client.stdin = io.StringIO(input_text)
    client.onecmd = Mock(side_effect=[True])
    client.cmdloop()
    client.onecmd.assert_called_once_with(command)
    assert client.stdout.getvalue() == client.prompt


@pytest.mark.parametrize("error", [KeyboardInterrupt(), ValueError("bad input")])
def test_default_command_boundary_reports_failure_and_closes_owned_output(client, monkeypatch, error):
    monkeypatch.setattr(client, "_do_default", Mock(side_effect=error))
    output = io.StringIO()
    client.out_file = output
    client.default("bad")
    assert output.closed and client.out_file is None and client.no_stdout is False
    assert ("exception occurred" in client.stdout.getvalue()) is isinstance(error, ValueError)


def test_timeout_display_uses_existing_value_and_local_pwd_is_reported(client, monkeypatch):
    client.api.cmd_timeout = 4.5
    client.do_timeout("")
    monkeypatch.setattr(subject.os, "getcwd", lambda: "/local/work")
    client.do_lpwd("")
    assert client.stdout.getvalue() == "4.5\n/local/work\n"
    client.api.set_command_timeout.assert_not_called()
