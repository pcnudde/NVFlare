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
from unittest.mock import Mock

import pytest

from nvflare.fuel.hci.client import cli as subject
from nvflare.fuel.hci.client.event import EventContext, EventPropKey, EventType
from nvflare.fuel.hci.table import Table
from tests.timing_utils import isolate_time


@pytest.fixture
def client():
    value = object.__new__(subject.AdminClient)
    value.api, value.debug, value.stopped = Mock(), True, False
    value.login_timeout, value.no_stdout, value.out_file = 1, False, None
    value.stdout = io.StringIO()
    return value


@pytest.mark.parametrize("failure", ["closed", "transport"])
def test_client_run_always_closes_api_after_connection_exception(client, capsys, failure):
    client.api.connect.side_effect = subject._SessionClosed("closed") if failure == "closed" else OSError("offline")
    client.run()
    assert client.stopped
    client.api.close.assert_called_once_with()
    output = capsys.readouterr().out
    assert ("Session closed" if failure == "closed" else "Exception OSError: offline") in output
    client.api.login.assert_not_called()


def test_interactive_prompt_trims_whitespace_and_updates_idle_budget(client, monkeypatch):
    prompt = Mock(return_value="  yes  ")
    monkeypatch.setattr("builtins.input", prompt)
    clock = isolate_time(monkeypatch, subject)
    clock.time = Mock(return_value=10)
    assert client._user_input("Continue? ") == "yes"
    prompt.assert_called_once_with("Continue? ")
    assert client.last_active_time == 10


def test_session_event_and_signal_handler_report_debug_context_without_sending_signal(client, capsys):
    context = EventContext()
    context.set_prop(EventPropKey.MSG, "working")
    client.handle_event(EventType.BEFORE_EXECUTE_CMD, context)
    assert client.stdout.getvalue() == "working\n"
    with pytest.raises(subject._SessionClosed, match="Session Closed"):
        client.session_signal_handler(0, None)
    client.api.close.assert_called_once_with()
    output = capsys.readouterr().out
    assert "received session event" in output and "signal received" in output


def test_failed_default_command_closes_output_before_and_after_error(client, monkeypatch):
    close, write = Mock(), Mock()
    monkeypatch.setattr(client, "_close_output_file", close)
    monkeypatch.setattr(client, "_do_default", Mock(side_effect=ValueError("bad command")))
    monkeypatch.setattr(client, "write_stdout", write)
    monkeypatch.setattr(subject, "secure_log_traceback", Mock())
    client.default("command")
    assert close.call_count == 2
    assert "bad command" in write.call_args.args[0]


def test_table_reply_is_written_to_explicit_output_file_and_login_runtime_error_is_clear(client):
    table = Table(["site", "status"])
    table.add_row(["one", "ready"])
    client.out_file = io.StringIO()
    client.write_table(table)
    assert client.out_file.getvalue() == client.stdout.getvalue()
    assert "one" in client.out_file.getvalue() and "ready" in client.out_file.getvalue()
    assert (
        client._format_login_failure(
            {subject.ResultKey.STATUS: subject.APIStatus.ERROR_RUNTIME, subject.ResultKey.DETAILS: "maintenance"}
        )
        == "Login failed: maintenance"
    )
