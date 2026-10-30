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

from unittest.mock import Mock

import pytest

from nvflare.fuel.hci.client import api as subject
from nvflare.fuel.hci.client.api_spec import CommandContext
from nvflare.fuel.hci.client.event import EventPropKey
from nvflare.fuel.hci.proto import MetaKey, ProtoKey
from tests.timing_utils import isolate_time


@pytest.fixture
def api():
    value = object.__new__(subject.AdminAPI)
    value.user_name, value.in_logout, value._debug = "owner", False, False
    value.debug, value.fire_event, value._print_hci = Mock(), Mock(), Mock()
    value.token, value.cmd_timeout, value.cell = "token", 2, Mock()
    return value


@pytest.mark.parametrize("timeout", ["invalid", 0, -1])
def test_connection_timeout_rejection_precedes_transport(api, timeout):
    with pytest.raises(ValueError, match="timeout must be a number"):
        api.connect(timeout)
    api.cell.send_request.assert_not_called()


def test_file_waiter_tracks_transaction_context_and_initial_progress_without_waiting(monkeypatch):
    clock = isolate_time(monkeypatch, subject)
    clock.time = Mock(return_value=12)
    waiter = subject.FileWaiter("transaction")
    assert waiter.tx_id == "transaction" and waiter.last_progress_time == 12
    assert waiter.get_stream_ctx() is None and not waiter.is_set()
    context = object()
    waiter.stream_ctx = context
    waiter.set()
    assert waiter.get_stream_ctx() is context and waiter.is_set()


def test_event_custom_properties_reach_wire_metadata_without_mutating_command_properties(api):
    api.fire_event.side_effect = lambda event, context: context.set_prop(EventPropKey.CUSTOM_PROPS, {"study": "trial"})
    ctx = api._new_command_context("check_status server", ["check_status", "server"], None)
    ctx.set_command_props({"output": "json"})
    api.cell.send_request.return_value.payload = '{"data": [{"type": "string", "data": "server output"}]}'
    api._try_command(ctx)
    wire = api.cell.send_request.call_args.kwargs["request"].payload
    decoded = subject.validate_proto(wire)
    assert decoded[ProtoKey.META][MetaKey.CUSTOM_PROPS] == {"study": "trial"}
    assert decoded[ProtoKey.META][MetaKey.CMD_PROPS] == {"output": "json"}
    result = ctx.get_command_result()
    assert result[ProtoKey.DATA][0][ProtoKey.DATA] == "server output"


def test_reply_processor_exception_receives_communication_failure_fallback(api):
    ctx = api._new_command_context("check_status server", ["check_status", "server"], None)
    processor = Mock()
    processor.process_server_reply.side_effect = [ValueError("reply handler rejected result"), None]
    ctx.set_json_processor(processor)
    api.cell.send_request.return_value.payload = '{"data": [{"type": "string", "data": "server output"}]}'
    api._send_to_cell(ctx)
    assert processor.process_server_reply.call_count == 2
    fallback = processor.process_server_reply.call_args.args[0]
    assert fallback[ProtoKey.DATA][0][ProtoKey.TYPE] == ProtoKey.ERROR
    assert "Admin Server" in fallback[ProtoKey.DATA][0][ProtoKey.DATA]


def test_transport_exception_is_returned_as_protocol_error(api):
    api.cell.send_request.side_effect = OSError("offline")
    ctx = api._new_command_context("check_status server", ["check_status", "server"], None)
    api._try_command(ctx)
    result = ctx.get_command_result()
    assert result[ProtoKey.DATA][0][ProtoKey.TYPE] == ProtoKey.ERROR
    assert "offline" in result[ProtoKey.DATA][0][ProtoKey.DATA]


def test_server_execute_reuses_explicit_context_and_skips_commands_during_logout(api, monkeypatch):
    api.in_logout = True
    result = api.server_execute("check_status server")
    assert result[subject.ResultKey.STATUS] == subject.APIStatus.SUCCESS
    assert "logging out" in result[subject.ResultKey.DETAILS]
    api.in_logout = False
    context = CommandContext()
    context.set_command_result({subject.ResultKey.STATUS: subject.APIStatus.SUCCESS})
    try_command = Mock()
    monkeypatch.setattr(api, "_try_command", try_command)
    assert api.server_execute("check_status server", cmd_ctx=context) is context.get_command_result()
    assert context.get_command() == "check_status server"
    try_command.assert_called_once_with(context)
    assert api._determine_api_status({subject.ResultKey.STATUS: "explicit"}) == "explicit"


def test_login_event_handler_exception_returns_error_without_authentication_attempt(api):
    api.auto_login_max_tries = 1
    api.fire_event.side_effect = ValueError("invalid login policy")
    api._user_login = Mock()
    result = api._try_login()
    assert result[subject.ResultKey.STATUS] == subject.APIStatus.ERROR_RUNTIME
    assert "trying_login" in result[subject.ResultKey.DETAILS]
    api._user_login.assert_not_called()
