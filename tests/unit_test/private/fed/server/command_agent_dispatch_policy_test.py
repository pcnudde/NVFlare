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

from nvflare.apis.fl_constant import ReservedKey, ServerCommandKey
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import ReservedHeaderKey, Shareable
from nvflare.fuel.f3.cellnet.defs import MessageHeaderKey, ReturnCode
from nvflare.fuel.f3.cellnet.utils import new_cell_message
from nvflare.private.defs import CellChannel, CellMessageHeaderKeys
from nvflare.private.fed.client import command_agent as client_subject
from nvflare.private.fed.server import server_command_agent as server_subject


@pytest.fixture
def dispatch_env(monkeypatch):
    engine = Mock()
    ctx = FLContext()
    ctx.set_prop(ReservedKey.ENGINE, engine)
    engine.new_context.return_value = ctx
    engine.server.client_manager.clients = {"token": object()}
    engine.server.authentication_check.return_value = None
    command = Mock()
    commands = SimpleNamespace(get_command=Mock(return_value=command), client_request_commands_names=["client-task"])
    monkeypatch.setattr(server_subject, "ServerCommands", commands)
    monkeypatch.setattr(client_subject, "AdminCommands", SimpleNamespace(commands={}, get_command=commands.get_command))
    agent = server_subject.ServerCommandAgent(engine, Mock())
    client = Mock()
    client_agent = client_subject.CommandAgent(client)
    client_agent.start(ctx)
    return SimpleNamespace(
        engine=engine,
        ctx=ctx,
        command=command,
        commands=commands,
        agent=agent,
        client_agent=client_agent,
        client=client,
    )


def test_server_start_registers_command_and_aux_callbacks(dispatch_env):
    env = dispatch_env
    env.agent.start()
    calls = env.agent.cell.register_request_cb.call_args_list
    assert [(c.kwargs["channel"], c.kwargs["topic"]) for c in calls] == [
        (CellChannel.SERVER_COMMAND, "*"),
        (CellChannel.AUX_COMMUNICATION, "*"),
    ]
    assert calls[0].kwargs["cb"] == env.agent.execute_command
    assert calls[1].kwargs["cb"] == env.agent.aux_communicate


def test_server_dispatch_sets_authenticated_client_and_propagates_message_root_headers(dispatch_env):
    env = dispatch_env
    request = Shareable()
    reply = Shareable()
    reply.set_header(ReservedHeaderKey.MSG_ROOT_ID, "root-id")
    reply.set_header(ReservedHeaderKey.MSG_ROOT_TTL, 9)
    env.command.process.return_value = reply
    envelope = new_cell_message({MessageHeaderKey.TOPIC: "client-task", CellMessageHeaderKeys.TOKEN: "token"}, request)
    result = env.agent.execute_command(envelope)
    assert result.payload is reply
    assert result.get_header(MessageHeaderKey.RETURN_CODE) == ReturnCode.OK
    assert result.get_header(MessageHeaderKey.MSG_ROOT_ID) == "root-id"
    assert result.get_header(MessageHeaderKey.MSG_ROOT_TTL) == 9
    assert request.get_header(ServerCommandKey.FL_CLIENT) is env.engine.server.client_manager.clients["token"]
    env.engine.server.authentication_check.assert_called_once_with(envelope, env.command.get_state_check.return_value)
    env.command.process.assert_called_once_with(data=request, fl_ctx=env.ctx)
    assert env.agent.wait_for_callbacks(0)


@pytest.mark.parametrize(
    "failure,expected",
    [
        ("missing-token", ReturnCode.AUTHENTICATION_ERROR),
        ("invalid-token", ReturnCode.AUTHENTICATION_ERROR),
        ("state-auth", ReturnCode.AUTHENTICATION_ERROR),
        ("unknown-command", ReturnCode.INVALID_REQUEST),
        ("no-result", ReturnCode.PROCESS_EXCEPTION),
    ],
)
def test_server_dispatch_rejects_authentication_unknown_commands_and_empty_results(dispatch_env, failure, expected):
    env = dispatch_env
    headers = {MessageHeaderKey.TOPIC: "client-task", CellMessageHeaderKeys.TOKEN: "token"}
    if failure == "missing-token":
        headers.pop(CellMessageHeaderKeys.TOKEN)
    elif failure == "invalid-token":
        headers[CellMessageHeaderKeys.TOKEN] = "invalid"
    elif failure == "state-auth":
        env.engine.server.authentication_check.return_value = "not authorized"
    elif failure == "unknown-command":
        env.commands.get_command.return_value = None
    elif failure == "no-result":
        env.command.process.return_value = None
    reply = env.agent.execute_command(new_cell_message(headers, Shareable()))
    assert reply.get_header(MessageHeaderKey.RETURN_CODE) == expected
    if failure != "no-result":
        env.command.process.assert_not_called()
    assert env.agent.wait_for_callbacks(0)


@pytest.mark.parametrize("failure", ["authentication", "missing-engine", "success"])
def test_server_aux_requests_check_authentication_and_engine_readiness(dispatch_env, failure):
    env = dispatch_env
    if failure == "authentication":
        env.engine.server.authentication_check.return_value = "denied"
    if failure == "missing-engine":
        env.ctx.remove_prop(ReservedKey.ENGINE, force_removal=True)
    payload = Shareable()
    env.engine.dispatch.return_value = Shareable()
    reply = env.agent.aux_communicate(new_cell_message({MessageHeaderKey.TOPIC: "aux"}, payload))
    expected = {
        "authentication": ReturnCode.AUTHENTICATION_ERROR,
        "missing-engine": ReturnCode.SERVICE_UNAVAILABLE,
        "success": ReturnCode.OK,
    }[failure]
    assert reply.get_header(MessageHeaderKey.RETURN_CODE) == expected
    if failure == "success":
        env.engine.dispatch.assert_called_once_with(topic="aux", request=payload, fl_ctx=env.ctx)
        assert reply.payload.get_peer_context() is not None
    else:
        env.engine.dispatch.assert_not_called()


def test_closed_agents_reject_late_command_and_aux_requests(dispatch_env):
    env = dispatch_env
    for agent, aux in ((env.agent, env.agent.aux_communicate), (env.client_agent, env.client_agent.aux_communication)):
        agent.shutdown()
        assert agent.asked_to_stop
        assert agent.wait_for_callbacks(0)
        request = new_cell_message({MessageHeaderKey.TOPIC: "task"}, Shareable())
        assert agent.execute_command(request).get_header(MessageHeaderKey.RETURN_CODE) == ReturnCode.SERVICE_UNAVAILABLE
        assert aux(request).get_header(MessageHeaderKey.RETURN_CODE) == ReturnCode.SERVICE_UNAVAILABLE
    env.command.process.assert_not_called()


@pytest.mark.parametrize("result", [None, Shareable()])
def test_client_command_dispatch_returns_payload_and_tracks_callbacks(dispatch_env, result):
    env = dispatch_env
    env.command.process.return_value = result
    data = {"command": "run"}
    reply = env.client_agent.execute_command(new_cell_message({MessageHeaderKey.TOPIC: "task"}, data))
    assert reply.payload is result
    if result is not None:
        assert reply.get_header(MessageHeaderKey.RETURN_CODE) == ReturnCode.OK
    env.command.process.assert_called_once_with(data=data, fl_ctx=env.ctx)
    assert env.client_agent.wait_for_callbacks(0)


def test_client_command_unknown_dispatch_and_aux_peer_context(dispatch_env):
    env = dispatch_env
    env.commands.get_command.return_value = None
    reply = env.client_agent.execute_command(new_cell_message({MessageHeaderKey.TOPIC: "unknown"}, Shareable()))
    assert reply.get_header(MessageHeaderKey.RETURN_CODE) == ReturnCode.INVALID_REQUEST
    env.engine.dispatch.return_value = Shareable()
    data = Shareable()
    reply = env.client_agent.aux_communication(new_cell_message({MessageHeaderKey.TOPIC: "aux"}, data))
    env.engine.dispatch.assert_called_once_with(topic="aux", request=data, fl_ctx=env.ctx)
    assert reply.get_header(MessageHeaderKey.RETURN_CODE) == ReturnCode.OK
    assert reply.payload.get_peer_context() is not None
    assert env.client_agent.wait_for_callbacks(0)


def test_command_callback_exception_releases_admission(dispatch_env):
    env = dispatch_env
    env.command.process.side_effect = RuntimeError("handler failed")
    for agent in (env.agent, env.client_agent):
        with pytest.raises(RuntimeError, match="handler failed"):
            agent.execute_command(new_cell_message({MessageHeaderKey.TOPIC: "internal"}, Shareable()))
        assert agent.wait_for_callbacks(0)
