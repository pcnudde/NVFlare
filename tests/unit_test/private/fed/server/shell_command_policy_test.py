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

from nvflare.apis.fl_constant import SiteType
from nvflare.fuel.hci.conn import Connection
from nvflare.fuel.hci.server.authz import PreAuthzReturnCode
from nvflare.fuel.hci.server.constants import ConnProps
from nvflare.private.admin_defs import Message
from nvflare.private.fed.server import shell_cmd
from nvflare.private.fed.server.message_send import ClientReply
from nvflare.private.fed.server.server_engine_internal_spec import ServerEngineInternalSpec


@pytest.mark.parametrize(
    "target,expected", [(SiteType.SERVER, PreAuthzReturnCode.REQUIRE_AUTHZ), ("site-1", PreAuthzReturnCode.OK)]
)
def test_authorization_preserves_quoted_arguments_and_requires_server_auth(target, expected):
    connection = Connection()
    command = shell_cmd._CommandExecutor("grep", None)
    result = command.authorize_command(connection, ["grep", target, "two words", "events.log"])
    assert result == expected
    assert connection.get_prop("target_site") == target
    assert shell_cmd.shlex.split(connection.get_prop("shell_cmd")) == ["grep", "two words", "events.log"]


def test_missing_target_and_validator_rejection_do_not_store_executable_command():
    connection = Connection()
    command = shell_cmd._CommandExecutor("cat", Mock(validate=Mock(return_value=("denied", None))))
    assert command.authorize_command(connection, ["cat"]) == PreAuthzReturnCode.ERROR
    assert command.authorize_command(connection, ["cat", "server", "bad"]) == PreAuthzReturnCode.ERROR
    assert connection.get_prop("shell_cmd") is None
    assert "denied" in connection.close()


@pytest.mark.parametrize(
    "files,error",
    [
        ([], "only one file"),
        (["a.log", "b.log"], "only one file"),
        (["/tmp/a.log"], "absolute path"),
        (["a/../b.log"], ".. in path"),
        (["a;cat"], "unsupported file"),
        (["secret.key"], "file"),
        (["logs/server.log"], ""),
    ],
)
def test_file_command_confines_paths_and_text_types(files, error):
    executor = shell_cmd._FileCmdExecutor("cat", None)
    result = executor.validate_shell_command(["cat"], SimpleNamespace(files=files))
    assert error in result if error else result == ""


def test_file_command_rejects_missing_file_metadata_and_non_string_file():
    executor = shell_cmd._FileCmdExecutor("cat", None)
    assert "file is required" in executor.validate_shell_command(["cat"], SimpleNamespace())
    with pytest.raises(TypeError, match="file must be str"):
        executor.validate_shell_command(["cat"], SimpleNamespace(files=[1]))
    executor = shell_cmd._FileCmdExecutor("ls", None, text_file_only=False, single_file_only=False, file_required=False)
    assert executor.validate_shell_command(["ls"], SimpleNamespace(files=[])) == ""
    assert executor.validate_shell_command(["ls"], SimpleNamespace(files="folder")) == ""


def test_no_argument_command_rejects_extra_arguments_before_authorization():
    conn = Connection()
    executor = shell_cmd._NoArgCmdExecutor("pwd")
    assert executor.authorize_command(conn, ["pwd", "server", "extra"]) == PreAuthzReturnCode.ERROR
    assert conn.get_prop("shell_cmd") is None
    assert "does not accept extra args" in conn.close()


def test_server_execution_passes_argument_vector_without_invoking_shell(monkeypatch):
    execute = Mock(return_value="contents")
    monkeypatch.setattr(shell_cmd, "execute_command_directly", execute)
    conn = Connection(props={"target_site": "server", "shell_cmd": "grep 'two words' server.log"})
    shell_cmd._CommandExecutor("grep", None).execute_command(conn, [])
    execute.assert_called_once_with(["grep", "two words", "server.log"])
    assert "contents" in conn.close()


@pytest.mark.parametrize("reply_kind", ["valid", "missing", "empty", "wrong_outer", "wrong_inner"])
def test_client_execution_checks_timeout_and_reply_types(monkeypatch, reply_kind):
    engine = Mock(spec=ServerEngineInternalSpec)
    engine.validate_targets.return_value = ([SimpleNamespace(token="token-1")], [])
    conn = Connection(engine, props={"target_site": "site-1", "shell_cmd": "pwd"})
    reply = ClientReply("token-1", "site-1", None, Message("reply", "client-output"))
    if reply_kind == "missing":
        reply = None
    elif reply_kind == "empty":
        reply.reply = None
    elif reply_kind == "wrong_outer":
        reply = "invalid"
    elif reply_kind == "wrong_inner":
        reply.reply = "invalid"
    server = Mock(timeout=3.0)
    server.send_request_to_client.return_value = reply
    conn.set_prop(ConnProps.ADMIN_SERVER, server)
    monkeypatch.setattr(shell_cmd, "new_message", lambda **kwargs: Message(kwargs["topic"], kwargs["body"]))
    if reply_kind.startswith("wrong"):
        with pytest.raises(TypeError, match="must be"):
            shell_cmd._CommandExecutor("pwd", None).execute_command(conn, [])
    else:
        shell_cmd._CommandExecutor("pwd", None).execute_command(conn, [])
        output = conn.close()
        assert ("client-output" if reply_kind == "valid" else "timed out") in output
    args, kwargs = server.send_request_to_client.call_args
    assert args[0].body == "pwd" and args[1] == "token-1" and kwargs == {"timeout_secs": 3.0}


@pytest.mark.parametrize("clients,invalid", [([], ["bad"]), ([Mock(), Mock()], [])])
def test_client_execution_rejects_unknown_or_multiple_targets(clients, invalid):
    engine = Mock(spec=ServerEngineInternalSpec)
    engine.validate_targets.return_value = (clients, invalid)
    conn = Connection(engine, props={"target_site": "site-1", "shell_cmd": "pwd"})
    shell_cmd._CommandExecutor("pwd", None).execute_command(conn, [])
    assert "invalid target" in conn.close() if invalid else "one client at a time" in conn.close()


def test_client_execution_rejects_wrong_engine_type():
    conn = Connection(object(), props={"target_site": "site-1", "shell_cmd": "pwd"})
    with pytest.raises(TypeError, match="ServerEngineInternalSpec"):
        shell_cmd._CommandExecutor("pwd", None).execute_command(conn, [])


def test_registered_file_commands_enforce_file_policy_through_public_authorizer():
    commands = {s.name: s for s in shell_cmd.ShellCommandModule().get_spec().cmd_specs}
    assert set(commands) == {"pwd", "ls", "cat", "head", "tail", "grep"}
    for name in ("cat", "head", "tail"):
        connection = Connection()
        assert commands[name].authz_func(connection, [name, "server", "../secrets.log"]) == PreAuthzReturnCode.ERROR
        assert connection.get_prop("shell_cmd") is None
