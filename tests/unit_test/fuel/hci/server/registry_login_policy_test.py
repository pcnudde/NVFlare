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

from nvflare.fuel.f3.cellnet.defs import MessageHeaderKey
from nvflare.fuel.f3.message import Message
from nvflare.fuel.hci.conn import Connection
from nvflare.fuel.hci.proto import InternalCommands, ProtoKey, validate_proto
from nvflare.fuel.hci.reg import CommandModule, CommandModuleSpec, CommandSpec
from nvflare.fuel.hci.server.builtin import BuiltInCmdModule, new_command_register_with_builtin_module
from nvflare.fuel.hci.server.constants import ConnProps
from nvflare.fuel.hci.server.login import LoginModule
from nvflare.fuel.hci.server.reg import CommandFilter, ServerCommandRegister
from nvflare.fuel.hci.server.sess import Session, SessionManager


class RecordingModule(CommandModule):
    def __init__(self, scope, handler, extra=None):
        self.scope, self.handler, self.extra = scope, handler, extra
        self.close = Mock()

    def get_spec(self):
        return CommandModuleSpec(
            self.scope,
            [
                CommandSpec("command", "description", "usage", self.handler),
                CommandSpec("_hidden", "internal", "hidden usage", self.handler, visible=False),
                CommandSpec("disabled", "disabled", "usage", self.handler, enabled=False),
            ],
            conn_props=self.extra,
        )


class RecordingFilter(CommandFilter):
    def __init__(self, events, allow=True):
        self.events, self.allow = events, allow
        self.close = Mock()

    def pre_command(self, conn, args):
        self.events.append(("pre", args))
        return self.allow

    def post_command(self, conn, args):
        self.events.append(("post", args))


def response_text(connection):
    payload = connection.close()
    if not payload:
        return ""
    return " ".join(str(item.get(ProtoKey.DATA, "")) for item in validate_proto(payload)[ProtoKey.DATA])


def test_registry_resolves_scoped_commands_runs_filters_in_order_and_closes_each_owner_once():
    events = []
    app_context = object()
    registry = ServerCommandRegister(app_context)

    def handler(conn, args):
        events.append(("handler", args))
        conn.append_success("done")

    module = RecordingModule("scope", handler, {"shared": "value"})
    filter_component = RecordingFilter(events)
    registry.register_module(module)
    registry.add_filter(filter_component)
    finalized = Mock()
    registry.finalize(finalized)
    registry.finalize(finalized)
    assert finalized.call_count == 2
    assert registry.conn_props == {"shared": "value"}
    assert registry.get_command_entries("disabled") == []
    connection = Connection()
    registry.process_command(connection, 'scope.command "argument with spaces"')
    assert [event for event, _ in events] == ["pre", "handler", "post"]
    assert connection.args == ["scope.command", "argument with spaces"]
    assert connection.app_ctx is app_context
    assert connection.get_prop(ConnProps.CMD_ENTRY).name == "command"
    assert response_text(connection) == "done"
    registry.close()
    registry.close()
    filter_component.close.assert_called_once()
    module.close.assert_called_once()


@pytest.mark.parametrize("case", ["unknown", "ambiguous", "missing-handler", "filter-denied", "handler-error"])
def test_registry_reports_resolution_or_handler_failure_and_never_runs_denied_handler(case):
    events, handler = [], Mock()
    registry = ServerCommandRegister(None)
    registry.register_module(RecordingModule("scope", None if case == "missing-handler" else handler))
    if case == "ambiguous":
        registry.register_module(RecordingModule("another", handler))
    if case == "filter-denied":
        registry.add_filter(RecordingFilter(events, allow=False))
    elif case == "handler-error":
        handler.side_effect = RuntimeError("handler failed")
    registry.finalize()
    connection = Connection()
    registry.process_command(connection, "unknown" if case == "unknown" else "command")
    text = response_text(connection)
    if case == "filter-denied":
        assert text == ""
    elif case == "ambiguous":
        assert "multiple scopes" in text
    elif case == "handler-error":
        assert "Exception Occurred" in text and "handler failed" in text
    else:
        assert "Unknown command" in text
    assert handler.called is (case == "handler-error")


def test_registration_filters_visibility_and_keeps_client_mappings_out_of_server_dispatch():
    registry = ServerCommandRegister(None)
    registry.register_module(RecordingModule("scope", Mock()), include_invisible=False)
    registry.add_command(
        "scope",
        "mapped",
        "description",
        "usage",
        None,
        None,
        True,
        None,
        client_cmd="client.transfer",
        map_client_cmd=True,
    )
    registry.finalize()
    assert registry.get_command_entries("_hidden") == []
    assert registry.get_command_entries("mapped") == []
    assert len(registry.mapped_cmds) == 1
    assert registry.mapped_cmds[0].client_cmd == "client.transfer"
    assert registry.mapped_cmds[0].scope_name == "scope"
    with pytest.raises(AssertionError, match="CommandFilter"):
        registry.add_filter(object())


def test_builtin_help_lists_registered_commands_and_explains_scoped_and_unscoped_entries():
    registry = new_command_register_with_builtin_module(None)
    registry.register_module(RecordingModule("scope", Mock()))
    registry.register_module(RecordingModule("", Mock()))
    registry.finalize()
    builtin = BuiltInCmdModule(registry)
    connection = Connection()
    builtin.handle_list_commands(connection, ["_commands"])
    rows = validate_proto(connection.close())[ProtoKey.DATA][0][ProtoKey.ROWS]
    assert rows[0][0:2] == ["SCOPE", "COMMAND"]
    assert all(not row[1].startswith("_") for row in rows[1:])
    connection = Connection()
    builtin.handle_list_commands(connection, ["_commands", "command", "_hidden", "absent"])
    text = response_text(connection)
    assert "Command: scope.command" in text
    assert "Command: command" in text
    assert "undefined command absent" in text
    assert "internal" not in text


@pytest.fixture
def login():
    manager = Mock(spec=SessionManager)
    manager.idle_timeout = 10
    return LoginModule(manager), manager


@pytest.mark.parametrize("recreate", [False, True])
def test_login_filter_restores_trusted_session_properties_for_authenticated_command(login, recreate):
    module, manager = login
    session = Session("session", "user", "org", "lead", "admin", active_study="study", cert_studies=["study"])
    session.mark_active = Mock()
    manager.get_session.return_value = None if recreate else session
    manager.recreate_session.return_value = session
    connection = Connection()
    connection.request = {ProtoKey.DATA: [{ProtoKey.TYPE: ProtoKey.TOKEN, ProtoKey.DATA: "token"}]}
    server = Mock()
    connection.set_prop(ConnProps.HCI_SERVER, server)
    connection.set_prop(ConnProps.REQUEST, Message(headers={MessageHeaderKey.ORIGIN: "trusted-admin"}))
    assert module.pre_command(connection, ["command"])
    session.mark_active.assert_called_once()
    assert connection.get_prop(ConnProps.SESSION) is session
    assert connection.get_prop(ConnProps.USER_NAME) == "user"
    assert connection.get_prop(ConnProps.USER_ORG) == "org"
    assert connection.get_prop(ConnProps.USER_ROLE) == "lead"
    assert connection.get_prop(ConnProps.ACTIVE_STUDY) == "study"
    assert connection.get_prop(ConnProps.CERT_STUDIES) == ("study",)
    assert connection.get_prop(ConnProps.TOKEN) == "token"
    if recreate:
        manager.recreate_session.assert_called_once_with("token", "trusted-admin", server.get_id_asserter.return_value)
    module.close()
    manager.shutdown.assert_called_once()


def test_login_filter_rejects_missing_or_unrecoverable_token_and_allows_only_login_control_commands(login):
    module, manager = login
    connection = Connection()
    assert module.pre_command(connection, [InternalCommands.CERT_LOGIN])
    assert module.pre_command(connection, [InternalCommands.CHECK_SESSION])
    assert not module.pre_command(connection, ["command"])
    assert "no token" in response_text(connection)
    connection.request = {ProtoKey.DATA: [{ProtoKey.TYPE: ProtoKey.TOKEN, ProtoKey.DATA: "token"}]}
    connection.set_prop(ConnProps.HCI_SERVER, Mock())
    connection.set_prop(ConnProps.REQUEST, Message(headers={MessageHeaderKey.ORIGIN: "admin"}))
    manager.get_session.return_value = None
    manager.recreate_session.side_effect = ValueError("invalid token")
    assert not module.pre_command(connection, ["command"])
    assert "session_inactive" in response_text(connection)
    assert connection.get_prop(ConnProps.SESSION) is None
    assert module.get_spec().name == "login"
    with pytest.raises(TypeError, match="SessionManager"):
        LoginModule(object())
