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

from nvflare.private.admin_defs import Message
from nvflare.private.fed.server import sys_cmd as module


@pytest.fixture
def subject(monkeypatch):
    value = module.SystemCommandModule()
    monkeypatch.setattr(module, "new_message", lambda conn, topic, body, require_authz: Message(topic, body))
    return value


def test_command_spec_binds_version_and_log_commands_to_authorization_and_handlers(subject):
    spec = subject.get_spec()
    commands = {command.name: command for command in spec.cmd_specs}
    assert commands["configure_site_log"].handler_func == subject.configure_site_log
    assert commands["configure_site_log"].authz_func == subject.authorize_configure_site_log
    assert commands[module.AdminCommandNames.REPORT_VERSION].handler_func == subject.report_version


@pytest.mark.parametrize("mode", ["syntax", "engine", "target", "client"])
def test_site_logging_validates_arguments_and_routes_only_client_configuration(subject, monkeypatch, mode):
    conn = Mock()
    conn.app_ctx = object()
    monkeypatch.setattr(module, "validate_site_log_config", lambda config: {"level": "INFO"})
    subject.send_request_to_clients = Mock(return_value=[])
    subject.process_replies_to_table = Mock()
    if mode == "engine":
        with pytest.raises(TypeError, match="ServerEngine"):
            subject.configure_site_log(conn, ["configure", "server", "{}"])
    else:
        args = ["configure"] if mode == "syntax" else ["configure", "client" if mode == "client" else "invalid", "{}"]
        subject.configure_site_log(conn, args)
        if mode == "client":
            message = subject.send_request_to_clients.call_args.args[1]
            assert message.topic == module.SysCommandTopic.CONFIGURE_SITE_LOG and message.body == {"level": "INFO"}
            subject.process_replies_to_table.assert_called_once_with(conn, [])
        else:
            conn.append_error.assert_called_once()
    assert subject.send_request_to_clients.call_count == int(mode == "client")


@pytest.mark.parametrize("body,expected", [("[]", "invalid reply"), ("malformed", "Bad replies"), (None, "No replies")])
def test_version_report_keeps_malformed_and_absent_site_replies_as_explicit_errors(subject, body, expected):
    conn = Mock()
    reply = Message("topic", body) if body is not None else None
    subject.send_request_to_clients = Mock(return_value=[SimpleNamespace(client_name="site", reply=reply)])
    subject.report_version(conn, ["version", "client"])
    assert expected in conn.append_dict.call_args.args[0]["site"]["error"]
    subject.report_version(conn, ["version"])
    assert "missing site names" in conn.append_error.call_args.args[0]
