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
from nvflare.fuel.hci.client.api_spec import AdminConfigKey
from nvflare.fuel.hci.client.event import EventContext, EventHandler
from nvflare.fuel.hci.reg import CommandModule, CommandModuleSpec, CommandSpec


@pytest.fixture
def config():
    return {
        AdminConfigKey.CA_CERT: "root.pem",
        AdminConfigKey.CLIENT_CERT: "client.crt",
        AdminConfigKey.CLIENT_KEY: "client.key",
    }


@pytest.mark.parametrize(
    "kwargs,error",
    [
        ({"cmd_modules": "invalid"}, "must be a list"),
        ({"cmd_modules": [object()]}, "CommandModule"),
        ({"cmd_modules": [], "event_handlers": "invalid"}, "event_handlers must"),
        ({"cmd_modules": [], "event_handlers": [object()]}, "EventHandler"),
    ],
)
def test_admin_constructor_rejects_invalid_modules_and_handlers_before_transport(config, monkeypatch, kwargs, error):
    cell = Mock()
    monkeypatch.setattr(subject, "Cell", cell)
    with pytest.raises(TypeError, match=error):
        subject.AdminAPI(admin_config=config, user_name="owner", **kwargs)
    cell.assert_not_called()


@pytest.mark.parametrize(
    "missing,message",
    [
        (AdminConfigKey.CA_CERT, "CA Cert"),
        (AdminConfigKey.CLIENT_CERT, "Client Cert"),
        (AdminConfigKey.CLIENT_KEY, "Client Key"),
    ],
)
def test_admin_constructor_requires_each_credential_path_without_opening_transport(
    config, missing, message, monkeypatch
):
    config.pop(missing)
    cell = Mock()
    monkeypatch.setattr(subject, "Cell", cell)
    with pytest.raises(subject.ConfigError, match=message):
        subject.AdminAPI(admin_config=config, cmd_modules=[], user_name="owner")
    cell.assert_not_called()


@pytest.mark.parametrize("tries", [-1, subject.MAX_AUTO_LOGIN_TRIES + 1])
def test_admin_constructor_rejects_invalid_login_retry_budgets(config, tries):
    with pytest.raises(ValueError, match="out of range"):
        subject.AdminAPI(admin_config=config, cmd_modules=[], user_name="owner", auto_login_max_tries=tries)


def test_admin_constructor_requires_username_and_preserves_explicit_debug_configuration(config):
    with pytest.raises(Exception, match="user_name is required"):
        subject.AdminAPI(admin_config=config, cmd_modules=[], user_name="")
    api = subject.AdminAPI(admin_config=config, cmd_modules=[], user_name="owner", debug=True)
    assert api._debug is True
    ctx = api.new_context()
    assert ctx.get_engine() is api and ctx.get_identity_name() == "owner"
    api.close()


def test_command_modules_that_are_event_handlers_receive_api_events(config):
    class Module(CommandModule, EventHandler):
        def get_spec(self):
            return CommandModuleSpec("local", [])

        def handle_event(self, event_type, ctx):
            self.received = event_type, ctx

    module = Module()
    api = subject.AdminAPI(admin_config=config, cmd_modules=[module], user_name="owner")
    ctx = EventContext()
    api.fire_event("custom", ctx)
    assert api.event_handlers == [module] and module.received == ("custom", ctx)
    api.close()


def test_server_mapped_commands_generate_and_register_additional_local_handlers(config, monkeypatch):
    class Module(CommandModule):
        def get_spec(self):
            return CommandModuleSpec("local", [CommandSpec("local_cmd", "description", "usage", None)])

        def generate_module_spec(self, command):
            assert command.name == "remote_cmd"
            return CommandModuleSpec("dynamic", [CommandSpec("generated_cmd", "description", "usage", None)])

    api = subject.AdminAPI(admin_config=config, cmd_modules=[Module()], user_name="owner")
    monkeypatch.setattr(api, "_get_command_list_from_server", Mock(return_value=True))
    api.server_cmd_reg.mapped_cmds = [CommandSpec("remote_cmd", "description", "usage", None)]
    result = api._after_login()
    assert result[subject.ResultKey.STATUS] == subject.APIStatus.SUCCESS
    assert api.is_ready() and set(api.all_cmds) == {"local_cmd", "generated_cmd"}
    assert api.client_cmd_reg.get_command_entries("dynamic.generated_cmd")
    api.close()
