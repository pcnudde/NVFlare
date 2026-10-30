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

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from nvflare.apis.fl_context import FLContext
from nvflare.apis.server_engine_spec import ServerEngineSpec
from nvflare.fuel.hci.server.authz import PreAuthzReturnCode
from nvflare.fuel.hci.server.constants import ConnProps
from nvflare.private.fed.server import cmd_utils as subject


@pytest.fixture
def env(monkeypatch):
    util = subject.CommandUtil()
    props = {}
    conn = Mock()
    conn.get_prop.side_effect = lambda key, default=None: props.get(key, default)
    conn.set_prop.side_effect = lambda key, value: props.__setitem__(key, value)
    engine = Mock(spec=ServerEngineSpec)
    conn.app_ctx = engine
    monkeypatch.setattr(subject, "StudyRegistryService", SimpleNamespace(get_registry=lambda: None))
    return SimpleNamespace(util=util, conn=conn, engine=engine, props=props)


@pytest.mark.parametrize(
    "args,error", [([], "missing target"), (["bad"], "unknown target"), (["client"], subject.ReplyKeyword.NO_CLIENTS)]
)
def test_command_target_validation_reports_missing_unknown_and_empty_client_sets(env, args, error):
    env.engine.get_clients.return_value = []
    assert error in env.util.validate_command_targets(env.conn, args)


def test_command_target_validation_rejects_admin_identities_even_if_resolved(env, monkeypatch):
    monkeypatch.setattr(subject, "is_valid_admin_client_name", lambda name: True)
    env.engine.validate_targets.return_value = ([SimpleNamespace(token="token", name="admin")], [])
    assert "admin" in env.util.validate_command_targets(env.conn, ["client", "admin"])
    assert env.util.TARGET_CLIENT_TOKENS not in env.props


def test_target_validation_requires_server_engine_protocol(env):
    env.conn.app_ctx = object()
    with pytest.raises(TypeError, match="engine must be ServerEngineSpec"):
        env.util.validate_command_targets(env.conn, ["client"])


def test_authorization_failure_short_circuits_command_and_client_operation(env, monkeypatch):
    monkeypatch.setattr(env.util, "_apply_study_role_for_authz", Mock(return_value=False))
    assert env.util.command_authz_required(env.conn, ["command"]) == PreAuthzReturnCode.ERROR
    env.engine.get_clients.return_value = [SimpleNamespace(name="site-a", token="token")]
    assert env.util.authorize_client_operation(env.conn, ["command"]) == PreAuthzReturnCode.ERROR
    env.engine.validate_targets.return_value = ([], ["unknown"])
    assert env.util.authorize_client_operation(env.conn, ["command", "unknown"]) == PreAuthzReturnCode.ERROR
    env.conn.append_error.assert_called_once()


def test_client_request_fanout_uses_connection_timeout_then_admin_default(env):
    admin = Mock()
    admin.timeout = 13
    admin.sai.new_context.side_effect = lambda: nullcontext(FLContext())
    env.props.update(
        {env.util.TARGET_CLIENT_TOKENS: ["a", "b"], ConnProps.ADMIN_SERVER: admin, ConnProps.CMD_TIMEOUT: 7}
    )
    message = Mock()
    assert env.util.send_request_to_clients(env.conn, message) is admin.send_requests.return_value
    assert admin.send_requests.call_args.args[0] == {"a": message, "b": message}
    assert admin.send_requests.call_args.kwargs == {"timeout_secs": 7}
    env.props[ConnProps.CMD_TIMEOUT] = None
    env.util.send_request_to_clients(env.conn, message)
    assert admin.send_requests.call_args.kwargs == {"timeout_secs": 13}
    env.props[env.util.TARGET_CLIENT_TOKENS] = []
    assert env.util.send_request_to_clients(env.conn, message) is None
    assert admin.send_requests.call_count == 2


def test_reply_tables_fall_back_to_token_names_and_keep_missing_replies_visible(env):
    env.props[env.util.TARGET_CLIENTS] = {"token": "site-a"}
    reply = SimpleNamespace(client_name=None, client_token="token", reply=None)
    env.util.process_replies_to_table(env.conn, [reply])
    env.conn.append_table.return_value.add_row.assert_called_once_with(["site-a", ""])
    env.util.process_replies_to_table(env.conn, [])
    env.conn.append_string.assert_called_once_with("no responses from clients")
    response = env.util._process_replies_to_string(env.conn, [SimpleNamespace(client_name="site-a", reply=None)])
    assert response == "client:site-a : No replies\n"
    assert env.conn.update_meta.call_args.args[0][subject.MetaKey.CLIENT_STATUS] == {
        "site-a": subject.MetaStatusValue.NO_REPLY
    }
