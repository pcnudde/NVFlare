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

import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from nvflare.fuel.hci.conn import Connection
from nvflare.fuel.hci.server.authz import PreAuthzReturnCode
from nvflare.fuel.hci.server.constants import ConnProps
from nvflare.private.admin_defs import Message, MsgHeader, ReturnCode
from nvflare.private.fed.server import info_coll_cmd, sys_cmd
from nvflare.private.fed.server.message_send import ClientReply


@pytest.fixture
def system(monkeypatch):
    module = sys_cmd.SystemCommandModule()
    monkeypatch.setattr(sys_cmd, "new_message", lambda *args, **kw: Message(kw["topic"], kw["body"]))
    vm = SimpleNamespace(total=100, available=25, _asdict=lambda: {"total": 100, "available": 25})
    monkeypatch.setattr(sys_cmd, "psutil", SimpleNamespace(virtual_memory=lambda: vm))
    module.send_request_to_clients = Mock(return_value=[])
    return module


@pytest.mark.parametrize("method", ["sys_info", "report_resources"])
def test_system_commands_require_valid_target_and_distinguish_server_policy(system, method):
    conn = Connection()
    getattr(system, method)(conn, [method])
    assert "missing site names" in conn.close()
    conn = Connection()
    getattr(system, method)(conn, [method, "invalid"])
    assert "invalid target" in conn.close()
    conn = Connection()
    getattr(system, method)(conn, [method, "server"])
    assert ("25.0" if method == "sys_info" else "unlimited") in conn.close()
    system.send_request_to_clients.assert_not_called()


@pytest.mark.parametrize(
    "body,error,expected",
    [
        (None, False, "No replies"),
        ("{bad", False, "Bad replies"),
        (json.dumps({"available": 10}), False, "available"),
        ("permission denied", True, "permission denied"),
    ],
)
def test_remote_system_metrics_render_timeout_error_and_decoded_values(system, body, error, expected):
    reply = Message("reply", body) if body is not None else None
    if reply and error:
        reply.set_header(MsgHeader.RETURN_CODE, ReturnCode.ERROR)
    conn = Connection()
    system._process_replies(conn, [ClientReply("token", "site-1", None, reply)])
    output = conn.close()
    assert "site-1" in output and expected in output


def test_remote_telemetry_reports_no_responses_and_parses_resource_mapping(system):
    for method, args in [
        ("sys_info", ["sys_info", "client"]),
        ("report_resources", ["report_resources", "client"]),
        ("report_env", ["report_env", "site-1"]),
    ]:
        conn = Connection()
        getattr(system, method)(conn, args)
        assert "no responses" in conn.close()
    reply = Message("reply", json.dumps({"gpu": 1}))
    system.send_request_to_clients.return_value = [ClientReply("token", "site-1", None, reply)]
    conn = Connection()
    system.report_resources(conn, ["report_resources", "all"])
    assert "unlimited" in conn.close()
    conn = Connection()
    system.report_env(conn, ["report_env", "site-1"])
    assert "gpu" in conn.close()


def test_resource_reply_parser_preserves_each_clients_error_and_timeout():
    error = Message("reply", "denied")
    error.set_header(MsgHeader.RETURN_CODE, ReturnCode.ERROR)
    result = sys_cmd._parse_replies(
        Connection(),
        [
            ClientReply("a", "a", None, error),
            ClientReply("b", "b", None, None),
            ClientReply("c", "c", None, Message("reply", "bad-json")),
        ],
    )
    assert result["a"] == "denied" and result["b"] == "No replies" and result["c"].startswith("Bad replies")


@pytest.mark.parametrize("failure", [None, ValueError("invalid config"), RuntimeError("cannot write config")])
def test_site_log_configuration_validates_then_applies_server_config(system, monkeypatch, failure):
    engine = Mock(spec=sys_cmd.ServerEngine)
    engine.get_workspace.return_value.get_root_dir.return_value = "/workspace"
    engine.get_workspace.return_value.get_log_config_file_path.return_value = "/workspace/log.json"
    conn = Connection(engine)
    validate, configure = Mock(return_value={"level": "INFO"}), Mock()
    if isinstance(failure, ValueError):
        validate.side_effect = failure
    elif failure:
        configure.side_effect = failure
    monkeypatch.setattr(sys_cmd, "validate_site_log_config", validate)
    monkeypatch.setattr(sys_cmd, "dynamic_log_config", configure)
    system.configure_site_log(conn, ["configure_site_log", "server", "INFO"])
    output = conn.close()
    assert (str(failure) if failure else "successfully configured") in output
    if isinstance(failure, ValueError):
        configure.assert_not_called()
    else:
        configure.assert_called_once_with(
            config={"level": "INFO"}, dir_path="/workspace", reload_path="/workspace/log.json"
        )


def test_site_log_authorization_does_not_treat_config_as_client_target(system):
    system.authorize_server_operation = Mock(return_value=PreAuthzReturnCode.REQUIRE_AUTHZ)
    conn = Connection()
    assert system.authorize_configure_site_log(conn, ["configure_site_log", "client"]) == PreAuthzReturnCode.ERROR
    assert "provide target_type and config" in conn.close()
    args = ["configure_site_log", "client", "site-1", "INFO"]
    assert system.authorize_configure_site_log(Connection(), args) == PreAuthzReturnCode.REQUIRE_AUTHZ
    system.authorize_server_operation.assert_called_once()
    assert system.authorize_server_operation.call_args.args[1] == args[:-1]


def test_dead_client_notification_requires_job_identifier(system):
    conn = Connection(Mock())
    system.dead_client(conn, ["dead", "site-1"])
    assert "Usage" in conn.close()
    system.dead_client(conn, ["dead", "site-1", "job-1"])
    conn.app_ctx.notify_dead_job.assert_called_once_with("job-1", "site-1", "AdminCommand: dead")


@pytest.mark.parametrize(
    "state,error",
    [
        ("missing_collector", "not available"),
        ("bad_collector", "not right object"),
        ("not_running", "not running"),
        ("no_run_info", "Cannot find job"),
        ("ready", None),
    ],
)
def test_job_telemetry_authorization_requires_collector_and_live_job(state, error):
    module = info_coll_cmd.InfoCollectorCommandModule()
    module.authorize_job = Mock(return_value=PreAuthzReturnCode.REQUIRE_AUTHZ)
    engine = Mock(spec=info_coll_cmd.ServerEngineInternalSpec)
    collector = Mock(spec=info_coll_cmd.InfoCollector)
    engine.get_widget.return_value = (
        None if state == "missing_collector" else object() if state == "bad_collector" else collector
    )
    engine.run_processes = {} if state == "not_running" else {"job": {}}
    engine.get_app_run_info.return_value = None if state == "no_run_info" else {"job": "running"}
    conn = Connection(engine, props={module.JOB_ID: "job"})
    result = module.authorize_info_collection(conn, ["show_stats", "job", "server"])
    if error:
        assert result == PreAuthzReturnCode.ERROR and error in conn.close()
    else:
        assert result == PreAuthzReturnCode.REQUIRE_AUTHZ
        assert conn.get_prop(module.CONN_KEY_COLLECTOR) is collector


def test_job_telemetry_authorization_preserves_upstream_denial_and_syntax_usage():
    module = info_coll_cmd.InfoCollectorCommandModule()
    conn = Connection(props={ConnProps.CMD_ENTRY: SimpleNamespace(usage="show_stats job server|client")})
    assert module.authorize_info_collection(conn, ["show_stats"]) == PreAuthzReturnCode.ERROR
    assert "Usage: show_stats" in conn.close()
    module.authorize_job = Mock(return_value=PreAuthzReturnCode.ERROR)
    assert module.authorize_info_collection(Connection(), ["show_stats", "job", "server"]) == PreAuthzReturnCode.ERROR


@pytest.mark.parametrize(
    "method,operation", [("show_stats", "show_stats"), ("show_errors", "get_errors"), ("reset_errors", "reset_errors")]
)
def test_job_telemetry_combines_server_and_client_payloads_with_job_scoping(monkeypatch, method, operation):
    module = info_coll_cmd.InfoCollectorCommandModule()
    engine = Mock()
    getattr(engine, operation).return_value = {"count": 3}
    module.send_request_to_clients = Mock(
        return_value=[ClientReply("token", "site-1", None, Message("reply", json.dumps({"count": 7})))]
    )
    monkeypatch.setattr(info_coll_cmd, "new_message", lambda *args, **kw: Message(kw["topic"], kw["body"]))
    conn = Connection(engine, props={module.JOB_ID: "job"})
    getattr(module, method)(conn, [method, "job", "all"])
    getattr(engine, operation).assert_called_once_with("job")
    request = module.send_request_to_clients.call_args.args[1]
    assert request.get_header(info_coll_cmd.RequestHeader.JOB_ID) == "job"
    output = conn.close()
    assert '"server": {"count": 3}' in output and '"site-1": {"count": 7}' in output


def test_job_stats_parser_marks_malformed_reply_instead_of_raising():
    result = {}
    info_coll_cmd.InfoCollectorCommandModule._process_stats_replies(
        Connection(), [ClientReply("token", "site-1", None, None)], result
    )
    assert result == {"site-1": "invalid_reply"}
