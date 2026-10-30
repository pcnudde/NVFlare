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
from nvflare.fuel.hci.proto import MetaKey, MetaStatusValue
from nvflare.private.admin_defs import Message, MsgHeader, ReturnCode
from nvflare.private.defs import ClientStatusKey, ScopeInfoKey
from nvflare.private.fed.server import training_cmds
from nvflare.private.fed.server.message_send import ClientReply


def _engine():
    engine = Mock(spec=training_cmds.ServerEngine)
    engine.job_runner = SimpleNamespace(running_jobs={})
    engine.get_clients.return_value = []
    engine.shutdown_server.return_value = ""
    engine.restart_server.return_value = ""
    engine.has_relays.return_value = False
    return engine


@pytest.mark.parametrize("error", [None, "cannot remove"])
def test_remove_client_cleans_only_authorized_tokens(error):
    engine = _engine()
    engine.remove_clients.return_value = error
    conn = Connection(engine, props={training_cmds.CommandUtil.TARGET_CLIENT_TOKENS: ["token"]})
    training_cmds.TrainingCommandModule().remove_client(conn, ["remove_client"])
    engine.remove_clients.assert_called_once_with(["token"])
    result = conn.close()
    assert "cannot remove" in result if error else '"success"' in result


@pytest.mark.parametrize("method", ["disable_client", "enable_client"])
def test_client_registration_commands_require_name_and_return_change_map(method):
    engine = _engine()
    operation = engine.disable_clients if method == "disable_client" else engine.enable_clients
    operation.return_value = {"changed": ["site-1", "site-2"]}
    module = training_cmds.TrainingCommandModule()
    conn = Connection(engine)
    getattr(module, method)(conn, [method])
    assert "missing client name" in conn.close()
    operation.assert_not_called()
    conn = Connection(engine)
    getattr(module, method)(conn, [method, "site-1", "site-2"])
    operation.assert_called_once_with(["site-1", "site-2"])
    assert '"changed"' in conn.close()


@pytest.mark.parametrize(
    "method,args",
    [
        ("shutdown", ["shutdown", "server"]),
        ("restart", ["restart", "server"]),
        ("remove_client", ["remove_client"]),
        ("disable_client", ["disable_client", "one"]),
        ("enable_client", ["enable_client", "one"]),
        ("check_status", ["check_status", "server"]),
        ("show_scopes", ["show_scopes", "server"]),
    ],
)
def test_admin_commands_reject_non_engine_context(method, args):
    with pytest.raises(TypeError, match="engine must be"):
        getattr(training_cmds.TrainingCommandModule(), method)(Connection(object()), args)


@pytest.mark.parametrize("target", ["server", "all", "client"])
def test_shutdown_refuses_active_jobs_before_contacting_clients(target):
    engine = _engine()
    engine.job_runner.running_jobs = {"job": SimpleNamespace(job_id="job", run_aborted=False)}
    conn = Connection(engine)
    module = training_cmds.TrainingCommandModule()
    module.send_request_to_clients = Mock()
    module.shutdown(conn, ["shutdown", target])
    assert "still jobs running" in conn.close()
    engine.shutdown_server.assert_not_called()
    module.send_request_to_clients.assert_not_called()


def test_shutdown_server_requires_clients_to_disconnect_first():
    engine = _engine()
    engine.get_clients.return_value = [Mock()]
    conn = Connection(engine)
    training_cmds.TrainingCommandModule().shutdown(conn, ["shutdown", "server"])
    assert "active clients" in conn.close()
    engine.shutdown_server.assert_not_called()


@pytest.mark.parametrize("error", ["", "backend failed"])
def test_shutdown_server_reports_engine_result(error):
    engine = _engine()
    engine.shutdown_server.return_value = error
    engine.job_runner.running_jobs = {"old": SimpleNamespace(run_aborted=True)}
    conn = Connection(engine)
    training_cmds.TrainingCommandModule().shutdown(conn, ["shutdown", "server"])
    output = conn.close()
    assert error in output if error else "Goodbye!" in output
    engine.shutdown_server.assert_called_once_with()


@pytest.mark.parametrize("client_error", [False, True])
def test_shutdown_all_stops_server_only_after_all_clients_acknowledge(monkeypatch, client_error):
    engine = _engine()
    engine.has_relays.return_value = True
    conn = Connection(engine, props={training_cmds.CommandUtil.TARGET_CLIENT_TOKENS: ["token"]})
    module = training_cmds.TrainingCommandModule()
    reply = Message("reply", "stopped")
    reply.set_header(MsgHeader.RETURN_CODE, ReturnCode.ERROR if client_error else ReturnCode.OK)
    module.send_request_to_clients = Mock(return_value=[ClientReply("token", "site-1", None, reply)])
    module.process_replies_to_table = Mock()
    monkeypatch.setattr(training_cmds, "new_message", lambda *args, **kwargs: Message(kwargs["topic"], kwargs["body"]))
    bus = Mock()
    monkeypatch.setattr(training_cmds, "DataBus", lambda: bus)
    monkeypatch.setattr(training_cmds, "time", SimpleNamespace(sleep=Mock()))
    module.shutdown(conn, ["shutdown", "all"])
    response_meta = dict(conn.buffer.meta)
    output = conn.close()
    if client_error:
        assert response_meta[MetaKey.STATUS] == MetaStatusValue.ERROR
        assert response_meta[MetaKey.INFO] == "failed to shut down all clients"
        engine.shutdown_server.assert_not_called()
        bus.publish.assert_not_called()
    else:
        assert "Goodbye!" in output
        engine.shutdown_server.assert_called_once()
        bus.publish.assert_called_once()


def test_shutdown_clients_with_no_authorized_targets_is_success(monkeypatch):
    module = training_cmds.TrainingCommandModule()
    monkeypatch.setattr(training_cmds, "new_message", lambda *args, **kwargs: Message(kwargs["topic"], kwargs["body"]))
    module.send_request_to_clients = Mock()
    assert module._shutdown_app_on_clients(Connection(_engine())) is True
    module.send_request_to_clients.assert_not_called()


@pytest.mark.parametrize("target", ["server", "all", "client"])
def test_restart_refuses_jobs_before_scheduling_any_restart(target):
    engine = _engine()
    engine.job_runner.running_jobs = {"job": object()}
    conn = Connection(engine)
    training_cmds.TrainingCommandModule().restart(conn, ["restart", target])
    assert "still jobs running" in conn.close()
    engine.restart_server.assert_not_called()


@pytest.mark.parametrize("error", ["", "restart failed"])
def test_restart_server_notifies_connected_clients_and_reports_result(error):
    engine = _engine()
    engine.get_clients.return_value = [SimpleNamespace(token="one"), SimpleNamespace(token="two")]
    engine.restart_server.return_value = error
    conn = Connection(engine)
    module = training_cmds.TrainingCommandModule()
    module._restart_clients = Mock(return_value="client restarts queued")
    module.restart(conn, ["restart", "server"])
    assert conn.get_prop(module.TARGET_CLIENT_TOKENS) == ["one", "two"]
    output = conn.close()
    assert "client restarts queued" in output
    assert error in output if error else "Server scheduled for restart" in output
    module._restart_clients.assert_called_once_with(conn)


@pytest.mark.parametrize("tokens", [None, ["one"]])
def test_restart_clients_requires_authorized_targets(tokens):
    module = training_cmds.TrainingCommandModule()
    module._restart_clients = Mock(return_value="queued")
    conn = Connection(_engine(), props={module.TARGET_CLIENT_TOKENS: tokens})
    module.restart(conn, ["restart", "client"])
    output = conn.close()
    if tokens:
        assert "queued" in output
        module._restart_clients.assert_called_once_with(conn)
    else:
        module._restart_clients.assert_not_called()
        assert "clients" in output.lower()


@pytest.mark.parametrize(
    "reply_body,code,expected",
    [
        (None, None, "No Reply"),
        ("bad", ReturnCode.ERROR, "bad"),
        (json.dumps({ClientStatusKey.RUNNING_JOBS: []}), ReturnCode.OK, "No Jobs"),
        (
            json.dumps(
                {
                    ClientStatusKey.RUNNING_JOBS: [
                        {
                            ClientStatusKey.APP_NAME: "model",
                            ClientStatusKey.JOB_ID: "job",
                            ClientStatusKey.STATUS: "running",
                        }
                    ]
                }
            ),
            ReturnCode.OK,
            "model",
        ),
        ("not-json", ReturnCode.OK, None),
        ("[]", ReturnCode.OK, None),
    ],
)
def test_client_status_reply_table_reports_missing_error_idle_and_running(reply_body, code, expected):
    reply = Message("reply", reply_body) if reply_body is not None else None
    if reply:
        reply.set_header(MsgHeader.RETURN_CODE, code)
    conn = Connection()
    module = training_cmds.TrainingCommandModule()
    module.logger = Mock()
    module._process_client_status_replies(conn, [ClientReply("token", "site-1", None, reply)])
    output = conn.close()
    if expected:
        assert expected in output
    elif reply_body == "not-json":
        module.logger.error.assert_called_once()
    else:
        assert "site-1" not in output


@pytest.mark.parametrize(
    "body,expected",
    [
        (None, "no reply"),
        (
            json.dumps({ScopeInfoKey.SCOPE_NAMES: ["private", "public"], ScopeInfoKey.DEFAULT_SCOPE: "private"}),
            "private, public",
        ),
        ("[]", "expect dict"),
        ("bad-json", "bad response"),
    ],
)
def test_scope_replies_include_defaults_and_reject_invalid_payload(body, expected):
    reply = Message("reply", body) if body is not None else None
    conn = Connection()
    table = conn.append_table(["site", "scopes", "default"])
    training_cmds.TrainingCommandModule()._process_scope_replies(
        table, conn, [ClientReply("token", "site-1", None, reply)]
    )
    assert expected in conn.close()


def test_no_status_or_scope_replies_have_visible_error():
    module = training_cmds.TrainingCommandModule()
    for process in (
        module._process_client_status_replies,
        lambda conn, replies: module._process_scope_replies(Mock(), conn, replies),
    ):
        conn = Connection()
        process(conn, [])
        assert "no responses from clients" in conn.close()


def test_show_scopes_combines_local_and_remote_scope_information(monkeypatch):
    module = training_cmds.TrainingCommandModule()
    remote = Message("reply", json.dumps({ScopeInfoKey.SCOPE_NAMES: ["remote"], ScopeInfoKey.DEFAULT_SCOPE: "remote"}))
    module.send_request_to_clients = Mock(return_value=[ClientReply("token", "site-1", None, remote)])
    monkeypatch.setattr(training_cmds, "new_message", lambda *args, **kwargs: Message(kwargs["topic"], kwargs["body"]))
    monkeypatch.setattr(training_cmds, "get_scope_info", lambda: (["local"], "local"))
    conn = Connection(_engine())
    module.show_scopes(conn, ["show_scopes", "all"])
    output = conn.close()
    assert "server" in output and "local" in output and "remote" in output and "site-1" in output
