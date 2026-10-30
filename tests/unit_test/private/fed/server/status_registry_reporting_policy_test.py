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

from nvflare.apis.client import Client
from nvflare.apis.fl_constant import MachineStatus
from nvflare.fuel.hci.server.constants import ConnProps
from nvflare.private.defs import RequestHeader, TrainingTopic
from nvflare.private.fed.server import training_cmds as module


@pytest.fixture
def status():
    engine = Mock(spec=module.ServerEngineInternalSpec)
    engine.get_engine_info.return_value = SimpleNamespace(
        status=MachineStatus.STARTED, start_time=0, app_names={"job": "app"}
    )
    client = Client("site", "token")
    client.set_fqcn("relay.site")
    client.set_fqsn("org.site")
    client.set_is_leaf(False)
    client.last_connect_time = 10
    engine.get_clients.return_value = [client]
    connection = Mock(app_ctx=engine)
    props = {ConnProps.CMD_ENTRY: SimpleNamespace(name="check_status")}
    connection.get_prop.side_effect = lambda key, default=None: props.get(key, default)
    tables = [Mock(), Mock()]
    connection.append_table.side_effect = tables
    command = module.TrainingCommandModule()
    command.send_request_to_clients = Mock(return_value=[object()])
    command._process_client_status_replies = Mock()
    return SimpleNamespace(engine=engine, client=client, conn=connection, tables=tables, command=command)


@pytest.mark.parametrize("target", ["server", "client", "all", "invalid"])
def test_status_command_reports_registered_job_and_client_identity_and_routes_remote_status(
    monkeypatch, status, target
):
    monkeypatch.setattr(module, "time", SimpleNamespace(localtime=lambda value: value, asctime=lambda value: "clock"))
    status.command.check_status(status.conn, ["check_status", target])
    local = target in ["server", "all"]
    assert status.conn.append_table.call_count == (2 if local else 0)
    if local:
        status.tables[0].add_row.assert_called_once_with(
            ["job", "app"], meta={module.MetaKey.APP_NAME: "app", module.MetaKey.JOB_ID: "job"}
        )
        row = status.tables[1].add_row.call_args
        assert row.args[0] == ["site", "relay.site", "org.site", "False", "token", "clock"]
        assert row.kwargs["meta"][module.MetaKey.CLIENT_LAST_CONNECT_TIME] == 10
    remote = target in ["client", "all"]
    assert status.command.send_request_to_clients.call_count == int(remote)
    if remote:
        message = status.command.send_request_to_clients.call_args.args[1]
        assert message.topic == TrainingTopic.CHECK_STATUS
        assert message.get_header(RequestHeader.REQUIRE_AUTHZ) == "true"
        status.command._process_client_status_replies.assert_called_once_with(
            status.conn, status.command.send_request_to_clients.return_value
        )
    assert status.conn.append_error.call_count == int(target == "invalid")


def test_status_rejects_nonclient_record_and_restart_requests_authorized_client_action(status):
    status.engine.get_clients.return_value = [object()]
    with pytest.raises(TypeError, match="must be Client"):
        status.command.check_status(status.conn, ["check_status", "server"])
    status.command._process_replies_to_string = Mock(return_value="restarted")
    assert status.command._restart_clients(status.conn) == "restarted"
    message = status.command.send_request_to_clients.call_args.args[1]
    assert message.topic == TrainingTopic.RESTART and message.get_header(RequestHeader.REQUIRE_AUTHZ) == "true"
    status.command._process_replies_to_string.assert_called_once_with(
        status.conn, status.command.send_request_to_clients.return_value
    )
    with pytest.raises(TypeError, match="ServerEngineInternalSpec"):
        status.command._restart_clients(Mock(app_ctx=object()))
