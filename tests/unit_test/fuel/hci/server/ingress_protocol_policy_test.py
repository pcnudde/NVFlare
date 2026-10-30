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

import json
from unittest.mock import Mock

import pytest

from nvflare.apis.fl_context import FLContext
from nvflare.fuel.f3.message import Message
from nvflare.fuel.hci.proto import Buffer, MetaKey, MetaStatusValue, ProtoKey, make_meta, validate_proto
from nvflare.fuel.hci.server import hci as subject
from nvflare.fuel.hci.server.constants import ConnProps


@pytest.fixture
def runtime(monkeypatch):
    keeper = Mock()
    monkeypatch.setattr(subject, "CredKeeper", Mock(return_value=keeper))
    monkeypatch.setattr(subject.FileStreamer, "register_stream_processing", Mock())
    engine, registry, cell = Mock(), Mock(), Mock()
    registry.conn_props = {"registered": 1}
    server = subject.AdminServer(cell, registry, engine, extra_conn_props={"extra": 2})
    return server, registry, engine, cell, keeper


def command_request(command="list_jobs", meta=None):
    buffer = Buffer()
    buffer.append_command(command)
    if meta:
        buffer.update_meta(meta)
    return buffer.encode()


def test_admin_connection_ingress_combines_trusted_properties_request_headers_and_command_metadata(runtime):
    server, registry, engine, cell, keeper = runtime
    registry.finalize.assert_called_once()
    cell.register_request_cb.assert_called_once()
    headers = {"origin": "admin"}
    metadata = {MetaKey.CMD_TIMEOUT: 9, MetaKey.CUSTOM_PROPS: {"custom": 1}, MetaKey.CMD_PROPS: {"study": "default"}}
    ok, command, connection = server._create_conn(command_request(meta=metadata), headers)
    assert ok and command == "list_jobs"
    assert connection.get_prop(ConnProps.ENGINE) is engine
    assert connection.get_prop(ConnProps.HCI_SERVER) is server
    assert connection.get_prop("registered") == 1
    assert connection.get_prop("extra") == 2
    assert connection.get_prop(ConnProps.CMD_HEADERS) is headers
    assert connection.get_prop(ConnProps.CMD_TIMEOUT) == 9
    assert connection.get_prop(ConnProps.CUSTOM_PROPS) == {"custom": 1}
    assert connection.get_prop(ConnProps.CMD_PROPS) == {"study": "default"}
    assert server.get_id_asserter() is keeper.get_id_asserter.return_value
    assert server.get_id_verifier() is keeper.get_id_verifier.return_value


@pytest.mark.parametrize(
    "payload",
    ["invalid", None, json.dumps({ProtoKey.DATA: [{ProtoKey.TYPE: ProtoKey.STRING, ProtoKey.DATA: "not-command"}]})],
)
def test_invalid_admin_requests_never_invoke_command_registry(runtime, payload):
    server, registry, _, _, _ = runtime
    response = server._process_admin_request(Message(headers={}, payload=payload))
    registry.process_command.assert_not_called()
    assert isinstance(response, Message)
    if payload is not None:
        protocol = validate_proto(response.payload)
        assert protocol[ProtoKey.META][MetaKey.STATUS] == MetaStatusValue.INTERNAL_ERROR


def test_admin_request_and_file_upload_dispatch_use_original_connection_and_return_encoded_result(runtime, monkeypatch):
    server, registry, _, _, _ = runtime
    received = []

    def execute(connection, command):
        received.append((connection, command))
        connection.append_success("accepted", make_meta(MetaStatusValue.OK))

    registry.process_command.side_effect = execute
    request = Message(headers={"origin": "admin"}, payload=command_request("status"))
    response = server._process_admin_request(request)
    assert received[0][0].get_prop(ConnProps.REQUEST) is request
    assert received[0][1] == "status"
    assert validate_proto(response.payload)[ProtoKey.DATA][0][ProtoKey.TYPE] == ProtoKey.SUCCESS
    monkeypatch.setattr(subject.FileStreamer, "get_file_location", Mock(return_value="/fake/upload"))
    uploaded = server._process_upload({"conn_data": command_request("upload")}, FLContext())
    assert received[-1][1] == "upload"
    assert received[-1][0].get_prop(ConnProps.FILE_LOCATION) == "/fake/upload"
    assert validate_proto(uploaded)[ProtoKey.META][MetaKey.STATUS] == MetaStatusValue.OK


def test_command_registry_replacement_finalizes_new_registry_and_closes_previous_owner(runtime):
    server, old, _, _, _ = runtime
    replacement = Mock()
    server.set_command_registry(replacement)
    replacement.finalize.assert_called_once()
    old.close.assert_called_once()
    assert server.cmd_reg is replacement
    server.stop()
    replacement.close.assert_called_once()
    server.set_command_registry(None)
    assert server.cmd_reg is replacement


def test_disabled_hci_owns_no_handlers_or_credential_keeper_and_missing_registry_is_rejected():
    cell = Mock()
    server = subject.AdminServer(cell, None, None, enable_hci=False)
    assert server.get_id_asserter() is None
    assert server.get_id_verifier() is None
    cell.register_request_cb.assert_not_called()
    server.start()
    server.stop()
    with pytest.raises(ValueError, match="cmd_reg"):
        subject.AdminServer(cell, None, None)
