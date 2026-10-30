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
from unittest.mock import Mock

import pytest

from nvflare.edge.constants import EdgeApiStatus
from nvflare.edge.web.models.capabilities import Capabilities
from nvflare.edge.web.models.device_info import DeviceInfo
from nvflare.edge.web.models.job_request import JobRequest
from nvflare.edge.web.models.job_response import JobResponse
from nvflare.edge.web.models.result_report import ResultReport
from nvflare.edge.web.models.result_response import ResultResponse
from nvflare.edge.web.models.selection_request import SelectionRequest
from nvflare.edge.web.models.selection_response import SelectionResponse
from nvflare.edge.web.models.task_request import TaskRequest
from nvflare.edge.web.models.task_response import TaskResponse
from nvflare.edge.web.models.user_info import UserInfo
from nvflare.edge.web.service import client, query, utils
from nvflare.edge.web.service.constants import QueryType
from nvflare.edge.web.service.edge_api_pb2 import Request
from nvflare.edge.web.service.query_handler import QueryHandler


class Handler(QueryHandler):
    def __init__(self):
        self.handle = Mock()
        super().__init__()

    def handle_job_request(self, request):
        return self.handle(request)

    def handle_task_request(self, request):
        return self.handle(request)

    def handle_selection_request(self, request):
        return self.handle(request)

    def handle_result_report(self, request):
        return self.handle(request)


def cases():
    device, user = DeviceInfo("device-1"), UserInfo("user-1")
    return [
        (JobRequest("edge", device, user, Capabilities(["edge"])), utils.job_request_to_grpc_request, JobResponse),
        (TaskRequest(device, user, "job", {"round": 1}), utils.task_request_to_grpc_request, TaskResponse),
        (SelectionRequest(device, "job"), utils.selection_request_to_grpc_request, SelectionResponse),
        (
            ResultReport(device, user, "job", "task", "train", result={"w": 2}),
            utils.result_report_to_grpc_request,
            ResultResponse,
        ),
    ]


@pytest.mark.parametrize("index", range(4))
def test_query_protocol_decodes_typed_request_and_encodes_handler_result(index):
    req, convert, response_class = cases()[index]
    handler = Handler()
    handler.handle.return_value = response_class(EdgeApiStatus.OK, marker="accepted")
    reply = handler.handle_query(convert(req))
    assert reply.status == EdgeApiStatus.OK
    assert json.loads(reply.payload)["marker"] == "accepted"
    decoded = handler.handle.call_args.args[0]
    assert type(decoded) is type(req) and decoded.get_device_id() == "device-1"
    if not isinstance(req, JobRequest):
        assert decoded.job_id == "job"


@pytest.mark.parametrize("payload", [b"not-json", b"{}", b' {"job_name":"edge"} '])
def test_invalid_query_payload_is_rejected_before_dispatch(payload):
    handler = Handler()
    assert (
        handler.handle_query(Request(type=QueryType.JOB_REQUEST, payload=payload)).status
        == EdgeApiStatus.INVALID_REQUEST
    )
    handler.handle.assert_not_called()


def test_unknown_query_and_handler_exception_are_distinct_protocol_errors():
    handler = Handler()
    assert handler.handle_query(Request(type="unknown", payload=b"{}")).status == EdgeApiStatus.INVALID_REQUEST
    handler.handle.side_effect = ValueError("invalid state")
    req, convert, _ = cases()[0]
    assert handler.handle_query(convert(req)).status == EdgeApiStatus.ERROR


@pytest.mark.parametrize("index", range(4))
@pytest.mark.parametrize("outcome", ["success", "offline", "empty", "retry"])
def test_client_query_routes_by_device_and_translates_success_and_retry_responses(tmp_path, index, outcome):
    mapping = tmp_path / "mapping.json"
    mapping.write_text(json.dumps({"lcp": {"host": "edge-host", "port": 1234}}))
    transport = query.Query(str(mapping))
    transport.client = Mock()
    if outcome == "offline":
        transport.client.query.side_effect = RuntimeError("offline")
    else:
        status = EdgeApiStatus.RETRY if outcome == "retry" else EdgeApiStatus.OK
        transport.client.query.return_value = utils.make_reply(status, None if outcome == "empty" else {"marker": 7})
    req, _, response_class = cases()[index]
    response = transport(req)
    assert type(response) is response_class
    assert response.status == (EdgeApiStatus.OK if outcome == "success" else EdgeApiStatus.RETRY)
    if outcome == "success":
        assert response.marker == 7
    address, grpc_request = transport.client.query.call_args.args
    assert address == "edge-host:1234"
    assert json.loads(grpc_request.payload)["device_info"]["device_id"] == "device-1"


def test_no_mapping_and_unknown_input_do_not_call_transport():
    transport = query.Query()
    transport.client = Mock()
    assert transport(cases()[0][0]).status == EdgeApiStatus.RETRY
    assert transport(object()) is None
    transport.client.query.assert_not_called()


def test_ca_file_is_required_and_credentials_are_passed_to_transport(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="does not exist"):
        query.Query(ca_cert_file=str(tmp_path / "missing.pem"))
    cert = tmp_path / "ca.pem"
    cert.write_text("certificate bytes handled by credential boundary")
    credentials = object()
    get_credentials = Mock(return_value=credentials)
    factory = Mock()
    monkeypatch.setattr(query, "get_grpc_client_credentials", get_credentials)
    monkeypatch.setattr(query, "EdgeApiClient", factory)
    query.Query(ca_cert_file=str(cert))
    factory.assert_called_once_with(ssl_credentials=credentials)
    assert str(cert) in get_credentials.call_args.args[0].values()


@pytest.mark.parametrize("secure", [False, True])
def test_grpc_client_uses_selected_channel_and_forwards_request_without_network(monkeypatch, secure):
    channel, stub, credentials = object(), Mock(), object()
    secured, plain = Mock(return_value=channel), Mock(return_value=channel)
    monkeypatch.setattr(client.grpc, "secure_channel", secured)
    monkeypatch.setattr(client.grpc, "insecure_channel", plain)
    monkeypatch.setattr(client, "EdgeApiStub", Mock(return_value=stub))
    options = [("grpc.keepalive_time_ms", 1000)]
    api = client.EdgeApiClient(credentials if secure else None, options)
    req = Request(type=QueryType.JOB_REQUEST)
    assert api.query("host:9", req) is stub.Query.return_value
    stub.Query.assert_called_once_with(req)
    if secure:
        secured.assert_called_once_with("host:9", options=options, credentials=credentials)
        plain.assert_not_called()
    else:
        plain.assert_called_once_with("host:9", options=options)
        secured.assert_not_called()
