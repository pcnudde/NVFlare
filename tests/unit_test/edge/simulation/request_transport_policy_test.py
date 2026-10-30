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
from urllib.parse import parse_qs

import pytest

from nvflare.edge.constants import HttpHeaderKey
from nvflare.edge.simulation import feg_api
from nvflare.edge.simulation import run_device_simulator as subject
from nvflare.edge.web.models.api_error import ApiError
from nvflare.edge.web.models.device_info import DeviceInfo
from nvflare.edge.web.models.job_request import JobRequest
from nvflare.edge.web.models.result_report import ResultReport
from nvflare.edge.web.models.selection_request import SelectionRequest
from nvflare.edge.web.models.task_request import TaskRequest
from nvflare.edge.web.models.user_info import UserInfo


@pytest.mark.parametrize("mapping", [None, "mapping.json"])
def test_simulator_selects_proxy_or_lcp_before_starting_fake_runtime(monkeypatch, mapping):
    parser, simulator, factory, query = Mock(), Mock(), Mock(), Mock()
    parser.get_num_devices.return_value = 2
    parser.get_num_workers.return_value = 1
    parser.get_job_name.return_value = "job"
    monkeypatch.setattr(subject, "ConfigParser", Mock(return_value=parser))
    monkeypatch.setattr(subject, "TPDeviceFactory", Mock(return_value=factory))
    create = Mock(return_value=simulator)
    monkeypatch.setattr(subject, "Simulator", create)
    monkeypatch.setattr(subject, "Query", Mock(return_value=query))
    subject.run_simulator("config.json", mapping, "ca.pem")
    create.assert_called_once_with(
        job_name="job", get_job_timeout=parser.get_job_timeout, device_factory=factory, num_devices=2, num_workers=1
    )
    if mapping:
        simulator.set_send_func.assert_called_once_with(subject._send_request_to_lcp, query=query)
    else:
        simulator.set_send_func.assert_called_once_with(subject._send_request_to_proxy, parser=parser)
    simulator.start.assert_called_once_with()


@pytest.mark.parametrize(
    "kind,method",
    [("job", "get_job"), ("task", "get_task"), ("result", "report_result"), ("selection", "get_selection")],
)
def test_proxy_transport_uses_matching_method_and_authentication_headers(monkeypatch, kind, method):
    device, user = DeviceInfo("device", app_name="app"), UserInfo(user_id="alice")
    api = feg_api.FegApi("https://example/api/", device, user)
    post = Mock(return_value=SimpleNamespace(status_code=200, json=lambda: {"status": "OK"}))
    monkeypatch.setattr(feg_api, "requests", SimpleNamespace(post=post))
    requests = {
        "job": JobRequest("name", device, user),
        "task": TaskRequest(device, user, "job", {"v": 1}),
        "result": ResultReport(device, user, "job", "task", "train", result={"w": 3}, cookie={"v": 1}),
        "selection": SelectionRequest(device, "job"),
    }
    result = getattr(api, method)(requests[kind])
    assert result.status == "OK"
    call = post.call_args
    assert call.args[0] == f"https://example/api/{kind}"
    assert call.kwargs["headers"][HttpHeaderKey.DEVICE_ID] == "device"
    assert parse_qs(call.kwargs["headers"][HttpHeaderKey.DEVICE_INFO])["app_name"] == ["app"]
    assert parse_qs(call.kwargs["headers"][HttpHeaderKey.USER_INFO])["user_id"] == ["alice"]
    if kind == "result":
        assert call.kwargs["params"] == {"job_id": "job", "task_id": "task"}
        assert call.kwargs["json"]["result"] == {"w": 3} and call.kwargs["json"]["cookie"] == {"v": 1}
    transport = Mock()
    monkeypatch.setattr(subject, "FegApi", Mock(return_value=transport))
    simulated = Mock()
    parser = Mock()
    subject._send_request_to_proxy(requests[kind], simulated, parser)
    getattr(transport, method).assert_called_once_with(requests[kind])


def test_proxy_error_keeps_http_status_and_details_and_unknown_request_is_rejected(monkeypatch):
    api = feg_api.FegApi("https://example/", DeviceInfo("device"), UserInfo())
    monkeypatch.setattr(
        feg_api,
        "requests",
        SimpleNamespace(post=Mock(return_value=SimpleNamespace(status_code=403, json=lambda: {"reason": "denied"}))),
    )
    with pytest.raises(ApiError) as raised:
        api.get_selection(SelectionRequest(DeviceInfo("device"), "job"))
    assert raised.value.status_code == 403 and raised.value.details == {"reason": "denied"}
    monkeypatch.setattr(subject, "FegApi", Mock())
    with pytest.raises(ValueError, match="unknown type"):
        subject._send_request_to_proxy(object(), Mock(), Mock())
    query = Mock(return_value="reply")
    request = object()
    assert subject._send_request_to_lcp(request, Mock(), query) == "reply"
    query.assert_called_once_with(request)
