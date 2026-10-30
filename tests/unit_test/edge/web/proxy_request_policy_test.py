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
import runpy
import sys
from unittest.mock import Mock

import pytest
from flask import Flask

from nvflare.edge.constants import HttpHeaderKey
from nvflare.edge.web import routing_proxy as subject
from nvflare.edge.web.models.api_error import ApiError
from nvflare.edge.web.views import feg_views as views


@pytest.mark.parametrize("option", ["--ssl-cert", "--ssl-key"])
def test_proxy_requires_complete_tls_pair(monkeypatch, option, capsys):
    monkeypatch.setattr(sys, "argv", ["proxy", "9000", "map.json", "ca.pem", option, "value"])
    with pytest.raises(SystemExit) as raised:
        subject.parse_args()
    assert raised.value.code == 2
    assert "must be provided together" in capsys.readouterr().err


@pytest.mark.parametrize("tls", [False, True])
def test_proxy_entrypoint_configures_query_before_serving_without_starting_server(monkeypatch, tls):
    query, serve = Mock(), Mock()
    monkeypatch.setattr(views, "api_query", query)
    monkeypatch.setattr(Flask, "run", serve)
    monkeypatch.setattr(subject.logging, "basicConfig", Mock())
    argv = ["proxy", "9000", "map.json", "ca.pem"]
    if tls:
        argv += ["--ssl-cert", "server.pem", "--ssl-key", "server.key"]
    monkeypatch.setattr(sys, "argv", argv)
    runpy.run_path(subject.__file__, run_name="__main__")
    assert [call[0] for call in query.method_calls] == ["set_lcp_mapping", "set_ca_cert", "start"]
    query.set_lcp_mapping.assert_called_once_with("map.json")
    query.set_ca_cert.assert_called_once_with("ca.pem")
    serve.assert_called_once_with(
        host="0.0.0.0", port=9000, debug=False, ssl_context=("server.pem", "server.key") if tls else None
    )


@pytest.fixture
def client(monkeypatch):
    app = Flask("edge-policy")
    app.json = subject.FilteredJSONProvider(app)
    app.register_blueprint(views.feg_bp)
    app.register_error_handler(ApiError, subject.handle_api_error)
    query = Mock(return_value={"status": "OK", "nested": {"omit": None, "keep": 2}})
    monkeypatch.setattr(views, "api_query", query)
    return app.test_client(), query


@pytest.mark.parametrize(
    "path, body",
    [
        ("/task?job_id=job", {"cookie": {"version": 2}}),
        ("/selection?job_id=job", {}),
        ("/result?job_id=job&task_id=task&task_name=train", {"result": {"delta": 1}}),
    ],
)
def test_views_convert_headers_and_query_fields_to_typed_request(client, path, body):
    http, query = client
    result = http.post(
        path,
        json=body,
        headers={
            HttpHeaderKey.DEVICE_ID: "device",
            HttpHeaderKey.DEVICE_INFO: "app_name=sample",
            HttpHeaderKey.USER_INFO: "name=alice",
        },
    )
    assert result.status_code == 200
    assert result.json == {"status": "OK", "nested": {"keep": 2}}
    request = query.call_args.args[0]
    assert request.get_device_id() == "device" and request.device_info.app_name == "sample"
    assert request.user_info["name"] == "alice" and request.job_id == "job"
    if path.startswith("/result"):
        assert request.task_id == "task" and request.task_name == "train"
        assert request.result == {"delta": 1}


@pytest.mark.parametrize(
    "headers, body, message",
    [
        ({}, {}, "Device ID missing"),
        ({HttpHeaderKey.DEVICE_ID: "one", HttpHeaderKey.DEVICE_INFO: "device_id=two"}, {}, "mismatch"),
        ({HttpHeaderKey.DEVICE_ID: "one"}, {}, "missing job_id"),
    ],
)
def test_bad_request_headers_or_required_fields_return_structured_error(client, headers, body, message):
    http, query = client
    result = http.post("/task", json=body, headers=headers)
    assert result.status_code == 400 and message in result.json["message"]
    query.assert_not_called()


def test_unknown_query_response_is_a_client_error_and_none_fields_are_filtered(client):
    http, query = client
    query.return_value = None
    result = http.post("/selection?job_id=job", json={}, headers={HttpHeaderKey.DEVICE_ID: "one"})
    assert result.status_code == 400 and result.json["message"] == "unknown request type"
    assert "details" not in result.json
    assert json.loads(subject.FilteredJSONProvider(Flask("filter")).dumps({"first": 1, "empty": None})) == {"first": 1}


def test_api_query_initialization_passes_mapping_and_ca_to_query_boundary(monkeypatch):
    factory = Mock()
    monkeypatch.setattr(views, "Query", factory)
    query = views.APIQuery()
    query.set_lcp_mapping("map.json")
    query.set_ca_cert("ca.pem")
    query.start()
    factory.assert_called_once_with(lcp_mapping_file="map.json", ca_cert_file="ca.pem")
    request = object()
    query(request)
    factory.return_value.assert_called_once_with(request)
