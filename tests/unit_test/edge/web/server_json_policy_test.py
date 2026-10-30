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
from flask import Flask

from nvflare.edge.web import web_server as subject
from nvflare.edge.web.models.api_error import ApiError


@pytest.fixture
def app(monkeypatch):
    app = Flask(__name__)
    app.register_error_handler(ApiError, subject.handle_api_error)
    monkeypatch.setattr(subject, "app", app)
    monkeypatch.setattr(subject.feg_bp, "_got_registered_once", subject.feg_bp._got_registered_once)
    return app


def test_startup_installs_filtered_json_and_registers_gateway_without_serving(app, monkeypatch):
    serve = Mock()
    monkeypatch.setattr(app, "run", serve)
    subject.run_server("127.0.0.1", 8101)
    serve.assert_called_once_with(host="127.0.0.1", port=8101, debug=False)
    assert app.blueprints[subject.feg_bp.name] is subject.feg_bp
    assert isinstance(app.json, subject.FilteredJSONProvider)
    with app.app_context():
        encoded = app.json.dumps({"z": 1, "missing": None, "a": {"ok": False, "gone": None}})
    assert json.loads(encoded) == {"z": 1, "a": {"ok": False}}
    assert encoded.index('"z"') < encoded.index('"a"')


def test_api_failure_becomes_structured_json_with_its_status(app):
    @app.route("/failure")
    def failure():
        raise ApiError(403, "denied", "not permitted")

    response = app.test_client().get("/failure")
    assert response.status_code == 403
    assert response.is_json
    assert response.json == {"status": "denied", "message": "not permitted", "details": None}


def test_dictionary_cleaning_preserves_false_empty_values_and_input():
    payload = {"none": None, "zero": 0, "empty": "", "false": False, "nested": {"x": None, "y": []}}
    assert subject.clean_dict(payload) == {"zero": 0, "empty": "", "false": False, "nested": {"y": []}}
    assert payload["nested"] == {"x": None, "y": []}
