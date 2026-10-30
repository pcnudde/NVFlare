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

import io
from types import SimpleNamespace
from unittest.mock import Mock
from zipfile import ZipFile

import pytest

from nvflare.apis.fl_context import FLContext
from nvflare.fuel.f3.cellnet.defs import MessageHeaderKey, ReturnCode
from nvflare.fuel.f3.message import Message
from nvflare.private.fed.server import server_engine as module


@pytest.fixture
def engine(tmp_path):
    (tmp_path / "startup").mkdir()
    (tmp_path / "local").mkdir()
    subject = module.ServerEngine.__new__(module.ServerEngine)
    subject.logger, subject.server = Mock(), Mock()
    subject.server.admin_server.file_upload_dir = str(tmp_path)
    subject.args = SimpleNamespace(workspace=str(tmp_path))
    return subject


def test_app_staging_returns_archive_bytes_and_missing_app_error(engine, tmp_path):
    error, content = engine.get_app_data("missing")
    assert "does not exist" in error and content is None
    app = tmp_path / "app"
    app.mkdir()
    (app / "config.json").write_text('{"name":"test"}')
    error, content = engine.get_app_data("app")
    assert error == ""
    with ZipFile(io.BytesIO(content)) as archive:
        assert archive.read("config.json") == b'{"name":"test"}'
    assert engine.get_staging_path_of_app("app") == str(app)
    assert engine._get_run_folder("job").endswith("/job")
    engine.deploy_app = Mock(return_value="failed")
    assert engine.deploy_app_to_server("job", "app", str(app)) == "failed"
    engine.deploy_app.assert_called_once_with(
        "job", "app", module.WorkspaceConstants.APP_PREFIX + module.SiteType.SERVER
    )


def test_custom_import_path_cleanup_is_scoped_to_run_custom_directories(engine, monkeypatch):
    paths = ["/python/site-packages", "/workspace/run_job/app/custom", "/workspace/other/custom"]
    proxy = SimpleNamespace(path=list(paths))
    monkeypatch.setattr(module, "sys", proxy)
    engine.remove_custom_path()
    assert proxy.path == [paths[0], paths[2]]


def test_empty_aux_target_resolution_never_sends_and_missing_parent_clients_are_unavailable(engine):
    engine._to_aux_msg_targets = Mock(return_value=[])
    engine.run_manager = Mock()
    assert engine.send_aux_to_targets([], "topic", Message(), 1, FLContext(), False, False) == {}
    engine.run_manager.aux_runner.send_aux_request.assert_not_called()
    engine.server.cell.send_request.return_value = Message(
        headers={MessageHeaderKey.RETURN_CODE: ReturnCode.OK}, payload={}
    )
    assert engine._retrieve_clients_data("job") is None
    assert "failed to return clients" in engine.logger.debug.call_args.args[0]
    requests, context = {"token": Message()}, FLContext()
    engine._send_admin_requests(requests, context, timeout_secs=7)
    engine.server.admin_server.send_requests.assert_called_once_with(requests, context, timeout_secs=7)
