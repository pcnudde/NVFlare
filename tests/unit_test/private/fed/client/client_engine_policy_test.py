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

from nvflare.apis.event_type import EventType
from nvflare.apis.fl_component import FLComponent
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.apis.workspace import Workspace
from nvflare.fuel.f3.cellnet.defs import MessageHeaderKey, ReturnCode
from nvflare.fuel.f3.message import Message
from nvflare.private.defs import ClientStatusKey
from nvflare.private.fed.client import client_engine
from nvflare.private.fed.client.client_status import ClientStatus


@pytest.fixture
def engine(tmp_path):
    (tmp_path / "startup").mkdir()
    (tmp_path / "local").mkdir()
    engine = client_engine.ClientEngine.__new__(client_engine.ClientEngine)
    engine.args = SimpleNamespace(workspace=str(tmp_path))
    engine.client = Mock(client_name="site-1", components={})
    engine.client_executor, engine.aux_runner, engine.object_streamer = Mock(), Mock(), Mock()
    engine.logger, engine.fire_event = Mock(), Mock()
    engine.rank, engine.fl_components = 2, []
    engine.new_context = lambda: FLContext()
    return engine


def test_auxiliary_receiver_dispatches_topic_and_adds_peer_context_to_wire_reply(engine):
    reply = Shareable()
    engine.aux_runner.dispatch.return_value = reply
    request = Message(headers={MessageHeaderKey.TOPIC: "topic"}, payload=Shareable())
    response = engine._handle_aux_message(request)
    assert response.get_header(MessageHeaderKey.RETURN_CODE) == ReturnCode.OK
    assert response.payload is reply and reply.get_peer_context() is not None
    assert engine.aux_runner.dispatch.call_args.kwargs["topic"] == "topic"
    assert engine.aux_runner.dispatch.call_args.kwargs["request"] is request.payload
    with pytest.raises(AssertionError, match="CellMessage"):
        engine._handle_aux_message(object())


def test_auxiliary_parent_request_routes_only_to_server_and_handles_no_reply(engine):
    context, request, response = FLContext(), Shareable(), Shareable()
    engine.aux_runner.send_aux_request.return_value = {"server": response}
    assert engine.send_aux_request("topic", request, 2.0, context, optional=True, secure=True) is response
    kwargs = engine.aux_runner.send_aux_request.call_args.kwargs
    assert [(t.name, t.fqcn) for t in kwargs["targets"]] == [("server", "server")]
    assert kwargs["timeout"] == 2.0 and kwargs["optional"] is True and kwargs["secure"] is True
    engine.aux_runner.send_aux_request.return_value = {}
    assert isinstance(engine.send_aux_request("topic", request, 1.0, context), Shareable)


def test_parent_streaming_ignores_client_targets_and_preserves_callbacks(engine):
    context, producer = FLContext(), object()
    engine.stream_objects("channel", "topic", {}, ["site-2"], producer, context, True, True)
    kwargs = engine.object_streamer.stream.call_args.kwargs
    assert [(t.name, t.fqcn) for t in kwargs["targets"]] == [("server", "server")]
    assert kwargs["producer"] is producer and kwargs["secure"] and kwargs["optional"]
    done, consumed, factory = Mock(), Mock(), object()
    engine.register_stream_processing("channel", "topic", factory, done, consumed, token="retained")
    engine.object_streamer.register_stream_processing.assert_called_once_with(
        topic="topic", channel="channel", factory=factory, stream_done_cb=done, consumed_cb=consumed, token="retained"
    )
    engine.shutdown_streamer()
    engine.object_streamer.shutdown.assert_called_once_with()
    engine.object_streamer = None
    with pytest.raises(RuntimeError, match="not been created"):
        engine.stream_objects("channel", "topic", {}, [], producer, context)
    with pytest.raises(RuntimeError, match="not been created"):
        engine.register_stream_processing("channel", "topic", factory)
    engine.shutdown_streamer()


def test_component_registration_enforces_unique_string_id_and_registers_event_handlers(engine):
    component, plain = FLComponent(), object()
    with pytest.raises(TypeError, match="component id"):
        engine.add_component(17, plain)
    engine.add_component("handler", component)
    engine.add_component("plain", plain)
    assert engine.fl_components == [component]
    assert engine.get_component("plain") is plain
    with pytest.raises(ValueError, match="duplicate"):
        engine.add_component("plain", object())
    assert engine.get_component("plain") is plain


def test_engine_status_reports_app_name_and_executor_state_for_each_owned_job(engine, tmp_path):
    workspace = Workspace(str(tmp_path), site_name="site-1")
    from pathlib import Path

    run = Path(workspace.get_run_dir("job-a"))
    run.mkdir()
    (run / "fl_app.txt").write_text("my-model\nignored")
    engine.get_all_job_ids = lambda: ["job-a", "job-b"]
    engine.client_executor.check_status.side_effect = ["running", "starting"]
    result = engine.get_engine_status()
    assert result == {
        ClientStatusKey.CLIENT_NAME: "site-1",
        ClientStatusKey.RUNNING_JOBS: [
            {ClientStatusKey.APP_NAME: "my-model", ClientStatusKey.JOB_ID: "job-a", ClientStatusKey.STATUS: "running"},
            {ClientStatusKey.APP_NAME: "", ClientStatusKey.JOB_ID: "job-b", ClientStatusKey.STATUS: "starting"},
        ],
    }


def test_app_start_requires_deployment_and_passes_all_resource_reservation_fields(engine, tmp_path):
    engine.client_executor.get_status.return_value = ClientStatus.STARTED
    assert "already started" in engine.start_app("job", {})
    engine.client_executor.get_status.return_value = ClientStatus.NOT_STARTED
    assert "does not exist" in engine.start_app("job", {})
    from pathlib import Path

    Path(Workspace(str(tmp_path), site_name="site-1").get_app_dir("job")).mkdir(parents=True)
    resource_manager = object()
    assert (
        engine.start_app("job", {"metadata": 1}, {"gpu": 1}, "reservation", resource_manager)
        == "Start the client app..."
    )
    args, kwargs = engine.client_executor.start_app.call_args
    assert args == (engine.client, "job", {"metadata": 1}, engine.args, {"gpu": 1}, "reservation", resource_manager)
    assert isinstance(kwargs["fl_ctx"], FLContext)


@pytest.mark.parametrize(
    "state,error",
    [
        (ClientStatus.NOT_STARTED, "has not started"),
        (ClientStatus.STARTING, "is starting"),
        (ClientStatus.STARTED, None),
    ],
)
def test_abort_task_checks_job_state_before_sending_signal(engine, state, error):
    engine.client_executor.get_status.return_value = state
    result = engine.abort_task("job")
    if error:
        assert error in result
        engine.client_executor.abort_task.assert_not_called()
    else:
        assert "Abort signal" in result
        engine.client_executor.abort_task.assert_called_once_with("job")


@pytest.mark.parametrize("method,marker", [("shutdown", "shutdown.fl"), ("restart", "restart.fl")])
def test_lifecycle_launches_shutdown_boundary_with_distinct_marker(engine, monkeypatch, method, marker):
    thread = Mock()
    factory = Mock(return_value=thread)
    monkeypatch.setattr(client_engine, "threading", SimpleNamespace(Thread=factory))
    assert "client" in getattr(engine, method)().lower()
    assert factory.call_args.kwargs["target"] is client_engine.shutdown_client
    assert factory.call_args.kwargs["args"] == (engine.client, engine.args.workspace + "/" + marker)
    thread.start.assert_called_once_with()
    assert engine.fire_event.call_args.args[0] == EventType.SYSTEM_END
    assert engine.object_streamer.shutdown.called is (method == "shutdown")


@pytest.mark.parametrize("custom", [False, True])
@pytest.mark.parametrize("error", ["", "denied"])
def test_app_deploy_uses_custom_or_default_deployer_and_preserves_errors(engine, monkeypatch, custom, error):
    deployer = Mock()
    deployer.deploy.return_value = error
    engine.get_component = Mock(return_value=deployer if custom else None)
    monkeypatch.setattr(client_engine, "AppDeployer", Mock(return_value=deployer))
    result = engine.deploy_app("app", "job", {"metadata": 1}, "site-1", b"zip")
    assert result == (client_engine.ERROR_MSG_PREFIX + ": " + error if error else "")
    kwargs = deployer.deploy.call_args.kwargs
    assert kwargs["job_id"] == "job" and kwargs["job_meta"] == {"metadata": 1}
    assert kwargs["app_data"] == b"zip" and kwargs["app_name"] == "app"
    assert kwargs["workspace"].site_name == "site-1"


def test_run_deletion_is_idempotent_and_does_not_touch_sibling_job(engine, tmp_path):
    workspace = Workspace(str(tmp_path), site_name="site-1")
    from pathlib import Path

    run, sibling = Path(workspace.get_run_dir("job")), Path(workspace.get_run_dir("other"))
    run.mkdir()
    sibling.mkdir()
    assert "Delete run folder" in engine.delete_run("job")
    assert not run.exists() and sibling.exists()
    engine.delete_run("job")
    assert sibling.exists()


@pytest.mark.parametrize("close_error", [False, True])
def test_shutdown_boundary_stops_heartbeats_touches_marker_and_contains_close_error(
    monkeypatch, tmp_path, close_error, capsys
):
    client = Mock()
    client.close.side_effect = RuntimeError("close failed") if close_error else None
    security = Mock()
    monkeypatch.setattr(client_engine, "security_close", security)
    monkeypatch.setattr(client_engine, "secure_log_traceback", Mock())
    marker = tmp_path / "shutdown.fl"
    client_engine.shutdown_client(client, str(marker))
    assert marker.exists() and client.communicator.heartbeat_done is True
    client.close.assert_called_once_with()
    if close_error:
        assert "Failed to shutdown client" in capsys.readouterr().out
        security.assert_not_called()
    else:
        assert client.status == ClientStatus.STOPPED
        security.assert_called_once_with()


def test_custom_import_path_cleanup_removes_only_job_custom_folders(monkeypatch):
    paths = ["/python/site-packages", "/workspace/run_one/custom", "/workspace/run_two/custom", "/workspace/custom"]
    monkeypatch.setattr(client_engine.sys, "path", paths.copy())
    client_engine._remove_custom_path()
    assert client_engine.sys.path == ["/python/site-packages", "/workspace/custom"]
