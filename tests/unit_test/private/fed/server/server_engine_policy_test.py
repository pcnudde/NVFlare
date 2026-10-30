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

from nvflare.apis.fl_constant import (
    FLContextKey,
    MachineStatus,
    ReservedKey,
    ReturnCode,
    ServerCommandKey,
    ServerCommandNames,
)
from nvflare.apis.fl_context import FLContext
from nvflare.apis.job_def import Job
from nvflare.apis.shareable import Shareable
from nvflare.apis.workspace import Workspace
from nvflare.private.admin_defs import Message, MsgHeader
from nvflare.private.defs import RequestHeader
from nvflare.private.fed.server import server_engine
from nvflare.private.fed.server.message_send import ClientReply
from nvflare.private.scheduler_constants import ShareableHeader


@pytest.fixture
def engine(tmp_path):
    (tmp_path / "startup").mkdir()
    (tmp_path / "local").mkdir()
    engine = server_engine.ServerEngine.__new__(server_engine.ServerEngine)
    engine.args = SimpleNamespace(workspace=str(tmp_path), job_id="job")
    engine.logger, engine.server, engine.executor = Mock(), Mock(), Mock()
    engine.run_processes = {}
    engine.engine_info = SimpleNamespace(app_names={}, status=MachineStatus.STOPPED)
    engine.client_manager, engine.job_runner = Mock(), Mock()
    engine.run_manager = None
    return engine


def test_engine_info_reads_each_job_app_and_marks_missing_name_without_stale_entries(engine, tmp_path):
    first = tmp_path / "job-a"
    first.mkdir()
    (first / "fl_app.txt").write_text("model-a\nignored")
    engine._get_run_folder = lambda job: str(tmp_path / job)
    engine.run_processes = {"job-a": {}, "job-b": {}}
    engine.engine_info.app_names = {"stale": "remove"}
    info = engine.get_engine_info()
    assert info.app_names == {"job-a": "model-a", "job-b": "?"}
    assert info.status == MachineStatus.STARTED
    engine.run_processes = {}
    assert engine.get_engine_info().status == MachineStatus.STOPPED
    assert info.app_names == {}


def test_app_start_requires_real_job_deployed_app_and_one_process_per_job(engine):
    context = FLContext()
    assert "job object" in engine.start_app_on_server(context, object())
    job = Mock(spec=Job)
    job.job_id = "job"
    engine.run_processes = {"job": {}}
    assert "already started" in engine.start_app_on_server(context, job)
    engine.run_processes.clear()
    assert "does not exist" in engine.start_app_on_server(context, job)
    app = Workspace(engine.args.workspace, site_name="server").get_app_dir("job")
    from pathlib import Path

    Path(app).mkdir(parents=True)
    observations = []
    engine._start_runner_process = lambda *args: observations.append((engine.engine_info.status, args))
    assert engine.start_app_on_server(context, job, ["site-1"], "snapshot") == ""
    assert observations == [(MachineStatus.STARTING, (job, ["site-1"], "snapshot", context))]
    assert engine.engine_info.status == MachineStatus.STARTED


@pytest.mark.parametrize("method,marker", [("shutdown_server", "shutdown.fl"), ("restart_server", "restart.fl")])
def test_parent_lifecycle_commands_reject_starting_server_and_queue_marker_policy(engine, method, marker):
    engine.server.status = server_engine.ServerStatus.STARTING
    assert "starting" in getattr(engine, method)()
    engine.executor.submit.assert_not_called()
    engine.server.status = server_engine.ServerStatus.STARTED
    assert getattr(engine, method)() == ""
    callback, params = engine.executor.submit.call_args.args
    assert callable(callback)
    assert params == [engine.server, engine.args.workspace + "/" + marker]


def test_app_deploy_replaces_existing_destination_and_updates_name_marker(engine, tmp_path):
    uploads, run = tmp_path / "uploads", tmp_path / "job"
    (uploads / "source").mkdir(parents=True)
    (uploads / "source" / "model.txt").write_text("new")
    (run / "app_server").mkdir(parents=True)
    (run / "app_server" / "stale.txt").write_text("old")
    (run / "fl_app.txt").write_text("old-app")
    engine.server.admin_server.file_upload_dir = str(uploads)
    engine._get_run_folder = lambda job: str(run)
    assert "does not exist" in engine.deploy_app("job", "missing", "app_server")
    assert engine.deploy_app("job", "source", "app_server") == ""
    assert (run / "app_server" / "model.txt").read_text() == "new"
    assert not (run / "app_server" / "stale.txt").exists()
    assert (run / "fl_app.txt").read_text() == "source"
    engine.delete_job_id("job")
    assert not run.exists()


def test_app_commands_keep_topic_data_timeout_and_convert_dispatch_errors(engine):
    response = Shareable()
    engine.send_command_to_child_runner_process = Mock(return_value=response)
    assert engine.send_app_command("job", "topic", {"value": 2}, 3.0) is response
    kwargs = engine.send_command_to_child_runner_process.call_args.kwargs
    assert kwargs["command_name"] == ServerCommandNames.APP_COMMAND and kwargs["timeout"] == 3.0
    assert kwargs["command_data"][ServerCommandKey.TOPIC] == "topic"
    assert kwargs["command_data"][ServerCommandKey.DATA] == {"value": 2}
    engine.send_command_to_child_runner_process.side_effect = RuntimeError("offline")
    assert engine.send_app_command("job", "topic", {}, 1).get_return_code() == ReturnCode.EXECUTION_EXCEPTION
    assert engine.get_app_run_info("job") is None


def test_dead_job_notification_is_optional_fire_and_forget_with_identity(engine):
    engine.send_command_to_child_runner_process = Mock()
    engine.notify_dead_job("job", "site-1", "lost heartbeat")
    kwargs = engine.send_command_to_child_runner_process.call_args.kwargs
    assert kwargs["job_id"] == "job" and kwargs["timeout"] == 0.0 and kwargs["optional"] is True
    assert kwargs["command_data"].get_header(ServerCommandKey.FL_CLIENT) == "site-1"
    assert kwargs["command_data"].get_header(ServerCommandKey.REASON) == "lost heartbeat"


@pytest.mark.parametrize("available", [True, False])
def test_client_sync_accepts_participants_or_stops_at_independent_deadline(engine, monkeypatch, available):
    clients = {"token": object()}
    engine._retrieve_clients_data = Mock(return_value=clients if available else None)
    stamps = iter([0.0, 31.0])
    monkeypatch.setattr(server_engine, "time", SimpleNamespace(time=lambda: next(stamps), sleep=Mock()))
    if available:
        engine.sync_clients_from_main_process()
        engine.client_manager.set_clients.assert_called_once_with(clients)
    else:
        with pytest.raises(RuntimeError, match="Cannot get participating clients"):
            engine.sync_clients_from_main_process()
        engine.client_manager.set_clients.assert_not_called()


def test_resource_check_skips_server_and_unknown_clients_and_keeps_rejection_details(engine, monkeypatch):
    security = Mock()
    monkeypatch.setattr(server_engine, "set_message_security_data", security)
    job = Mock(spec=Job)
    job.job_id = "job"
    engine.get_client_from_name = lambda name: None if name == "missing" else SimpleNamespace(token=name + "-token")
    sufficient = Shareable()
    sufficient.set_header(ShareableHeader.IS_RESOURCE_ENOUGH, True)
    sufficient.set_header(ShareableHeader.RESOURCE_RESERVE_TOKEN, "reservation")
    ok = Message("reply", sufficient)
    refused = Message("reply", "capacity exceeded")
    refused.set_header(MsgHeader.RETURN_CODE, ReturnCode.ERROR)
    engine._send_admin_requests = Mock(
        return_value=[
            ClientReply("a-token", "a", None, ok),
            ClientReply("b-token", "b", None, refused),
            ClientReply("c-token", "c", None, None),
        ]
    )
    context = FLContext()
    result = engine.check_client_resources(job, {"server": {}, "missing": {}, "a": {"gpu": 1}, "b": {}}, context)
    assert result == {"a": (True, "reservation"), "b": (False, "capacity exceeded"), "c": (False, "")}
    requests = engine._send_admin_requests.call_args.args[0]
    assert set(requests) == {"a-token", "b-token"}
    assert requests["a-token"].body == {"gpu": 1}
    assert requests["a-token"].get_header(RequestHeader.JOB_ID) == "job"
    assert requests["a-token"].get_header(RequestHeader.REQUIRE_AUTHZ) == "false"
    assert security.call_count == 3


def test_client_job_start_includes_resource_reservation_and_job_metadata(engine):
    engine.get_client_from_name = lambda name: None if name == "missing" else SimpleNamespace(token="client-token")
    engine._send_admin_requests = Mock(return_value=[])
    job = SimpleNamespace(job_id="job", meta={"job": "metadata"})
    dispatch = SimpleNamespace(resource_requirements={"gpu": 1}, token="reservation")
    context = FLContext()
    assert engine.start_client_job(job, {"site-1": dispatch, "missing": dispatch}, context) == []
    args, kwargs = engine._send_admin_requests.call_args
    assert set(args[0]) == {"client-token"} and args[1] is context and kwargs == {"timeout_secs": 20}
    request = args[0]["client-token"]
    assert request.body == {"gpu": 1}
    assert request.get_header(RequestHeader.JOB_META) == {"job": "metadata"}
    assert request.get_header(ShareableHeader.RESOURCE_RESERVE_TOKEN) == "reservation"


def test_job_status_update_and_pause_use_current_job_set(engine):
    context = FLContext()
    context.set_prop(ReservedKey.RUN_NUM, "job")
    context.set_prop(FLContextKey.FATAL_SYSTEM_ERROR, True)
    engine.new_context = lambda: context
    engine.update_job_run_status()
    assert engine.server.cell.fire_and_forget.call_args.kwargs["message"].payload == {"execution_error": True}
    engine.run_processes = {"first": {}, "second": {}}
    engine.abort_app_on_server = Mock()
    engine.pause_server_jobs()
    assert [c.args[0] for c in engine.job_runner.remove_running_job.call_args_list] == ["first", "second"]
    assert [(c.args, c.kwargs) for c in engine.abort_app_on_server.call_args_list] == [
        (("first",), {"turn_to_cold": True}),
        (("second",), {"turn_to_cold": True}),
    ]
