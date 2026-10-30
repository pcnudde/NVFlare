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

from argparse import Namespace
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from nvflare.apis.client import Client
from nvflare.apis.fl_constant import MachineStatus
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.fuel.f3.cellnet.defs import MessageHeaderKey
from nvflare.fuel.f3.message import Message
from nvflare.private.fed.server import server_engine as module


@pytest.fixture
def engine(monkeypatch, tmp_path):
    monkeypatch.setattr(module, "ThreadPoolExecutor", Mock(return_value=Mock()))
    server = Mock()
    server.admin_server.file_upload_dir = str(tmp_path)
    subject = module.ServerEngine(server, Namespace(workspace=str(tmp_path), job_id="job", set=[]), Mock(), None)
    subject.logger = Mock()
    subject.run_manager = Mock()
    subject.run_manager.new_context.return_value = FLContext()
    return subject


def test_job_readiness_parent_selection_and_exception_cleanup_preserve_valid_jobs(engine):
    first, second = Client("first", "token-1"), Client("second", "token-2")
    engine.client_manager.get_client_from_name.side_effect = [first, None, second]
    assert engine.get_job_clients({"first": object(), "lost": object(), "second": object()}) == {
        "token-1": first,
        "token-2": second,
    }
    assert engine.get_job_clients({}) == {}
    engine.exception_run_processes = {"job": object(), "other": object()}
    engine.remove_exception_process("job")
    engine.remove_exception_process("missing")
    assert set(engine.exception_run_processes) == {"other"}
    assert "has not started" in engine.check_app_start_readiness("job")
    engine.run_processes["job"] = {}
    assert engine.check_app_start_readiness("job") == ""
    for status, message in [
        (MachineStatus.STOPPED, "has not started"),
        (MachineStatus.STARTING, "please wait"),
        (MachineStatus.STARTED, ""),
    ]:
        engine.engine_info.status = status
        result = engine.abort_app_on_clients(["first"])
        if message:
            assert message in result
        else:
            assert result == ""


def test_engine_component_dispatch_requires_configurator_and_forwards_runtime_events(engine):
    context = FLContext()
    handler = object()
    engine.set_job_runner(handler, "manager")
    assert engine.job_runner is handler and engine.job_def_manager == "manager"
    with pytest.raises(TypeError, match="ServerJsonConfigurator"):
        engine.set_configurator(object())
    with pytest.raises(RuntimeError, match="No configurator"):
        engine.build_component({"id": "example"})
    configurator = Mock(spec=module.ServerJsonConfigurator)
    engine.set_configurator(configurator)
    assert engine.build_component({"id": "example"}) is configurator.build_component.return_value
    configurator.build_component.assert_called_once_with({"id": "example"})
    engine.fire_event("custom", context)
    engine.run_manager.fire_event.assert_called_once_with("custom", context)
    assert engine.get_run_info() is engine.run_manager.get_run_info.return_value
    assert engine.get_workspace() is engine.run_manager.get_workspace.return_value
    engine.run_manager = None
    assert engine.get_run_info() is None
    assert isinstance(engine.new_context(), FLContext)


def test_aux_dispatch_wraps_success_with_public_peer_context_and_rejects_nonmessage(engine):
    context = engine.run_manager.new_context.return_value
    context.set_prop("public", "value", private=False)
    context.set_prop("private", "hidden", private=True)
    reply = Shareable({"result": 7})
    engine.run_manager.aux_runner.dispatch.return_value = reply
    request = Message(headers={MessageHeaderKey.TOPIC: "task"}, payload=Shareable({"input": 3}))
    result = engine._handle_aux_message(request)
    engine.run_manager.aux_runner.dispatch.assert_called_once_with(
        topic="task", request=request.payload, fl_ctx=context
    )
    assert result.payload is reply
    assert result.get_header(MessageHeaderKey.RETURN_CODE) == module.CellMsgReturnCode.OK
    peer = reply.get_peer_context()
    assert peer.get_prop("public") == "value" and peer.get_prop("private") is None
    with pytest.raises(AssertionError, match="CellMessage"):
        engine._handle_aux_message({})


@pytest.mark.parametrize("operation", ["stream", "register"])
def test_streaming_requires_created_streamer_and_preserves_routing_policy(engine, operation):
    context = FLContext()
    producer = object()
    targets = [object()]
    engine._to_aux_msg_targets = Mock(return_value=targets)

    def invoke():
        if operation == "stream":
            return engine.stream_objects(
                "channel", "topic", {"id": 1}, ["site"], producer, context, optional=True, secure=True
            )
        return engine.register_stream_processing("channel", "topic", producer, consumed_cb="callback", user="metadata")

    engine.run_manager.object_streamer = None
    with pytest.raises(RuntimeError, match="object_streamer"):
        invoke()
    streamer = Mock()
    engine.run_manager.object_streamer = streamer
    result = invoke()
    if operation == "stream":
        assert result is streamer.stream.return_value
        streamer.stream.assert_called_once_with(
            channel="channel",
            topic="topic",
            stream_ctx={"id": 1},
            targets=targets,
            producer=producer,
            fl_ctx=context,
            secure=True,
            optional=True,
        )
    else:
        streamer.register_stream_processing.assert_called_once_with(
            channel="channel",
            topic="topic",
            factory=producer,
            stream_done_cb=None,
            consumed_cb="callback",
            user="metadata",
        )
    engine.run_manager = None
    with pytest.raises(RuntimeError, match="run_manager"):
        invoke()


def test_aux_send_contains_transport_failure_and_empty_target_routes_do_not_send(engine):
    context = FLContext()
    engine.send_aux_to_targets = Mock(side_effect=ValueError("transport unavailable"))
    assert engine.send_aux_request(["site"], "topic", Shareable(), 1, context) is None
    assert "transport unavailable" in engine.logger.error.call_args.args[0]
    engine._get_aux_msg_target = Mock(return_value=None)
    assert engine.multicast_aux_requests("topic", {"unknown": Shareable()}, 1, context) == {}
    engine.run_manager.aux_runner.multicast_aux_requests.assert_not_called()


@pytest.mark.parametrize("has_admin", [True, False])
def test_disabling_client_removes_active_sessions_and_dead_client_removal_is_consistent(engine, has_admin):
    admin = Mock() if has_admin else None
    engine.server.admin_server = admin
    manager = engine.server.client_manager
    manager.is_client_disabled.return_value = False
    manager.disable_client.return_value = ["token-1", "token-2"]
    result = engine.disable_clients(["site"])["clients"][0]
    assert result == dict(
        client_name="site",
        state="disabled",
        already_disabled=False,
        active_session_removed=True,
        credential_revoked=False,
        rejoin_allowed=False,
    )
    assert [call.args for call in engine.server.remove_client_data.call_args_list] == [("token-1",), ("token-2",)]
    if has_admin:
        assert [call.args for call in admin.client_dead.call_args_list] == [("token-1",), ("token-2",)]
    engine.server.remove_client_data.reset_mock()
    assert engine.remove_clients(["token-3"]) == ""
    manager.remove_client.assert_called_once_with("token-3")
    engine.server.remove_client_data.assert_called_once_with("token-3")


def test_client_sync_retries_once_then_installs_parent_membership_without_real_sleep(monkeypatch, engine):
    clients = {"token": object()}
    retrieve = Mock(side_effect=[None, clients])
    engine._retrieve_clients_data = retrieve
    now, pauses = [0.0], []

    def sleep(seconds):
        pauses.append(seconds)
        assert len(pauses) == 1, "client synchronization exceeded poll budget"
        now[0] += seconds

    monkeypatch.setattr(module, "time", SimpleNamespace(time=lambda: now[0], sleep=sleep))
    engine.sync_clients_from_main_process()
    assert retrieve.call_count == 2 and pauses == [1.0]
    engine.client_manager.set_clients.assert_called_once_with(clients)
    engine.client_manager.clients = clients
    assert engine.get_participating_clients() is clients


def test_engine_cleanup_stops_jobs_and_closes_executor_before_streamer(engine):
    context = engine.run_manager.new_context.return_value
    engine.job_runner = Mock()
    engine.stop_all_jobs()
    engine.job_runner.stop_all_runs.assert_called_once_with(context)
    events = []
    engine.executor.shutdown.side_effect = lambda: events.append("executor")
    engine.run_manager.object_streamer.shutdown.side_effect = lambda: events.append("streamer")
    engine.close()
    assert events == ["executor", "streamer"]


@pytest.mark.parametrize("failure", [True, False])
def test_server_shutdown_always_closes_security_and_marks_shutdown(monkeypatch, tmp_path, failure):
    server = Mock()
    if failure:
        server.fl_shutdown.side_effect = ValueError("shutdown failed")
    security = Mock()
    monkeypatch.setattr(module, "security_close", security)
    marker = tmp_path / "shutdown"
    if failure:
        with pytest.raises(ValueError, match="shutdown failed"):
            module.server_shutdown(server, str(marker))
    else:
        module.server_shutdown(server, str(marker))
    assert marker.is_file()
    assert server.status == module.ServerStatus.SHUTDOWN
    security.assert_called_once_with()
