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

from nvflare.apis.fl_constant import FLContextKey, ServerCommandKey
from nvflare.apis.fl_context import FLContext
from nvflare.apis.fl_exception import FLCommunicationError
from nvflare.apis.shareable import ReservedHeaderKey, Shareable
from nvflare.fuel.f3.cellnet.utils import new_cell_message
from nvflare.private.fed.client import fed_client_base as subject
from tests.timing_utils import isolate_time


@pytest.fixture
def client(monkeypatch):
    communicator = Mock()
    monkeypatch.setattr(subject, "Communicator", Mock(return_value=communicator))
    monkeypatch.setattr(subject, "set_scope_property", Mock())
    value = subject.FederatedClientBase(
        "site-a", {}, False, server_args={"project": {"target": "old"}}, args=SimpleNamespace(job_id=None)
    )
    value.token = "token"
    value.ssid = "session"
    return value


@pytest.mark.parametrize("existing_cell", [True, False])
def test_connect_updates_target_and_establishes_before_creating_cell(client, monkeypatch, existing_cell):
    client.cell = Mock() if existing_cell else None

    def create_cell(*args):
        assert client.sp_established

    create = Mock(side_effect=create_cell)
    monkeypatch.setattr(client, "_create_cell", create)
    client.connect_to_server("project", "host:1234", "tcp")
    assert client.servers["project"]["target"] == "host:1234"
    assert client.sp_established
    subject.set_scope_property.assert_called_once_with(
        scope_name="site-a", value="host", key=FLContextKey.SERVER_HOST_NAME
    )
    if existing_cell:
        client.cell.change_server_root.assert_called_once_with("tcp://host:1234")
    else:
        create.assert_called_once_with("host:1234", "tcp")


@pytest.mark.parametrize("failure", [False, True])
def test_registration_sets_identity_or_stops_heartbeat_on_communication_failure(client, failure):
    client.token = None
    if failure:
        client.communicator.client_registration.side_effect = FLCommunicationError("offline")
    else:
        client.communicator.client_registration.return_value = ("issued", "signature", "session")
    ctx = FLContext()
    client.register(ctx)
    client.communicator.client_registration.assert_called_once_with("site-a", "project", ctx)
    if failure:
        assert client.communicator.heartbeat_done
    else:
        assert (client.token, client.token_signature, client.ssid) == ("issued", "signature", "session")
        assert client.fl_ctx.get_prop(FLContextKey.CLIENT_NAME) == "site-a"


@pytest.mark.parametrize("failure", [False, True])
def test_task_pull_returns_task_payload_and_contains_communication_error(client, failure):
    ctx = FLContext()
    data = Shareable()
    data.set_header(ServerCommandKey.TASK_NAME, "train")
    if failure:
        client.communicator.pull_task.side_effect = FLCommunicationError("offline")
    else:
        client.communicator.pull_task.return_value = new_cell_message({}, data)
    result = client.pull_task(ctx, timeout=8)
    assert result == ((False, None, None) if failure else (True, "train", data))
    client.communicator.pull_task.assert_called_once_with("project", "token", "session", ctx, timeout=8)


@pytest.mark.parametrize("header", [True, False])
def test_result_submission_prefers_shareable_task_name_then_context(client, header):
    ctx = FLContext()
    ctx.set_prop(FLContextKey.TASK_NAME, "context-task")
    result = Shareable()
    if header:
        result.set_header(ReservedHeaderKey.TASK_NAME, "result-task")
    code = client.push_results(result, ctx, timeout=4)
    client.communicator.submit_update.assert_called_once_with(
        "project", "token", "session", ctx, "site-a", result, "result-task" if header else "context-task", timeout=4
    )
    assert code is client.communicator.submit_update.return_value
    client.communicator.submit_update.side_effect = FLCommunicationError("offline")
    assert client.push_results(result, ctx) is None


def test_heartbeat_requires_engine_and_marks_failed_communication(client, monkeypatch):
    clock = isolate_time(monkeypatch, subject)
    monkeypatch.setattr(clock, "time", Mock(side_effect=[0, 61]))
    monkeypatch.setattr(clock, "sleep", Mock(side_effect=[None]))
    with pytest.raises(RuntimeError, match="No engine created"):
        client.send_heartbeat("project", 2)
    client.engine = Mock()
    monkeypatch.setattr(clock, "time", lambda: 100)
    client.communicator.send_heartbeat.side_effect = FLCommunicationError("offline")
    client.send_heartbeat("project", 2)
    assert client.communicator.heartbeat_done


def test_heartbeat_success_dispatch_and_logout_preserve_session(client):
    client.engine = Mock()
    client.heartbeat(7)
    client.communicator.send_heartbeat.assert_called_once_with(
        client.servers, "project", "token", "session", "site-a", client.engine, 7
    )
    ctx = FLContext()
    assert client.logout_client(ctx) is client.communicator.quit_remote.return_value
    client.communicator.quit_remote.assert_called_once_with(client.servers, "project", "token", "session", ctx)


def test_heartbeat_thread_is_daemon_and_callback_contains_exception(client, monkeypatch):
    thread = Mock()
    factory = Mock(return_value=thread)
    monkeypatch.setattr(subject, "threading", SimpleNamespace(Thread=factory))
    client.start_heartbeat(3)
    factory.assert_called_once_with(target=client.run_heartbeat, args=[3])
    assert thread.daemon
    thread.start.assert_called_once_with()
    client.heartbeat = Mock(side_effect=RuntimeError("failed"))
    client.logger = Mock()
    client.run_heartbeat(3)
    client.logger.error.assert_called_once_with("Failed to start run_heartbeat.")


def test_engine_and_runner_are_assigned_and_existing_cell_can_be_stopped(client):
    engine, runner = Mock(), Mock()
    client.set_client_engine(engine)
    client.set_client_runner(runner)
    assert client.engine is engine and client.client_runner is runner
    client.stop_cell()
    client.communicator.cell.stop.assert_called_once_with()
