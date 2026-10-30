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

import threading
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from nvflare.apis.client import Client
from nvflare.apis.fl_constant import ConnPropKey, FLContextKey, MachineStatus, RunProcessKey, SecureTrainConst
from nvflare.apis.fl_context import FLContext
from nvflare.fuel.f3.cellnet.defs import MessageHeaderKey
from nvflare.fuel.f3.drivers.driver_params import DriverParams
from nvflare.fuel.f3.message import Message
from nvflare.private.defs import CellMessageHeaderKeys, JobOutcomeMsgKey
from nvflare.private.fed.server import fed_server as module


@pytest.fixture
def server():
    subject = module.FederatedServer.__new__(module.FederatedServer)
    subject.logger = Mock()
    subject.lock = threading.Lock()
    subject.engine = Mock()
    subject.engine.new_context.side_effect = lambda: FLContext()
    subject.engine.run_processes = {}
    subject.client_manager = Mock()
    subject.client_manager.clients = {}
    subject.server_runner = Mock()
    subject.cell = Mock()
    subject.shutdown = False
    return subject


def test_secure_server_deployment_propagates_credentials_and_registers_auth_filters_without_runtime(
    monkeypatch, server
):
    cell = Mock()
    factory = Mock(return_value=cell)
    monkeypatch.setattr(module, "Cell", factory)
    cleanup_worker, registration_worker = Mock(), Mock()
    workers = Mock(side_effect=[cleanup_worker, registration_worker])
    monkeypatch.setattr(module, "threading", SimpleNamespace(Thread=workers))
    cleanup = Mock()
    monkeypatch.setattr(module, "mpm", SimpleNamespace(add_cleanup_cb=cleanup))
    config = {
        "service": {"target": "localhost:8002", "scheme": "grpc"},
        SecureTrainConst.SSL_ROOT_CERT: "root.pem",
        SecureTrainConst.SSL_CERT: "server.pem",
        SecureTrainConst.PRIVATE_KEY: "server.key",
        ConnPropKey.CONNECTION_SECURITY: "mtls",
    }
    server.deploy(object(), config, secure_train=True, enable_admin_listener=False)
    assert factory.call_args.kwargs["credentials"] == {
        DriverParams.CA_CERT.value: "root.pem",
        DriverParams.SERVER_CERT.value: "server.pem",
        DriverParams.SERVER_KEY.value: "server.key",
        DriverParams.CONNECTION_SECURITY.value: "mtls",
    }
    cell.start.assert_called_once_with()
    cleanup.assert_called_once_with(cell.stop)
    cleanup_worker.start.assert_called_once_with()
    registration_worker.start.assert_called_once_with()
    assert workers.call_args_list[0].kwargs["target"] == server.client_cleanup
    assert workers.call_args_list[1].kwargs["target"] == server._check_regs
    cell.core_cell.add_incoming_filter.assert_called_once_with(channel="*", topic="*", cb=server._validate_auth_headers)
    cell.core_cell.add_outgoing_reply_filter.assert_called_once_with(
        channel="*", topic="*", cb=server._add_auth_headers
    )
    cell.core_cell.add_outgoing_request_filter.assert_called_once_with(
        channel="*", topic="*", cb=server._add_auth_headers
    )
    assert server.get_cell() is cell


def test_server_deployment_rejects_malformed_service_target_before_starting_cell(monkeypatch, server):
    factory = Mock()
    monkeypatch.setattr(module, "Cell", factory)
    with pytest.raises(RuntimeError, match="bad service target"):
        module.BaseServer.deploy(server, object(), {"service": {"target": "missing-port"}})
    factory.assert_not_called()


@pytest.mark.parametrize("payload", [[], {}, {JobOutcomeMsgKey.JOB_ID: "job"}])
def test_terminal_outcome_rejects_bad_payload_missing_job_and_unregistered_token(server, payload):
    server.client_manager.is_from_authorized_client.return_value = True
    request = Message(
        headers={CellMessageHeaderKeys.TOKEN: "unknown", MessageHeaderKey.ORIGIN: "site"}, payload=payload
    )
    response = server.process_job_outcome(request)
    assert response.get_header(MessageHeaderKey.RETURN_CODE) == (
        module.F3ReturnCode.UNAUTHENTICATED if payload else module.F3ReturnCode.INVALID_REQUEST
    )
    server.engine.job_runner.resolve_client_outcome.assert_not_called()
    assert server.logger.warning.call_count == int(bool(payload))


def test_auth_without_pki_strips_peer_transit_headers_and_asserter_uses_engine_context(server):
    server.cred_keeper = Mock()
    context = FLContext()
    server.engine.new_context.side_effect = None
    server.engine.new_context.return_value = context
    assert server._get_id_asserter() is server.cred_keeper.get_id_asserter.return_value
    server.cred_keeper.get_id_asserter.assert_called_once_with(context)
    server.cred_keeper.get_id_asserter.return_value = None
    message = Message(
        headers={MessageHeaderKey.CONN_URL: "tcp://untrusted:8002", MessageHeaderKey.SERVER_TRANSIT_REQUIRED: True}
    )
    assert server._validate_auth_headers(message) is None
    assert message.get_header(MessageHeaderKey.CONN_URL) is None
    assert server._resolve_client_fqcn_for_auth("site", "unknown") is None


def test_job_cell_credentials_include_transport_security_mode_and_reset_token_metadata(monkeypatch, server):
    factory = Mock()
    monkeypatch.setattr(module, "Cell", factory)
    agent, command, cleanup = Mock(), Mock(), Mock()
    monkeypatch.setattr(module, "NetAgent", Mock(return_value=agent))
    monkeypatch.setattr(module, "ServerCommandAgent", Mock(return_value=command))
    monkeypatch.setattr(module, "mpm", SimpleNamespace(add_cleanup_cb=cleanup))
    config = {
        SecureTrainConst.SSL_ROOT_CERT: "root",
        SecureTrainConst.SSL_CERT: "certificate",
        SecureTrainConst.PRIVATE_KEY: "key",
        ConnPropKey.CONNECTION_SECURITY: "tls",
    }
    result = server.create_job_cell("job", "grpc://root", "tcp://parent", True, config)
    assert result is factory.return_value
    assert [call.args for call in cleanup.call_args_list] == [(agent.close,), (result.stop,)]
    command.start.assert_called_once_with()
    credentials = factory.call_args.kwargs["credentials"]
    assert credentials[DriverParams.CONNECTION_SECURITY.value] == "tls"
    assert credentials[DriverParams.CLIENT_CERT.value] == credentials[DriverParams.SERVER_CERT.value] == "certificate"
    clients = {"token": Client("site", "token")}
    server.get_all_clients = Mock(return_value=clients)
    server.task_meta_info = Mock(return_value={"task": "train"})
    server.tokens = {}
    server.reset_tokens()
    assert server.tokens == {"token": {"task": "train"}}
    server.task_meta_info.assert_called_once_with("site")


@pytest.mark.parametrize("action", [module.NIS, module.ABORT_RUN])
def test_server_state_refusal_records_transient_context_error(server, action):
    context = FLContext()
    assert server._handle_state_check({module.ACTION: action, module.MESSAGE: "unavailable"}, context) == "unavailable"
    assert context.get_prop(FLContextKey.COMMUNICATION_ERROR) == "unavailable"


def test_dead_client_only_notifies_jobs_with_active_participation_and_contains_runner_error(server):
    client = Client("site", "token")
    server.engine.job_runner.get_client_outcome_jobs.return_value = ["pending"]
    resolve = Mock()
    server._resolve_missing_client_outcome = resolve
    server.engine.run_processes = {"active": {RunProcessKey.PARTICIPANTS: {"token": client}}, "other": {}}
    server.engine.notify_dead_job.side_effect = ValueError("runner stopped")
    server.notify_dead_client(client)
    resolve.assert_called_once_with(client, "pending", "client dead")
    server.engine.notify_dead_job.assert_called_once_with("active", "site", "client dead")
    assert "runner stopped" in server.logger.info.call_args.args[0]


@pytest.mark.parametrize("failure", [True, False])
def test_engine_run_always_releases_cell_and_transitions_from_started_to_stopped(server, failure):
    def run():
        assert server.engine.engine_info.status == MachineStatus.STARTED
        if failure:
            raise ValueError("workflow failed")

    server.server_runner.run.side_effect = run
    stop = Mock()
    server.stop_run_engine_cell = stop
    server.run_engine()
    stop.assert_called_once_with()
    assert server.engine.engine_info.status == MachineStatus.STOPPED
    assert server.logger.error.call_count == int(failure)
    server.abort_run()
    server.server_runner.abort.assert_called_once()


def test_server_parent_heartbeat_and_run_manager_factory_preserve_job_identity(monkeypatch, server):
    server.project_name = "project"
    server.runner_config = SimpleNamespace(components={"component": object()}, handlers=[])
    factory = Mock()
    monkeypatch.setattr(module, "RunManager", factory)
    assert server.create_run_manager("workspace", "job") is factory.return_value
    assert factory.call_args.kwargs == dict(
        server_name="project",
        engine=server.engine,
        job_id="job",
        workspace="workspace",
        components=server.runner_config.components,
        client_manager=server.client_manager,
        handlers=[],
    )
    server._send_parent_heartbeat("job")
    payload = server.cell.fire_and_forget.call_args.kwargs["message"]
    assert payload.get_header(CellMessageHeaderKeys.JOB_ID) == "job"
    assert server.cell.fire_and_forget.call_args.kwargs["targets"] == module.FQCN.ROOT_SERVER
