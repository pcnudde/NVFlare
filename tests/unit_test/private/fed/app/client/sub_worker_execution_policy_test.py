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
from nvflare.apis.executor import Executor
from nvflare.apis.fl_constant import FLContextKey
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.fuel.common.multi_process_executor_constants import (
    CommunicateData,
    CommunicationMetaData,
    MultiProcessCommandNames,
)
from nvflare.fuel.f3.cellnet.defs import MessageHeaderKey, ReturnCode
from nvflare.fuel.f3.message import Message
from nvflare.private.fed.app.client import sub_worker_process as worker


@pytest.fixture
def executor(monkeypatch):
    cell, net_agent = Mock(), Mock()
    monkeypatch.setattr(worker, "Cell", Mock(return_value=cell))
    monkeypatch.setattr(worker, "NetAgent", Mock(return_value=net_agent))
    cleanup = Mock()
    monkeypatch.setattr(worker.mpm, "add_cleanup_cb", cleanup)
    args = SimpleNamespace(
        client_name="site-1", job_id="job-1", root_url="tcp://root", parent_url="tcp://parent", simulator_engine="false"
    )
    executor = worker.SubWorkerExecutor(args, object(), 2, 0)
    executor._registered_cleanup = cleanup
    executor._net_agent = net_agent
    return executor


def test_constructor_registers_only_control_callback_and_cleanup_boundaries(executor):
    kwargs = worker.Cell.call_args.kwargs
    assert kwargs["fqcn"] == "site-1.job-1.0" and kwargs["secure"] is False
    assert kwargs["create_internal_listener"] is True
    executor.cell.start.assert_called_once_with()
    assert executor.cell.register_request_cb.call_args.kwargs["cb"] == executor.execute_command
    assert [c.args[0] for c in executor._registered_cleanup.call_args_list] == [
        executor._net_agent.close,
        executor.cell.stop,
    ]
    assert set(executor.commands) == {
        MultiProcessCommandNames.INITIALIZE,
        MultiProcessCommandNames.TASK_EXECUTION,
        MultiProcessCommandNames.FIRE_EVENT,
        MultiProcessCommandNames.CLOSE,
    }


def test_unknown_command_returns_protocol_error_and_known_command_gets_payload(executor):
    request = Message(headers={MessageHeaderKey.TOPIC: "unknown"}, payload={"input": 1})
    assert executor.execute_command(request).get_header(MessageHeaderKey.RETURN_CODE) == ReturnCode.INVALID_REQUEST
    handler = Mock(return_value="result")
    executor.commands["custom"] = handler
    request.set_header(MessageHeaderKey.TOPIC, "custom")
    assert executor.execute_command(request) == "result"
    handler.assert_called_once_with({"input": 1})


@pytest.mark.parametrize("simulator", ["true", "false", "TRUE"])
def test_run_manager_factory_uses_explicit_simulator_mode(executor, monkeypatch, simulator):
    executor.args.simulator_engine = simulator
    normal, simulated = Mock(), Mock()
    monkeypatch.setattr(worker, "ClientRunManager", normal)
    monkeypatch.setattr(worker, "SimulatorClientRunManager", simulated)
    executor._get_client_run_manager("job-7")
    selected, other = (simulated, normal) if simulator.lower() == "true" else (normal, simulated)
    assert executor.run_manager is selected.return_value
    assert selected.call_args.kwargs["job_id"] == "job-7"
    assert selected.call_args.kwargs["components"] is executor.components
    assert selected.call_args.kwargs["handlers"] is executor.handlers
    other.assert_not_called()


def test_initialize_builds_components_and_propagates_rank_metadata(executor, monkeypatch):
    component = Mock(spec=Executor)
    builder = Mock()
    builder.build_component.return_value = component
    monkeypatch.setattr(worker, "WorkerComponentBuilder", Mock(return_value=builder))
    context, incoming = FLContext(), FLContext()
    manager = SimpleNamespace(components={}, add_handler=Mock(), new_context=lambda: context)
    executor._get_client_run_manager = lambda job: setattr(executor, "run_manager", manager)
    relayer = Mock()
    monkeypatch.setattr(worker, "EventRelayer", Mock(return_value=relayer))
    response = executor._initialize(
        {
            CommunicationMetaData.LOCAL_EXECUTOR: "exec",
            CommunicationMetaData.COMPONENTS: [{"id": "exec", "path": "test.Executor"}],
            CommunicationMetaData.FL_CTX: incoming,
        }
    )
    assert response.get_header(MessageHeaderKey.RETURN_CODE) == ReturnCode.OK
    assert executor.executor is component and executor.components == {"exec": component}
    assert executor.handlers == [component]
    assert manager.components[CommunicationMetaData.RELAYER] is relayer
    assert context.get_prop(FLContextKey.RANK_NUMBER) == 0
    assert context.get_prop(FLContextKey.NUM_OF_PROCESSES) == 2
    manager.add_handler.assert_called_once_with(relayer)
    relayer.relay_event.assert_called_once()
    data = relayer.relay_event.call_args.args[1]
    assert data[CommunicationMetaData.EVENT_TYPE] == EventType.START_RUN
    assert data[CommunicationMetaData.FL_CTX] is incoming


def test_initialize_rejects_missing_component_id_before_constructing_component(executor, monkeypatch):
    builder = Mock()
    monkeypatch.setattr(worker, "WorkerComponentBuilder", Mock(return_value=builder))
    with pytest.raises(TypeError, match="missing component id"):
        executor._initialize(
            {
                CommunicationMetaData.LOCAL_EXECUTOR: "exec",
                CommunicationMetaData.COMPONENTS: [{"path": "test.Executor"}],
                CommunicationMetaData.FL_CTX: FLContext(),
            }
        )
    builder.build_component.assert_not_called()


@pytest.mark.parametrize("rank", [0, 1])
def test_task_execution_sends_only_rank_zero_result_and_context_back_to_parent(executor, rank):
    executor.local_rank = rank
    context, incoming = FLContext(), FLContext()
    incoming.set_prop("input-state", "provided", private=False)
    executor.run_manager = SimpleNamespace(new_context=lambda: context)
    executor.executor = Mock(spec=Executor)
    result, request = Shareable(), Shareable()
    result["value"] = 17
    executor.executor.execute.return_value = result
    executor._execute_task(
        {
            CommunicationMetaData.TASK_NAME: "train",
            CommunicationMetaData.SHAREABLE: request,
            CommunicationMetaData.FL_CTX: incoming,
        }
    )
    args = executor.executor.execute.call_args.kwargs
    assert args["task_name"] == "train" and args["shareable"] is request and args["fl_ctx"] is context
    assert context.get_prop("input-state") == "provided"
    assert not args["abort_signal"].triggered
    if rank == 0:
        kwargs = executor.cell.send_request.call_args.kwargs
        assert kwargs["target"] == "site-1.job-1"
        assert kwargs["request"].payload[CommunicationMetaData.SHAREABLE] is result
        assert kwargs["request"].payload[CommunicationMetaData.FL_CTX].get_prop("input-state") == "provided"
    else:
        executor.cell.send_request.assert_not_called()


def test_worker_event_dispatch_and_stop_state_are_local_decisions(executor):
    relayer, data = Mock(), {"event": "custom"}
    executor.run_manager = Mock()
    executor.run_manager.get_component.return_value = relayer
    executor._handle_event(data)
    relayer.relay_event.assert_called_once_with(executor.run_manager, data)
    executor._close({})
    assert executor.done is True
    executor.done = False
    executor.stop()
    assert executor.done is True
    executor.run()


def test_event_relayer_fires_start_once_and_marks_remote_origin():
    context, incoming = FLContext(), FLContext()
    incoming.set_prop("copied", "retained", private=False)
    manager = SimpleNamespace(new_context=lambda: context)
    relayer = worker.EventRelayer(Mock(), "site-1.job-1", 1)
    relayer.fire_event = Mock()
    data = {CommunicationMetaData.EVENT_TYPE: EventType.START_RUN, CommunicationMetaData.FL_CTX: incoming}
    relayer.relay_event(manager, data)
    relayer.relay_event(manager, data)
    assert relayer.start_run_fired is True
    assert context.get_prop("copied") == "retained"
    assert context.get_prop(FLContextKey.EVENT_ORIGIN_SITE) == CommunicateData.MULTI_PROCESS_EXECUTOR
    relayer.fire_event.assert_called_once_with(event_type=EventType.START_RUN, fl_ctx=context)


def test_event_relayer_roundtrip_merges_parent_updates_and_prevents_echo():
    cell = Mock()
    context, returned = FLContext(), FLContext()
    context.set_prop("request", 1, private=False)
    returned.set_prop("response", 2, private=False)
    cell.send_request.return_value = Message(payload={CommunicationMetaData.FL_CTX: returned})
    relayer = worker.EventRelayer(cell, "site-1.job-1", 1)
    relayer.handle_event("custom", context)
    kwargs = cell.send_request.call_args.kwargs
    assert kwargs["target"] == "site-1.job-1"
    data = kwargs["request"].payload
    assert data[CommunicationMetaData.EVENT_TYPE] == "custom" and data[CommunicationMetaData.RANK_NUMBER] == 1
    assert data[CommunicationMetaData.FL_CTX].get_prop("request") == 1
    assert context.get_prop("response") == 2
    context.set_prop(FLContextKey.EVENT_ORIGIN_SITE, CommunicateData.MULTI_PROCESS_EXECUTOR)
    relayer.handle_event("custom", context)
    assert cell.send_request.call_count == 1


def test_event_relay_transport_failure_is_contained_as_warning():
    cell = Mock()
    cell.send_request.side_effect = RuntimeError("parent disconnected")
    relayer = worker.EventRelayer(cell, "site-1.job-1", 1)
    relayer.log_warning = Mock()
    relayer.handle_event("custom", FLContext())
    assert "Failed to relay" in relayer.log_warning.call_args.args[1]
    assert relayer.log_warning.call_args.kwargs["fire_event"] is False
