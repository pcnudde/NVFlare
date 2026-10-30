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
from nvflare.apis.fl_constant import FLContextKey, ReservedKey
from nvflare.apis.fl_context import FLContext
from nvflare.private.fed.server import server_runner
from nvflare.private.fed.server.server_engine import ServerEngine


@pytest.fixture
def runner():
    context = FLContext()
    engine = Mock(spec=ServerEngine)
    engine.new_context.side_effect = lambda: context
    engine.get_participating_clients.return_value = {}
    context.set_prop(ReservedKey.ENGINE, engine)
    workflows = [SimpleNamespace(id="first", controller=Mock()), SimpleNamespace(id="second", controller=Mock())]
    config = server_runner.ServerRunnerConfig(5, 1, workflows, {}, {}, handlers=[], components={})
    runner = server_runner.ServerRunner(config, "job", engine)
    for name in ("log_info", "log_debug", "log_error", "log_exception", "fire_event"):
        setattr(runner, name, Mock())
    runner._test_context = context
    return runner


def test_component_config_adds_handlers_only_for_flcomponents_and_rejects_duplicate_ids(runner):
    config = runner.config
    component, plain = FLComponent(), object()
    config.add_component("handler", component)
    config.add_component("plain", plain)
    assert config.components == {"handler": component, "plain": plain} and config.handlers == [component]
    with pytest.raises(TypeError, match="must be str"):
        config.add_component(1, plain)
    with pytest.raises(ValueError, match="duplicate"):
        config.add_component("plain", object())


def test_workflow_is_exposed_only_after_initialization_and_removed_before_finalize(runner):
    observations = []
    for workflow in runner.config.workflows:
        workflow.controller.initialize.side_effect = lambda ctx: observations.append(("initialize", runner.current_wf))
        workflow.controller.control_flow.side_effect = lambda signal, ctx: observations.append(
            ("run", runner.current_wf.id)
        )
        workflow.controller.stop_controller.side_effect = lambda ctx: observations.append(
            ("finalize", runner.current_wf)
        )
    runner._execute_run()
    assert observations == [
        ("initialize", None),
        ("run", "first"),
        ("finalize", None),
        ("initialize", None),
        ("run", "second"),
        ("finalize", None),
    ]
    assert runner.current_wf is None and runner.current_wf_index == 2
    assert [c.args[0] for c in runner.fire_event.call_args_list] == [
        EventType.START_WORKFLOW,
        EventType.END_WORKFLOW,
        EventType.START_WORKFLOW,
        EventType.END_WORKFLOW,
    ]
    for workflow in runner.config.workflows:
        workflow.controller.communicator.initialize_run.assert_called_once_with(runner._test_context)
        workflow.controller.communicator.finalize_run.assert_called_once_with(runner._test_context)


@pytest.mark.parametrize("phase", ["initialize", "control_flow", "finalize"])
def test_workflow_failure_always_removes_current_workflow_and_emits_end_event(runner, phase):
    first, second = runner.config.workflows
    if phase == "finalize":
        first.controller.stop_controller.side_effect = RuntimeError("finalization rejected")
    else:
        getattr(first.controller, phase).side_effect = RuntimeError("execution rejected")
    runner.system_panic = Mock(side_effect=lambda message, context: runner.abort_signal.trigger(True))
    runner._execute_run()
    assert runner.current_wf is None
    assert EventType.END_WORKFLOW in [c.args[0] for c in runner.fire_event.call_args_list]
    first.controller.stop_controller.assert_called_once_with(runner._test_context)
    runner.log_exception.assert_called()
    if phase != "finalize":
        runner.system_panic.assert_called_once()
        second.controller.initialize.assert_not_called()
        first.controller.communicator.finalize_run.assert_called_once_with(runner._test_context)
    else:
        second.controller.initialize.assert_called_once_with(runner._test_context)


@pytest.mark.parametrize("cold,execution_error", [(False, False), (True, False), (False, True)])
def test_run_teardown_releases_services_and_skips_client_end_run_for_cold_transition(
    runner, monkeypatch, cold, execution_error
):
    reliable, downloads = Mock(), Mock()
    monkeypatch.setattr(server_runner, "ReliableMessage", reliable)
    monkeypatch.setattr(server_runner, "DownloadService", downloads)
    runner.turn_to_cold = cold
    runner._execute_run = Mock(side_effect=RuntimeError("workflow failed") if execution_error else None)
    runner.check_end_run_readiness = Mock()
    runner.run()
    assert runner.status == "done"
    assert runner._test_context.get_prop(ReservedKey.RUN_ABORT_SIGNAL) is runner.abort_signal
    assert runner._test_context.get_prop(FLContextKey.CLIENT_HIERARCHY) is not None
    assert [c.args[0] for c in runner.fire_event.call_args_list] == [
        EventType.START_RUN,
        EventType.ABOUT_TO_END_RUN,
        EventType.END_RUN,
    ]
    persist_calls = [c.kwargs["completed"] for c in runner.engine.persist_components.call_args_list]
    assert persist_calls == ([False] if cold else [False, True])
    if cold:
        runner.engine.send_aux_request.assert_not_called()
    else:
        kwargs = runner.engine.send_aux_request.call_args.kwargs
        assert kwargs["targets"] is None and kwargs["timeout"] == 0.0
        assert kwargs["optional"] is True and kwargs["secure"] is False
    runner.check_end_run_readiness.assert_called_once_with(runner._test_context)
    reliable.enable.assert_called_once_with(runner._test_context)
    reliable.shutdown.assert_called_once_with()
    downloads.shutdown.assert_called_once_with()
    runner.engine.shutdown_streamer.assert_called_once_with()


def test_dead_job_and_heartbeat_update_only_active_workflow_communicator(runner):
    context, peer = runner._test_context, FLContext()
    peer.set_prop(ReservedKey.IDENTITY_NAME, "site-1")
    context.set_peer_context(peer)
    runner.handle_dead_job("site-1", context)
    workflow = runner.config.workflows[0]
    runner.current_wf = workflow
    runner.handle_dead_job("site-1", context)
    workflow.controller.communicator.process_dead_client_report.assert_called_once_with("site-1", context)
    runner._handle_job_heartbeat("heartbeat", None, context)
    workflow.controller.communicator.client_is_active.assert_called_once_with("site-1", "jobHeartbeat", context)
    workflow.controller.communicator.process_dead_client_report.side_effect = RuntimeError("workflow failure")
    runner.handle_dead_job("site-1", context)
    runner.log_exception.assert_called_once()


def test_fatal_system_event_marks_context_and_aborts_current_run(runner):
    context = runner._test_context
    context.set_prop(FLContextKey.EVENT_DATA, "unsafe payload")
    runner.abort = Mock()
    runner.handle_event(EventType.FATAL_SYSTEM_ERROR, context)
    assert context.get_prop(FLContextKey.FATAL_SYSTEM_ERROR) is True
    runner.abort.assert_called_once_with(context)
