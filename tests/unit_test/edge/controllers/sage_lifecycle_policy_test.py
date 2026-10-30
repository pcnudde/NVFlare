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

from nvflare.apis.client import Client
from nvflare.apis.controller_spec import ClientTask, Task, TaskCompletionStatus
from nvflare.apis.fl_constant import FLContextKey, ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.apis.signal import Signal
from nvflare.app_common.app_constant import AppConstants
from nvflare.app_common.app_event_type import AppEventType
from nvflare.edge.assessor import Assessment, Assessor
from nvflare.edge.constants import EdgeTaskHeaderKey
from nvflare.edge.controllers import sage as subject
from nvflare.fuel.utils.waiter_utils import WaiterRC
from nvflare.widgets.info_collector import GroupInfoCollector, InfoCollector
from tests.timing_utils import isolate_time


@pytest.fixture
def runtime(monkeypatch):
    controller = subject.ScatterAndGatherForEdge(num_rounds=2, update_interval=0.25)
    controller.assessor = Mock(spec=Assessor)
    controller.assessor.start_task.side_effect = lambda ctx: Shareable({"weights": [1, 2]})
    controller._children = ["a", "b"]
    controller._num_children = 2
    for name in (
        "log_info",
        "log_debug",
        "log_warning",
        "log_exception",
        "system_panic",
        "fire_event",
        "fire_event_with_data",
        "broadcast",
        "cancel_task",
    ):
        monkeypatch.setattr(controller, name, Mock())
    monkeypatch.setattr(subject.ScatterAndGatherForEdge, "next_task_seq", 0)
    local_time = isolate_time(monkeypatch, subject)
    monkeypatch.setattr(local_time, "sleep", Mock())
    monkeypatch.setattr(subject.gc, "collect", Mock())
    ctx = FLContext()
    engine = Mock()
    monkeypatch.setattr(ctx, "get_engine", lambda: engine)
    return controller, ctx, engine, local_time


def test_start_validates_assessor_and_registers_child_update_handler(runtime):
    controller, ctx, engine, _ = runtime
    assessor = controller.assessor
    engine.get_component.return_value = assessor
    ctx.set_prop(FLContextKey.CLIENT_HIERARCHY, SimpleNamespace(roots=["root-one"]))
    controller.start_controller(ctx)
    assert controller.assessor is assessor
    assert controller._children == ["root-one"]
    assert controller._num_children == 1
    engine.register_aux_message_handler.assert_called_once_with("SAGE__train_update", controller._process_update_report)
    controller.system_panic.assert_not_called()


def test_start_rejects_non_assessor_before_registering_messages(runtime):
    controller, ctx, engine, _ = runtime
    engine.get_component.return_value = object()
    controller.start_controller(ctx)
    assert "must be an Assessor" in controller.system_panic.call_args.args[0]
    engine.register_aux_message_handler.assert_not_called()


def test_two_rounds_stamp_task_headers_and_cancel_incomplete_broadcasts(runtime, monkeypatch):
    controller, ctx, _, local_time = runtime
    monkeypatch.setattr(controller, "_monitor_task", Mock(return_value=subject.TaskDoneReason.ASSESSED_TASK_DONE))
    controller.control_flow(Signal(), ctx)
    assert controller.broadcast.call_count == 2
    for index, call in enumerate(controller.broadcast.call_args_list):
        task = call.kwargs["task"]
        assert task.name == "train"
        assert task.data.get_header(AppConstants.CURRENT_ROUND) == index
        assert task.data.get_header(AppConstants.NUM_ROUNDS) == 2
        assert task.data.get_header(EdgeTaskHeaderKey.TASK_SEQ) == index + 1
        assert task.data.get_header(EdgeTaskHeaderKey.UPDATE_INTERVAL) == 0.25
        assert task.data.get_cookie(AppConstants.CONTRIBUTION_ROUND) == index
        assert call.kwargs["targets"] == ["a", "b"]
        assert call.kwargs["min_responses"] == 2
    assert controller.cancel_task.call_count == 2
    assert controller.assessor.end_task.call_count == 2
    assert [c.args[0] for c in controller.fire_event.call_args_list] == [
        AppEventType.TRAINING_STARTED,
        AppEventType.ROUND_DONE,
        AppEventType.ROUND_DONE,
    ]
    assert ctx.get_prop(AppConstants.CURRENT_ROUND) == 1
    assert controller._wf_done and controller._current_task_seq == 0
    local_time.sleep.assert_called_once_with(1.25)
    controller.system_panic.assert_not_called()


@pytest.mark.parametrize("reason", [subject.TaskDoneReason.ABORTED, subject.TaskDoneReason.ASSESSED_WORKFLOW_DONE])
def test_task_monitor_stop_reason_ends_workflow_after_first_round(runtime, monkeypatch, reason):
    controller, ctx, _, _ = runtime
    monkeypatch.setattr(controller, "_monitor_task", Mock(return_value=reason))
    controller.control_flow(Signal(), ctx)
    assert controller.broadcast.call_count == 1
    controller.assessor.end_task.assert_called_once_with(ctx)
    assert AppEventType.ROUND_DONE not in [c.args[0] for c in controller.fire_event.call_args_list]
    assert controller._wf_done and controller._current_task_seq == 0


def test_preexisting_abort_does_not_start_or_broadcast_a_task(runtime):
    controller, ctx, _, _ = runtime
    signal = Signal()
    signal.trigger(True)
    controller.control_flow(signal, ctx)
    controller.assessor.start_task.assert_not_called()
    controller.broadcast.assert_not_called()
    assert controller._wf_done


@pytest.mark.parametrize("stage", ["start", "monitor", "end", "broadcast"])
def test_workflow_errors_panic_and_do_not_start_another_round(runtime, monkeypatch, stage):
    controller, ctx, _, _ = runtime
    monkeypatch.setattr(controller, "_monitor_task", Mock(return_value=subject.TaskDoneReason.ALL_CHILDREN_DONE))
    boundary = {
        "start": controller.assessor.start_task,
        "monitor": controller._monitor_task,
        "end": controller.assessor.end_task,
        "broadcast": controller.broadcast,
    }[stage]
    boundary.side_effect = RuntimeError("boundary failed")
    controller.control_flow(Signal(), ctx)
    controller.system_panic.assert_called_once()
    assert "boundary failed" in controller.system_panic.call_args.args[0]
    assert controller.assessor.start_task.call_count == 1


@pytest.mark.parametrize(
    "assessment, expected",
    [
        (Assessment.TASK_DONE, subject.TaskDoneReason.ASSESSED_TASK_DONE),
        (Assessment.WORKFLOW_DONE, subject.TaskDoneReason.ASSESSED_WORKFLOW_DONE),
    ],
)
def test_monitor_notifies_all_children_when_assessor_finishes(runtime, assessment, expected):
    controller, ctx, engine, _ = runtime
    controller._current_task_seq = 17
    controller.assessor.assess.side_effect = [assessment]
    result = controller._monitor_task(Task("train", Shareable()), ctx, Signal())
    assert result is expected
    request = engine.send_aux_request.call_args.kwargs
    assert request["targets"] == ["a", "b"]
    assert request["topic"] == "SAGE__train_end"
    assert request["request"].get_header(EdgeTaskHeaderKey.TASK_SEQ) == 17
    assert request["timeout"] == 0 and request["optional"] is True


def test_monitor_waits_once_then_aborts_without_notifying_children(runtime, monkeypatch):
    controller, ctx, engine, _ = runtime
    controller.assessor.assess.side_effect = [Assessment.CONTINUE]
    wait = Mock(side_effect=[WaiterRC.ABORTED])
    monkeypatch.setattr(subject, "conditional_wait", wait)
    signal = Signal()
    assert controller._monitor_task(Task("train", Shareable()), ctx, signal) is subject.TaskDoneReason.ABORTED
    wait.assert_called_once_with(waiter=None, timeout=0.5, abort_signal=signal)
    engine.send_aux_request.assert_not_called()


def test_completed_task_needs_no_assessment_or_cancellation(runtime, monkeypatch):
    controller, ctx, _, _ = runtime

    def completed(**kwargs):
        kwargs["task"].completion_status = TaskCompletionStatus.OK

    controller.broadcast.side_effect = completed
    controller.control_flow(Signal(), ctx)
    controller.assessor.assess.assert_not_called()
    controller.cancel_task.assert_not_called()
    assert controller._wf_done


@pytest.mark.parametrize("has_data", [False, True])
def test_result_is_consumed_and_only_update_payloads_reach_assessor(runtime, has_data):
    controller, ctx, _, _ = runtime
    result = Shareable()
    result.set_header(EdgeTaskHeaderKey.HAS_UPDATE_DATA, has_data)
    client_task = ClientTask(Client("child", "token"), Task("train", Shareable()))
    client_task.result = result
    controller._prepare_train_task_data(client_task, ctx)
    controller.fire_event_with_data.assert_called_once_with(
        AppEventType.BEFORE_TRAIN_TASK, ctx, AppConstants.TRAIN_SHAREABLE, client_task.task.data
    )
    controller._process_train_result(client_task, ctx)
    assert client_task.result is None
    assert controller.assessor.process_child_update.call_count == int(has_data)
    if has_data:
        controller.assessor.process_child_update.assert_called_once_with(result, ctx)


def test_error_result_panics_without_processing_update(runtime):
    controller, ctx, _, _ = runtime
    result = Shareable()
    result.set_return_code(ReturnCode.EXECUTION_EXCEPTION)
    client_task = ClientTask(Client("child", "token"), Task("train", Shareable()))
    client_task.result = result
    controller._process_train_result(client_task, ctx)
    assert client_task.result is None
    assert "child" in controller.system_panic.call_args.args[0]
    controller.assessor.process_child_update.assert_not_called()


def test_shutdown_logs_assessor_failure_and_late_results_only_during_workflow(runtime):
    controller, ctx, _, _ = runtime
    controller.assessor.finalize.side_effect = RuntimeError("finalization")
    controller.stop_controller(ctx)
    controller.log_exception.assert_called_once()
    client = Client("child", "token")
    controller.process_result_of_unknown_task(client, "train", "old", Shareable(), ctx)
    controller._wf_done = True
    controller.process_result_of_unknown_task(client, "train", "old", Shareable(), ctx)
    assert controller.log_warning.call_count == 1


def test_stats_require_collector_and_report_round_progress(runtime):
    controller, ctx, _, _ = runtime
    controller._current_round = 1
    collector = Mock(spec=GroupInfoCollector)
    ctx.set_prop(InfoCollector.CTX_KEY_STATS_COLLECTOR, collector)
    controller.handle_event(InfoCollector.EVENT_TYPE_GET_STATS, ctx)
    collector.add_info.assert_called_once_with(group_name=controller._name, info={"current_round": 1, "num_rounds": 2})
    ctx.set_prop(InfoCollector.CTX_KEY_STATS_COLLECTOR, object())
    with pytest.raises(TypeError, match="GroupInfoCollector"):
        controller.handle_event(InfoCollector.EVENT_TYPE_GET_STATS, ctx)


def test_update_report_preserves_protocol_reply_and_current_sequence(runtime, monkeypatch):
    controller, ctx, _, _ = runtime
    controller._current_task_seq = 12
    update, reply = Shareable(), Shareable({"accepted": True})
    process = Mock(return_value=(True, reply))
    monkeypatch.setattr(subject, "process_update_from_child", process)
    assert controller._process_update_report("train_update", update, ctx) is reply
    process.assert_called_once_with(
        processor=controller, update=update, update_f=controller._accept_update, current_task_seq=12, fl_ctx=ctx
    )
