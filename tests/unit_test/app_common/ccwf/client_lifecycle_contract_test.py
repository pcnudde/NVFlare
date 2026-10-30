# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

from unittest.mock import Mock

import pytest

from nvflare.apis.event_type import EventType
from nvflare.apis.fl_constant import FLContextKey, ReservedKey
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import ReturnCode, Shareable, make_reply
from nvflare.apis.signal import Signal
from nvflare.app_common.abstract.learnable import Learnable
from nvflare.app_common.abstract.learnable_persistor import LearnablePersistor
from nvflare.app_common.abstract.shareable_generator import ShareableGenerator
from nvflare.app_common.app_constant import AppConstants
from nvflare.app_common.app_event_type import AppEventType
from nvflare.app_common.ccwf import client_ctl as subject
from nvflare.app_common.ccwf.common import Constant, ResultType
from tests.timing_utils import isolate_time


class ClientController(subject.ClientSideController):
    def start_workflow(self, shareable, fl_ctx, abort_signal):
        self.start_payload = shareable
        return make_reply(ReturnCode.OK)

    def do_learn_task(self, name, data, fl_ctx, abort_signal):
        raise AssertionError("unit tests must not start a real learner loop")


@pytest.fixture
def runtime():
    ctl = ClientController("workflow")
    ctl.engine = Mock()
    ctl.me = "site-1"
    ctl.learn_thread = Mock()
    ctl.system_panic = Mock()
    ctl.fire_event = Mock()
    ctl.start_controller = Mock()
    ctl.persistor = Mock(spec=LearnablePersistor)
    ctl.shareable_generator = Mock(spec=ShareableGenerator)
    ctl.engine.get_component.side_effect = {
        ctl.persistor_id: ctl.persistor,
        ctl.shareable_generator_id: ctl.shareable_generator,
    }.get
    ctl.config = {
        FLContextKey.WORKFLOW: "wf-1",
        AppConstants.NUM_ROUNDS: 4,
        Constant.RESULT_CLIENTS: ["site-1", "site-2"],
        Constant.PRIVATE_P2P: False,
    }
    ctx = FLContext()
    ctx.set_prop(ReservedKey.ENGINE, ctl.engine, private=True, sticky=True)
    ctx.set_prop(ReservedKey.IDENTITY_NAME, ctl.me, private=True, sticky=True)
    ctx.set_prop(FLContextKey.RUNNER, Mock(), private=True, sticky=True)
    peer = FLContext()
    peer.set_prop(ReservedKey.IDENTITY_NAME, "site-2", private=True, sticky=True)
    ctx.set_peer_context(peer)
    return ctl, ctx


def test_start_resolves_components_and_initializes_before_launching_learner_boundary(runtime):
    ctl, ctx = runtime
    ctl.start_run(ctx)
    assert ctl.me == "site-1"
    assert ctl.learn_executor is ctx.get_prop(FLContextKey.RUNNER).find_executor.return_value
    ctl.learn_thread.start.assert_called_once()
    assert ctx.get_prop(Constant.EXECUTOR) is ctl
    ctl.fire_event.assert_called_once_with(Constant.EXECUTOR_INITIALIZED, ctx)
    ctl.system_panic.assert_not_called()


@pytest.mark.parametrize(
    "missing,reason",
    [
        ("engine", "no engine"),
        ("runner", "no client runner"),
        ("executor", "no executor"),
        ("persistor", "Persistor"),
        ("generator", "Shareable generator"),
    ],
)
def test_start_rejects_unavailable_dependencies_without_launch(runtime, missing, reason):
    ctl, ctx = runtime
    if missing == "engine":
        ctx.set_prop(ReservedKey.ENGINE, None, private=True, sticky=True)
    elif missing == "runner":
        ctx.set_prop(FLContextKey.RUNNER, None, private=True, sticky=True)
    elif missing == "executor":
        ctx.get_prop(FLContextKey.RUNNER).find_executor.return_value = None
    else:
        good = {ctl.persistor_id: ctl.persistor, ctl.shareable_generator_id: ctl.shareable_generator}
        good[ctl.persistor_id if missing == "persistor" else ctl.shareable_generator_id] = object()
        ctl.engine.get_component.side_effect = good.get
    ctl.start_run(ctx)
    assert reason in ctl.system_panic.call_args.args[0]
    ctl.learn_thread.start.assert_not_called()


def test_configure_and_start_tasks_load_model_and_preserve_workflow_routing(runtime):
    ctl, ctx = runtime
    model = Learnable({"w": 1})
    ctl.persistor.load.return_value = model
    config = dict(ctl.config)
    assert (
        ctl.execute(ctl.configure_task_name, Shareable({Constant.CONFIG: config}), ctx, Signal()).get_return_code()
        == ReturnCode.OK
    )
    assert ctl.workflow_id == "wf-1"
    assert ctx.get_prop(AppConstants.GLOBAL_MODEL) is model
    ctl.engine.register_aux_message_handler.assert_called_once()
    assert ctl.engine.register_aux_message_handler.call_args.kwargs["message_handle_func"] == ctl._process_end_workflow
    initial = Shareable({"w": 1})
    ctl.shareable_generator.learnable_to_shareable.return_value = initial
    assert ctl.execute(ctl.start_task_name, Shareable(), ctx, Signal()).get_return_code() == ReturnCode.OK
    assert ctl.is_starting_client and ctl.start_payload is initial
    assert ctl.execute("unknown", Shareable(), ctx, Signal()).get_return_code() == ReturnCode.TASK_UNKNOWN


def test_configuration_requires_workflow_identity_before_loading_model(runtime):
    ctl, ctx = runtime
    reply = ctl.execute(ctl.configure_task_name, Shareable({Constant.CONFIG: {"x": 1}}), ctx, Signal())
    assert reply.get_return_code() == ReturnCode.BAD_REQUEST_DATA
    ctl.persistor.load.assert_not_called()


@pytest.mark.parametrize("kind", [ResultType.BEST, ResultType.LAST])
@pytest.mark.parametrize("outcome", ["ok", "missing", "error", "exception", "no-targets"])
def test_final_result_broadcast_excludes_self_and_records_delivery_status(runtime, kind, outcome):
    ctl, ctx = runtime
    result = Learnable({"w": 4})
    replies = {"site-2": make_reply(ReturnCode.OK)}
    if outcome == "missing":
        replies = {}
    elif outcome == "error":
        replies["site-2"] = make_reply(ReturnCode.ERROR)
    elif outcome == "no-targets":
        ctl.config[Constant.RESULT_CLIENTS] = []
    ctl.broadcast_and_wait = Mock(
        return_value=replies, side_effect=ValueError("failed") if outcome == "exception" else None
    )
    ctl.broadcast_final_result(ctx, kind, result, metric=0.75, round_num=2)
    report = ctl.current_status
    assert report.all_done is (kind == ResultType.LAST)
    assert bool(report.error) is (outcome in ("missing", "error", "exception"))
    if outcome not in ("exception", "no-targets"):
        call = ctl.broadcast_and_wait.call_args.kwargs
        assert call["targets"] == ["site-2"]
        assert call["task"].data[Constant.RESULT] is result
        assert call["task"].data.get_header(Constant.METRIC) == 0.75
        assert call["task"].data.get_header(Constant.ROUND) == 2
    if outcome == "no-targets":
        ctl.broadcast_and_wait.assert_not_called()


@pytest.mark.parametrize("kind", [ResultType.BEST, ResultType.LAST])
def test_received_final_model_is_published_or_persisted_by_result_type(runtime, kind):
    ctl, ctx = runtime
    model = Learnable({"w": 5})
    request = Shareable({Constant.RESULT: model})
    request.set_header(Constant.RESULT_TYPE, kind)
    request.set_header(Constant.METRIC, 0.9)
    request.set_header(Constant.ROUND, 3)
    assert ctl._process_final_result(request, ctx).get_return_code() == ReturnCode.OK
    assert ctx.get_prop(AppConstants.GLOBAL_MODEL) is model
    if kind == ResultType.BEST:
        assert ctx.get_prop(Constant.ROUND) == 3
        assert ctx.get_prop(Constant.CLIENT) == "site-2"
        assert ctx.get_prop(AppConstants.VALIDATION_RESULT) == 0.9
        ctl.fire_event.assert_called_once_with(AppEventType.GLOBAL_BEST_MODEL_AVAILABLE, ctx)
        ctl.persistor.save.assert_not_called()
    else:
        ctl.persistor.save.assert_called_once_with(model, ctx)


@pytest.mark.parametrize(
    "kind,result", [("unknown", Learnable({"w": 1})), (ResultType.LAST, None), (ResultType.BEST, {"w": 1})]
)
def test_invalid_final_models_never_reach_persistence(runtime, kind, result):
    ctl, ctx = runtime
    request = Shareable({Constant.RESULT: result})
    request.set_header(Constant.RESULT_TYPE, kind)
    assert ctl._process_final_result(request, ctx).get_return_code() == ReturnCode.BAD_REQUEST_DATA
    ctl.persistor.save.assert_not_called()


@pytest.mark.parametrize("mode", ["free", "busy", "replace", "timeout"])
def test_learning_admission_handles_busy_replacement_and_bounded_abort(runtime, monkeypatch, mode):
    ctl, ctx = runtime
    first = Shareable({"w": 1})
    second = Shareable({"w": 2})
    assert ctl.set_learn_task(first, ctx)
    assert first.get_header(AppConstants.NUM_ROUNDS) == 4
    if mode == "free":
        assert ctl.learn_task.task_data is first
        return
    ctl.allow_busy_task = mode != "busy"
    if mode == "replace":

        def finish(_):
            ctl.learn_task = None

        ctl._abort_current_task = Mock(side_effect=finish)
    else:
        ctl._abort_current_task = Mock()
    if mode == "timeout":
        clock = isolate_time(monkeypatch, subject)
        clock.time = Mock(side_effect=[0, 100])
    accepted = ctl.set_learn_task(second, ctx)
    assert accepted is (mode == "replace")
    assert ctl.learn_task.task_data is (second if accepted else first)


def test_learn_request_failure_is_reported_and_success_is_enqueued(runtime):
    ctl, ctx = runtime
    assert ctl._process_learn_request(Shareable({"w": 1}), ctx).get_return_code() == ReturnCode.OK
    assert ctl.learn_task is not None
    assert ctl._process_learn_request(Shareable({"w": 2}), ctx).get_return_code() == ReturnCode.EXECUTION_EXCEPTION
    ctl.learn_task = None
    ctl.set_learn_task = Mock(side_effect=ValueError("bad task"))
    assert ctl._process_learn_request(Shareable(), ctx).get_return_code() == ReturnCode.EXECUTION_EXCEPTION
    assert ctl.current_status.error == ReturnCode.EXECUTION_EXCEPTION


@pytest.mark.parametrize("reply", [None, make_reply(ReturnCode.ERROR), make_reply(ReturnCode.OK)])
def test_send_learning_task_requires_successful_shareable_acknowledgement(runtime, reply):
    ctl, ctx = runtime
    ctl.broadcast_and_wait = Mock(return_value={"site-2": reply})
    request = Shareable({"w": 1})
    assert ctl.send_learn_task(["site-2"], request, ctx) is (
        isinstance(reply, Shareable) and reply.get_return_code() == ReturnCode.OK
    )
    task = ctl.broadcast_and_wait.call_args.kwargs["task"]
    assert task.name == ctl.do_learn_task_name
    assert task.data.get_header(AppConstants.NUM_ROUNDS) == 4
    assert task.timeout == int(ctl.learn_task_ack_timeout)


@pytest.mark.parametrize("fails", [False, True])
def test_learning_executor_propagates_round_and_cookies_on_success_or_failure(runtime, fails):
    ctl, ctx = runtime
    ctl.learn_executor = Mock()
    ctl.learn_executor.execute.side_effect = RuntimeError("failed") if fails else None
    ctl.learn_executor.execute.return_value = Shareable({"w": 2})
    data = Shareable({"w": 1})
    data.set_header(AppConstants.CURRENT_ROUND, 3)
    data.add_cookie("source", "server")
    result = ctl.execute_learn_task(data, ctx, Signal())
    assert result.get_header(AppConstants.CURRENT_ROUND) == 3
    assert result.get_cookie("source") == "server"
    assert result.get_cookie(AppConstants.CONTRIBUTION_ROUND) == 3
    assert result.get_return_code() == (ReturnCode.EXECUTION_EXCEPTION if fails else ReturnCode.OK)


def test_reporting_and_finalization_are_monotonic_and_idempotent(runtime):
    ctl, ctx = runtime
    ctl.workflow_id = "wf-1"
    assert ctl._get_status_report() is None
    ctl.update_status(last_round=3, action="train", all_done=True)
    ctl.update_status(last_round=1, action="save")
    assert ctl.current_status.last_round == 3 and ctl.current_status.all_done
    ctl.handle_event(EventType.BEFORE_PULL_TASK, ctx)
    assert ctx.get_prop(Constant.STATUS_REPORTS)["wf-1"][Constant.LAST_ROUND] == 3
    ctl._abort_current_task = Mock()
    assert ctl._process_end_workflow("end", Shareable(), ctx).get_return_code() == ReturnCode.OK
    assert ctl.workflow_done and ctl.asked_to_stop
    ctl.finalize(ctx)
    assert ctl.fire_event.call_args_list[-1].args == (Constant.EXECUTOR_FINALIZED, ctx)
    assert sum(c.args[0] == Constant.EXECUTOR_FINALIZED for c in ctl.fire_event.call_args_list) == 1


def test_record_result_saves_only_learnables_and_updates_round(runtime):
    ctl, ctx = runtime
    ctl.record_last_result(ctx, 1, {"w": 2})
    ctl.persistor.save.assert_not_called()
    result = Learnable({"w": 2})
    ctl.record_last_result(ctx, 3, result)
    assert ctl.last_result is result and ctl.last_round == 3
    assert ctx.get_prop(AppConstants.GLOBAL_MODEL) is result
    ctl.persistor.save.assert_called_once_with(result, ctx)
