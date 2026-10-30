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

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from nvflare.apis.dxo import DXO, DataKind
from nvflare.apis.fl_constant import ReservedKey
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import ReturnCode, Shareable, make_reply
from nvflare.apis.signal import Signal
from nvflare.app_common.abstract.aggregator import Aggregator
from nvflare.app_common.abstract.learnable import Learnable
from nvflare.app_common.app_constant import AppConstants
from nvflare.app_common.app_event_type import AppEventType
from nvflare.app_common.ccwf import swarm_client_ctl as subject
from nvflare.app_common.ccwf.common import Constant, NumberMetricComparator, ResultType
from tests.timing_utils import isolate_time


def gatherer(ctx=None, minimum=1, metric=None, mine=None):
    ctx = ctx or FLContext()
    data = Shareable()
    if metric is not None:
        data.set_header(Constant.METRIC, metric)
        data.set_header(Constant.CLIENT, "previous-best")
        data.set_header(Constant.ROUND, 1)
    executor = SimpleNamespace(best_metric=mine, best_round=2, me="site-1", update_status=Mock())
    aggregator = Mock(spec=Aggregator)
    aggregator.aggregate.return_value = DXO(DataKind.WEIGHTS, {"w": 8}).to_shareable()
    g = subject.Gatherer(
        data,
        ctx,
        2,
        executor,
        aggregator,
        NumberMetricComparator(),
        ["site-1", "site-2"],
        ["site-1", "site-2"],
        minimum,
        5,
        20,
    )
    g.fire_event = Mock()
    g.fire_event_with_data = Mock()
    return g


def contribution(round_num=2, rc=ReturnCode.OK):
    result = DXO(DataKind.WEIGHTS, {"w": 1}).to_shareable()
    result.set_header(AppConstants.CURRENT_ROUND, round_num)
    result.set_return_code(rc)
    return result


def test_submission_permissions_are_idempotent_bounded_and_released_after_gather():
    g = gatherer()
    request = contribution()
    assert g.can_accept_submission("unknown", request, g.fl_ctx) == ReturnCode.MODEL_UNRECOGNIZED
    assert g.can_accept_submission("site-1", request, g.fl_ctx) == ReturnCode.OK
    assert g.can_accept_submission("site-1", request, g.fl_ctx) == ReturnCode.OK
    assert g.can_accept_submission("site-2", request, g.fl_ctx) == ReturnCode.SERVICE_UNAVAILABLE
    assert g.gather("site-1", request, g.fl_ctx).get_return_code() == ReturnCode.OK
    assert not g.trainer_statuses["site-1"].busy
    assert g.can_accept_submission("site-2", request, g.fl_ctx) == ReturnCode.OK
    assert g.fl_ctx.get_prop(AppConstants.TRAINING_RESULT) is request
    assert g.fl_ctx.get_prop(AppConstants.CURRENT_ROUND) == 2
    assert g.fl_ctx.get_prop(AppConstants.AGGREGATION_ACCEPTED) is g.aggregator.accept.return_value


@pytest.mark.parametrize(
    "round_num,rc,name,expected",
    [
        (3, ReturnCode.OK, "site-1", ReturnCode.EXECUTION_EXCEPTION),
        (1, ReturnCode.OK, "site-1", ReturnCode.OK),
        (2, ReturnCode.ERROR, "site-1", ReturnCode.EXECUTION_EXCEPTION),
        (2, ReturnCode.OK, "unknown", ReturnCode.EXECUTION_EXCEPTION),
    ],
)
def test_gather_rejects_unexpected_clients_future_rounds_and_error_results(round_num, rc, name, expected):
    g = gatherer()
    result = contribution(round_num, rc)
    assert g.gather(name, result, g.fl_ctx).get_return_code() == expected
    assert g.aggregator.accept.called is (expected == ReturnCode.OK)
    if round_num == 1:
        assert g.trainer_statuses[name].reply_time is None
    if round_num == 3:
        g.executor.update_status.assert_called_once_with(action="gather", error=ReturnCode.EXECUTION_EXCEPTION)


def test_gatherer_contains_aggregator_exception_and_releases_submission_slot():
    g = gatherer()
    request = contribution()
    assert g.can_accept_submission("site-1", request, g.fl_ctx) == ReturnCode.OK
    g.aggregator.accept.side_effect = ValueError("invalid weights")
    assert g.gather("site-1", request, g.fl_ctx).get_return_code() == ReturnCode.EXECUTION_EXCEPTION
    assert not g.trainer_statuses["site-1"].busy
    assert g.min_resps_received_time is not None


@pytest.mark.parametrize(
    "prior,mine,best_client,best_metric,best_round",
    [
        (None, None, None, None, None),
        (None, 0.5, "site-1", 0.5, 2),
        (0.5, 0.7, "site-1", 0.7, 2),
        (0.7, 0.5, "previous-best", 0.7, 1),
        (0.7, None, "previous-best", 0.7, 1),
    ],
)
def test_aggregation_retains_global_best_or_promotes_better_local_result(
    prior, mine, best_client, best_metric, best_round
):
    g = gatherer(metric=prior, mine=mine)
    result = g.aggregate()
    assert result is g.aggregator.aggregate.return_value
    assert result.get_header(Constant.CLIENT) == best_client
    assert result.get_header(Constant.METRIC) == best_metric
    assert result.get_header(Constant.ROUND) == best_round
    g.fire_event.assert_called_once_with(AppEventType.BEFORE_AGGREGATION, g.fl_ctx)
    g.fire_event_with_data.assert_called_once_with(
        AppEventType.AFTER_AGGREGATION, g.fl_ctx, AppConstants.AGGREGATION_RESULT, result
    )


@pytest.mark.parametrize("mode", ["all", "timeout", "minimum", "pending"])
def test_gather_completion_obeys_full_quorum_deadline_and_post_quorum_grace(monkeypatch, mode):
    g = gatherer(minimum=1)
    clock = isolate_time(monkeypatch, subject)
    clock.time = lambda: 100
    g.start_time = 99 if mode != "timeout" else 0
    if mode == "all":
        for status in g.trainer_statuses.values():
            status.reply_time = 90
    elif mode == "minimum":
        g.min_resps_received_time = 90
    assert bool(g.is_done()) is (mode != "pending")


def test_out_of_range_minimum_defaults_to_all_trainers():
    for minimum in (0, 2, 3):
        assert gatherer(minimum=minimum).min_responses_required == 2


@pytest.fixture
def controller_runtime():
    ctl = subject.SwarmClientController(memory_gc_rounds=0)
    ctl.me = "site-1"
    ctl.engine = Mock()
    ctl.config = {
        Constant.CLIENTS: ["site-1", "site-2"],
        Constant.START_ROUND: 0,
        AppConstants.NUM_ROUNDS: 3,
        Constant.PRIVATE_P2P: False,
        Constant.RESULT_CLIENTS: ["site-2"],
    }
    ctl.broadcast_final_result = Mock()
    ctl.system_panic = Mock()
    ctl.shareable_generator = Mock()
    ctl.last_result = Learnable({"w": 2})
    ctl.last_round = 2
    ctl.best_result = Learnable({"w": 1})
    ctl.best_metric = 0.9
    ctl.best_round = 1
    ctx = FLContext()
    ctx.set_prop(ReservedKey.ENGINE, ctl.engine, private=True, sticky=True)
    peer = FLContext()
    peer.set_prop(ReservedKey.IDENTITY_NAME, "site-2", private=True, sticky=True)
    ctx.set_peer_context(peer)
    return ctl, ctx


@pytest.mark.parametrize(
    "mode,expected",
    [("late", ReturnCode.MODEL_UNRECOGNIZED), ("early", ReturnCode.SERVICE_UNAVAILABLE), ("current", ReturnCode.OK)],
)
def test_submission_request_distinguishes_old_rounds_and_pending_gatherer(controller_runtime, mode, expected):
    ctl, ctx = controller_runtime
    ctl.last_aggr_round_done = 1
    request = contribution(1 if mode == "late" else 2)
    if mode == "current":
        ctl.gatherer = gatherer(ctx)
    assert ctl._process_submission_request("request", request, ctx).get_return_code() == expected


@pytest.mark.parametrize(
    "mode,expected",
    [
        ("late", ReturnCode.OK),
        ("early", ReturnCode.EXECUTION_EXCEPTION),
        ("abort", ReturnCode.TASK_ABORTED),
        ("current", ReturnCode.OK),
        ("malformed", ReturnCode.EXECUTION_EXCEPTION),
    ],
)
def test_learn_result_handles_early_late_aborted_and_malformed_requests(controller_runtime, mode, expected):
    ctl, ctx = controller_runtime
    ctl.last_aggr_round_done = 1
    ctl.gatherer_waiter = Mock()
    request = contribution(1 if mode == "late" else 2)
    signal = Signal()
    if mode == "abort":
        signal.trigger(True)
    if mode == "current":
        ctl.gatherer = gatherer(ctx)
    if mode == "malformed":
        ctx.set_peer_context(None)
    reply = ctl._process_learn_result(request, ctx, signal)
    assert reply.get_return_code() == expected
    if mode == "early":
        ctl.gatherer_waiter.wait.assert_called_once_with(ctl.learn_task_abort_timeout)
        assert ctl.current_status.error == ReturnCode.EXECUTION_EXCEPTION


@pytest.mark.parametrize("owner", [None, "site-1", "site-2"])
def test_final_result_distribution_shares_best_owner_and_always_sends_last(controller_runtime, owner):
    ctl, ctx = controller_runtime
    ctl._ask_to_share_best_result = Mock()
    aggregate = Shareable()
    aggregate.set_header(Constant.CLIENT, owner)
    aggregate.set_header(Constant.METRIC, 0.9)
    ctl._distribute_final_results(aggregate, ctx)
    if owner == "site-1":
        ctl.broadcast_final_result.assert_any_call(
            ctx, ResultType.BEST, ctl.best_result, ctl.best_metric, ctl.best_round
        )
    elif owner:
        ctl._ask_to_share_best_result.assert_called_once_with(owner, 0.9, ctx)
    ctl.broadcast_final_result.assert_any_call(ctx, ResultType.LAST, ctl.last_result, round_num=2)


@pytest.mark.parametrize("has_best", [False, True])
def test_share_request_requires_best_model_and_updates_progress(controller_runtime, has_best):
    ctl, ctx = controller_runtime
    if not has_best:
        ctl.best_result = None
    reply = ctl._process_share_result("share", Shareable(), ctx)
    assert reply.get_return_code() == (ReturnCode.OK if has_best else ReturnCode.BAD_REQUEST_DATA)
    assert ctl.broadcast_final_result.called is has_best
    if has_best:
        assert ctl.current_status.action == "start_share_result_request_process"


@pytest.mark.parametrize("reply", [None, "invalid", make_reply(ReturnCode.ERROR), make_reply(ReturnCode.OK)])
def test_best_result_share_request_uses_nonsecure_control_message_and_contains_bad_ack(controller_runtime, reply):
    ctl, ctx = controller_runtime
    ctl.engine.send_aux_request.return_value = {"site-2": reply}
    ctl._ask_to_share_best_result("site-2", 0.9, ctx)
    args = ctl.engine.send_aux_request.call_args.kwargs
    assert args["targets"] == ["site-2"]
    assert args["timeout"] == ctl.final_result_ack_timeout
    assert args["secure"] is False


@pytest.mark.parametrize("fails", [False, True])
def test_monitor_clears_finished_gatherer_and_contains_end_round_failure(controller_runtime, monkeypatch, fails):
    ctl, ctx = controller_runtime
    ctl.gatherer = gatherer(ctx)
    completed = ctl.gatherer
    completed.is_done = Mock(return_value=True)
    ctl.gatherer_waiter.set()
    clock = isolate_time(monkeypatch, subject)
    clock.sleep = Mock()

    def finish(_):
        ctl.asked_to_stop = True
        if fails:
            raise ValueError("round failed")

    ctl._end_gather = Mock(side_effect=finish)
    ctl._monitor_gather()
    ctl._end_gather.assert_called_once_with(completed)
    assert ctl.gatherer is None and ctl.last_aggr_round_done == 2
    assert not ctl.gatherer_waiter.is_set()
    assert bool(ctl.current_status.error) is fails


def test_best_model_event_keeps_local_copy_and_ignores_other_clients(controller_runtime):
    ctl, ctx = controller_runtime
    original = Learnable({"w": [1]})
    ctx.set_prop(Constant.CLIENT, "site-2", private=True, sticky=False)
    ctx.set_prop(AppConstants.GLOBAL_MODEL, original, private=True, sticky=True)
    ctx.set_prop(AppConstants.VALIDATION_RESULT, 0.4, private=True, sticky=False)
    ctx.set_prop(AppConstants.CURRENT_ROUND, 4, private=True, sticky=False)
    ctl.handle_event(AppEventType.GLOBAL_BEST_MODEL_AVAILABLE, ctx)
    assert ctl.best_metric == 0.9
    ctx.set_prop(Constant.CLIENT, "site-1", private=True, sticky=False)
    ctl.handle_event(AppEventType.GLOBAL_BEST_MODEL_AVAILABLE, ctx)
    assert ctl.best_metric == 0.4 and ctl.best_round == 4
    original["w"][0] = 99
    assert ctl.best_result["w"] == [1]


@pytest.mark.parametrize("mode", ["no-aggregator", "no-trainers", "scatter-failed", "ok"])
def test_start_workflow_handles_unavailable_round_participants(controller_runtime, mode):
    ctl, ctx = controller_runtime
    ctl.trainers = [] if mode == "no-trainers" else ["site-2"]
    ctl.aggrs = [] if mode == "no-aggregator" else ["site-2"]
    if mode in ("scatter-failed", "ok"):
        ctl._scatter = Mock(return_value=mode == "ok")
    result = ctl.start_workflow(Shareable({"w": 1}), ctx, Signal())
    assert result.get_return_code() == (ReturnCode.OK if mode == "ok" else ReturnCode.EXECUTION_EXCEPTION)


@pytest.mark.parametrize(
    "mode",
    [
        "permission-missing",
        "permission-type",
        "permission-old",
        "upload-missing",
        "upload-type",
        "upload-error",
        "abort",
        "timeout",
        "learner-error",
        "busy-then-ok",
    ],
)
def test_remote_training_submission_propagates_permission_and_upload_failures(controller_runtime, monkeypatch, mode):
    ctl, ctx = controller_runtime
    ctl.is_trainer = True
    ctl._stamp_result_upload_receiver_ids = Mock()
    task = contribution()
    task.set_header(Constant.AGGREGATOR, "site-2")
    ctl._prepare_learn_task_data = Mock(return_value=(task, task))
    ctl.shareable_generator.shareable_to_learnable.return_value = Learnable({"w": 1})
    result = contribution(rc=ReturnCode.ERROR if mode == "learner-error" else ReturnCode.OK)
    ctl.execute_learn_task = Mock(return_value=result)
    ctl.fire_event = Mock()
    permission = {
        "permission-missing": None,
        "permission-type": "wrong",
        "permission-old": make_reply(ReturnCode.MODEL_UNRECOGNIZED),
    }.get(mode, make_reply(ReturnCode.OK))
    ctl.engine.send_aux_request.return_value = {"site-2": permission}
    upload = {"upload-missing": None, "upload-type": "wrong", "upload-error": make_reply(ReturnCode.ERROR)}.get(
        mode, make_reply(ReturnCode.OK)
    )
    ctl.broadcast_and_wait = Mock(return_value={"site-2": upload})
    signal = Signal()
    clock = isolate_time(monkeypatch, subject)
    clock.sleep = Mock()
    if mode == "abort":
        signal.trigger(True)
    elif mode == "timeout":
        ctl.request_to_submit_result_max_wait = 0.5
        ticks = iter(range(100))
        clock.time = lambda: next(ticks)
    elif mode == "busy-then-ok":
        ctl.engine.send_aux_request.side_effect = [
            {"site-2": make_reply(ReturnCode.SERVICE_UNAVAILABLE)},
            {"site-2": make_reply(ReturnCode.OK)},
        ]
    ctl.do_learn_task("train", task, ctx, signal)
    if mode.startswith("permission-") or mode in ("abort", "timeout", "learner-error"):
        ctl.broadcast_and_wait.assert_not_called()
    else:
        submitted = ctl.broadcast_and_wait.call_args.kwargs["task"]
        assert submitted.name == ctl.report_learn_result_task_name
        assert ctl.broadcast_and_wait.call_args.kwargs["targets"] == ["site-2"]
    if mode == "busy-then-ok":
        assert ctl.engine.send_aux_request.call_count == 2
        clock.sleep.assert_called_once_with(ctl.request_to_submit_result_interval)
        assert ctl.current_status.action == "finished_learn_task"
    elif mode not in ("abort", "timeout"):
        assert ctl.current_status.error
    assert ctx.get_prop(AppConstants.PROGRESS_OWNER) == "site-2"
