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

from nvflare.apis.executor import Executor
from nvflare.apis.fl_constant import FLContextKey, ReservedKey
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import ReturnCode, Shareable, make_reply
from nvflare.apis.signal import Signal
from nvflare.app_common.app_constant import AppConstants
from nvflare.edge.constants import EdgeTaskHeaderKey
from nvflare.edge.executors import hug
from nvflare.edge.executors.hug import HierarchicalUpdateGatherer, TaskInfo
from nvflare.edge.updater import Updater
from nvflare.fuel.utils.tree_utils import Forest, Node
from nvflare.fuel.utils.waiter_utils import WaiterRC
from tests.timing_utils import isolate_time


@pytest.fixture
def runtime(monkeypatch):
    clock = isolate_time(monkeypatch, hug)
    clock.time = Mock(return_value=101)
    ctx, peer = FLContext(), FLContext()
    ctx.set_prop(ReservedKey.IDENTITY_NAME, "me")
    peer.set_prop(ReservedKey.IDENTITY_NAME, "child")
    ctx.set_peer_context(peer)
    ctx.set_prop(ReservedKey.ENGINE, Mock())
    gatherer = HierarchicalUpdateGatherer("", "updater", 3)
    gatherer._updater = Mock(spec=Updater)
    gatherer._updater.prepare_update_for_parent.return_value = Shareable({"w": 2})
    gatherer._updater.process_child_update.return_value = (True, Shareable({"accepted": True}))
    gatherer._pending_task = task()
    gatherer._task_start_time = 100
    gatherer._parent_name = "parent"
    return gatherer, ctx


def task():
    data = Shareable({"weights": {"w": 1}})
    for key, value in (
        (ReservedKey.TASK_ID, "t1"),
        (ReservedKey.TASK_NAME, "train"),
        (AppConstants.CURRENT_ROUND, 4),
        (EdgeTaskHeaderKey.TASK_SEQ, 7),
    ):
        data.set_header(key, value)
    data.set_cookie_jar({"token": "round-4"})
    return TaskInfo(data)


@pytest.mark.parametrize("parent", [False, True])
def test_startup_resolves_hierarchy_components_and_parent_route(runtime, parent):
    gatherer, ctx = runtime
    ancestor = Node(SimpleNamespace(name="parent"), None) if parent else None
    node = Node(SimpleNamespace(name="me"), ancestor)
    node.children.append(Node(SimpleNamespace(name="child"), node))
    forest = Forest()
    forest.nodes["me"] = node
    ctx.set_prop(FLContextKey.CLIENT_HIERARCHY, forest)
    ctx.get_engine().get_component.return_value = gatherer._updater
    gatherer._hug_handle_start_run("start", ctx)
    assert gatherer._children == ["child"] and gatherer._num_children == 1
    assert gatherer._parent_name == ("parent" if parent else None)


@pytest.mark.parametrize("bad", ["updater", "learner", "forest", "node", "fallback"])
def test_startup_panics_on_unusable_components_or_hierarchy(runtime, bad):
    gatherer, ctx = runtime
    updater, learner = Mock(spec=Updater), Mock(spec=Executor)
    components = {"updater": updater, "learner": learner}
    if bad in components:
        components[bad] = object()
    gatherer.learner_id = "learner"
    ctx.get_engine().get_component.side_effect = components.get
    forest = Forest()
    if bad != "node":
        forest.nodes["me"] = Node(SimpleNamespace(name="me"), None)
    ctx.set_prop(FLContextKey.CLIENT_HIERARCHY, None if bad == "forest" else forest)
    if bad == "fallback":
        gatherer.updater_id = ""
    gatherer.system_panic = Mock()
    gatherer._hug_handle_start_run("start", ctx)
    gatherer.system_panic.assert_called_once()


@pytest.mark.parametrize("fails", [False, True])
def test_leaf_execution_calls_only_configured_learner_and_translates_failure(runtime, fails):
    gatherer, ctx = runtime
    ctx.set_prop(ReservedKey.IS_LEAF, True)
    gatherer._learner = Mock(spec=Executor)
    gatherer._learner.execute.return_value = Shareable({"trained": True})
    if fails:
        gatherer._learner.execute.side_effect = ValueError("training failed")
    payload, abort = Shareable(), Signal()
    result = gatherer.execute("train", payload, ctx, abort)
    gatherer._learner.execute.assert_called_once_with("train", payload, ctx, abort)
    assert result.get_return_code() == (ReturnCode.EXECUTION_EXCEPTION if fails else ReturnCode.OK)
    if not fails:
        assert result["trained"] is True
    ctx.get_engine().register_aux_message_handler.assert_not_called()


def test_execute_marks_task_ready_before_waiting_and_clears_round_state_once(runtime):
    gatherer, ctx = runtime
    payload = task().task
    gatherer._updater.start_task.return_value = Shareable({"model": 2})
    gatherer.task_started, gatherer.task_ended = Mock(), Mock()

    def do_task(fl_ctx, abort):
        assert payload.get_header(ReservedKey.TASK_IS_READY) is True
        assert gatherer._pending_task.task == gatherer._updater.start_task.return_value
        gatherer._pending_clients["child"] = 100
        gatherer._task_done = True
        return Shareable({"result": 3})

    gatherer._do_task = Mock(side_effect=do_task)
    for _ in range(2):
        assert gatherer.execute("train", payload, ctx, Signal())["result"] == 3
        assert gatherer._pending_task is None and gatherer._task_start_time is None
        assert gatherer._pending_clients == {} and not gatherer._task_done and not gatherer._process_error
    assert ctx.get_engine().register_aux_message_handler.call_count == 2
    assert gatherer._updater.end_task.call_count == 2
    assert gatherer.task_started.call_count == gatherer.task_ended.call_count == 2


@pytest.mark.parametrize(
    "mode", ["missing", "wrong_type", "done", "error", "parent_moved", "all_children", "abort", "wait_done"]
)
def test_collection_loop_terminates_for_parent_protocol_completion_and_cancellation(runtime, monkeypatch, mode):
    gatherer, ctx = runtime
    reply = make_reply(ReturnCode.OK)
    replies = {"parent": reply}
    if mode == "missing":
        replies = {}
    elif mode == "wrong_type":
        replies = {"parent": "invalid"}
    elif mode == "done":
        gatherer._task_done = True
    elif mode == "error":
        gatherer._process_error = True
    elif mode == "parent_moved":
        reply.set_header(EdgeTaskHeaderKey.TASK_SEQ, 8)
    elif mode == "all_children":
        gatherer._num_children = 1
        gatherer._pending_clients["child"] = 100
    ctx.get_engine().send_aux_request.return_value = replies
    wait = Mock(return_value=WaiterRC.ABORTED if mode == "abort" else WaiterRC.IS_SET)
    monkeypatch.setattr(hug, "conditional_wait", wait)
    result = gatherer._do_task(ctx, Signal())
    expected = (
        ReturnCode.EXECUTION_EXCEPTION
        if mode in ("missing", "wrong_type", "error")
        else (ReturnCode.TASK_ABORTED if mode == "abort" else ReturnCode.OK)
    )
    assert result.get_return_code() == expected
    if mode in ("done", "error"):
        ctx.get_engine().send_aux_request.assert_not_called()
    else:
        ctx.get_engine().send_aux_request.assert_called_once()
    if mode in ("abort", "wait_done"):
        gatherer._updater.process_parent_update_reply.assert_called_once_with(reply, ctx)
        wait.assert_called_once()


@pytest.mark.parametrize("has_update", [False, True])
def test_update_reports_preserve_round_cookie_sequence_and_payload_flag(runtime, has_update):
    gatherer, ctx = runtime
    gatherer._updater.prepare_update_for_parent.return_value = Shareable({"w": 5}) if has_update else None
    report = gatherer._make_update_report(gatherer._pending_task, ctx)
    assert report.get_header(EdgeTaskHeaderKey.HAS_UPDATE_DATA) is has_update
    assert report.get_header(EdgeTaskHeaderKey.TASK_SEQ) == 7
    assert report.get_cookie_jar() == {"token": "round-4"}
    assert ctx.get_prop(AppConstants.CURRENT_ROUND) == 4
    if has_update:
        assert report["w"] == 5


def test_updater_exception_is_contained_and_marks_processing_error(runtime):
    gatherer, ctx = runtime
    gatherer._updater.prepare_update_for_parent.side_effect = ValueError("invalid")
    assert gatherer._prepare_update(ctx) is None
    assert gatherer._process_error
    gatherer._updater.process_child_update.side_effect = ValueError("invalid")
    assert gatherer._accept_child_update(Shareable(), ctx, 6) == (False, None)
    assert ctx.get_prop(AppConstants.CURRENT_ROUND) == 6


@pytest.mark.parametrize("seq,ends", [(6, False), (7, True), (8, True)])
def test_parent_end_notification_is_forwarded_without_ending_a_newer_task(runtime, seq, ends):
    gatherer, ctx = runtime
    gatherer._children, gatherer._num_children = ["child"], 1
    req = Shareable()
    req.set_header(EdgeTaskHeaderKey.TASK_SEQ, seq)
    assert gatherer._process_task_end("end", req, ctx).get_return_code() == ReturnCode.OK
    assert gatherer._task_done is ends
    forwarded = ctx.get_engine().send_aux_request.call_args.kwargs
    assert forwarded["targets"] == ["child"] and forwarded["timeout"] == 0 and forwarded["optional"] is True
    assert forwarded["request"].get_header(EdgeTaskHeaderKey.TASK_SEQ) == seq
    assert gatherer._check_task_done() == (WaiterRC.IS_SET if ends else None)


def test_task_identity_gates_subclass_updates_and_completion(runtime):
    gatherer, ctx = runtime
    update = Shareable({"w": 4})
    assert not gatherer.accept_update("wrong", update, ctx)
    assert not gatherer.set_task_done("wrong", ctx)
    gatherer._updater.process_child_update.assert_not_called()
    assert gatherer.accept_update("t1", update, ctx)
    assert gatherer.set_task_done("t1", ctx)
    gatherer._pending_task = None
    assert not gatherer.accept_update("t1", update, ctx)
    assert not gatherer.set_task_done("t1", ctx)


def test_get_current_task_refreshes_state_but_preserves_task_after_refresh_failure(runtime):
    gatherer, ctx = runtime
    current = gatherer._pending_task
    gatherer._updater.get_current_state.return_value = Shareable({"latest": 2})
    assert gatherer.get_current_task(ctx) is current and current.task["latest"] == 2
    gatherer._updater.get_current_state.side_effect = ValueError("offline")
    assert gatherer.get_current_task(ctx) is current and current.task["latest"] == 2
    gatherer._pending_task = None
    assert gatherer.get_current_task(ctx) is None
    gatherer._updater = None
    assert gatherer.get_current_task(ctx) is None


@pytest.mark.parametrize("matches,has_update", [(False, False), (True, False), (True, True)])
def test_child_results_only_update_status_and_accept_matching_task(runtime, matches, has_update):
    gatherer, ctx = runtime
    result = make_reply(ReturnCode.OK)
    result.set_header(ReservedKey.TASK_ID, "t1" if matches else "old")
    result.set_header(EdgeTaskHeaderKey.HAS_UPDATE_DATA, has_update)
    ctx.set_prop(FLContextKey.TASK_RESULT, result)
    gatherer._handle_result_received("result", ctx)
    assert ctx.get_prop(FLContextKey.EVENT_PROCESSED) is True
    assert gatherer._pending_clients_status() == ((1, 1) if matches else (0, 0))
    assert gatherer._updater.process_child_update.called is (matches and has_update)
    if matches and has_update:
        assert ctx.get_prop(AppConstants.CURRENT_ROUND) == 4


def test_task_sent_tracks_only_current_assignments_and_child_updates_preserve_reply(runtime):
    gatherer, ctx = runtime
    ctx.set_prop(FLContextKey.TASK_ID, "old")
    gatherer._handle_task_sent("sent", ctx)
    assert gatherer._pending_clients_status() == (0, 0)
    ctx.set_prop(FLContextKey.TASK_ID, "t1")
    gatherer._handle_task_sent("sent", ctx)
    assert gatherer._pending_clients_status() == (0, 1)
    update = Shareable({"w": 3})
    update.set_header(EdgeTaskHeaderKey.TASK_SEQ, 7)
    update.set_header(EdgeTaskHeaderKey.HAS_UPDATE_DATA, True)
    reply = gatherer._process_child_update("updates", update, ctx)
    assert reply["accepted"] is True and reply.get_header(EdgeTaskHeaderKey.TASK_SEQ) == 7
