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

from nvflare.apis.event_type import EventType
from nvflare.apis.fl_constant import ReservedKey
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.app_common.abstract.aggregator import Aggregator
from nvflare.app_common.aggregators import lazy as subject


@pytest.mark.parametrize(
    "kwargs,error", [({"aggregator_id": None}, TypeError), ({"aggregator_id": "real", "accept_timeout": 0}, ValueError)]
)
def test_invalid_aggregator_configuration_is_rejected(kwargs, error):
    with pytest.raises(error):
        subject.LazyAggregator(**kwargs)


def test_queued_contribution_owns_context_snapshot_and_busy_round_rejects_new_work():
    wrapper = subject.LazyAggregator("real")
    ctx = FLContext()
    ctx.set_prop("round", 1, private=False, sticky=False)
    data = Shareable({"weight": 4})
    assert wrapper.accept(data, ctx)
    ctx.set_prop("round", 2, private=False, sticky=False)
    queued = wrapper.contributions.get_nowait()
    assert queued.data is data
    assert queued.fl_ctx.get_prop("round") == 1
    wrapper.aggregating = True
    assert wrapper.accept(Shareable({"weight": 5}), ctx) is False
    assert wrapper.contributions.empty()


@pytest.mark.parametrize("component", [None, object(), subject.LazyAggregator("nested")])
def test_start_rejects_missing_wrong_or_recursive_aggregators(component):
    wrapper = subject.LazyAggregator("real")
    wrapper.system_panic = Mock()
    engine = Mock()
    engine.get_component.return_value = component
    ctx = FLContext()
    ctx.set_prop(ReservedKey.ENGINE, engine, private=True, sticky=True)
    wrapper._lazy_aggr_start_run(EventType.START_RUN, ctx)
    wrapper.system_panic.assert_called_once()
    assert wrapper.aggregator is None


def test_start_schedules_accept_boundary_and_end_marks_shutdown(monkeypatch):
    wrapper = subject.LazyAggregator("real")
    real = Mock(spec=Aggregator)
    engine = Mock()
    engine.get_component.return_value = real
    ctx = FLContext()
    ctx.set_prop(ReservedKey.ENGINE, engine, private=True, sticky=True)
    thread = Mock()
    factory = Mock(return_value=thread)
    monkeypatch.setattr(subject, "threading", SimpleNamespace(Thread=factory))
    wrapper._lazy_aggr_start_run(EventType.START_RUN, ctx)
    assert wrapper.aggregator is real
    assert factory.call_args.kwargs == {"target": wrapper._do_accept, "daemon": True}
    thread.start.assert_called_once()
    assert wrapper._check_end_run() == subject._AcceptWaitRC.OK
    wrapper._lazy_aggr_end_run(EventType.END_RUN, ctx)
    assert wrapper._check_end_run() == subject._AcceptWaitRC.END_RUN


@pytest.mark.parametrize("fails", [False, True])
def test_accept_loop_drains_pending_work_and_signals_round_completion_without_threads(fails):
    wrapper = subject.LazyAggregator("real")
    wrapper.aggregator = Mock(spec=Aggregator)
    ctx = FLContext()
    data = Shareable({"weight": 1})
    wrapper.accept(data, ctx)
    wrapper.aggregating = True

    def accept(item, item_ctx):
        assert item is data
        assert item_ctx is not ctx
        wrapper.run_ended = True
        if fails:
            raise ValueError("rejected update")
        return True

    wrapper.aggregator.accept.side_effect = accept
    wrapper._do_accept()
    wrapper.aggregator.accept.assert_called_once()
    assert wrapper.contributions.empty()
    assert wrapper.accept_done.is_set()


@pytest.mark.parametrize(
    "rc",
    [
        subject._AcceptWaitRC.IS_SET,
        subject._AcceptWaitRC.TIMEOUT,
        subject._AcceptWaitRC.ABORTED,
        subject._AcceptWaitRC.END_RUN,
    ],
)
def test_aggregate_respects_wait_outcome_and_resets_successful_round(monkeypatch, rc):
    wrapper = subject.LazyAggregator("real", accept_timeout=7)
    wrapper.aggregator = Mock(spec=Aggregator)
    result = Shareable({"weight": 8})
    wrapper.aggregator.aggregate.return_value = result
    ctx = FLContext()
    wrapper.accept(Shareable({"weight": 1}), ctx)
    wait = Mock(return_value=rc)
    monkeypatch.setattr(subject, "conditional_wait", wait)
    reply = wrapper.aggregate(ctx)
    assert wait.call_args.kwargs["timeout"] == 7
    assert wait.call_args.kwargs["condition_cb"] == wrapper._check_end_run
    if rc in (subject._AcceptWaitRC.ABORTED, subject._AcceptWaitRC.END_RUN):
        assert isinstance(reply, Shareable)
        wrapper.aggregator.aggregate.assert_not_called()
    else:
        assert reply is result
        wrapper.aggregator.reset.assert_called_once_with(ctx)
        assert not wrapper.aggregating
        assert not wrapper.accept_done.is_set()
        assert wrapper.contributions.empty()


def test_explicit_reset_clears_queue_and_accept_state():
    wrapper = subject.LazyAggregator("real")
    wrapper.aggregator = Mock(spec=Aggregator)
    ctx = FLContext()
    wrapper.accept(Shareable(), ctx)
    wrapper.aggregating = True
    wrapper.accept_done.set()
    wrapper.reset(ctx)
    assert wrapper.contributions.empty()
    assert wrapper.contributions.unfinished_tasks == 0
    assert not wrapper.accept_done.is_set()
    wrapper.aggregator.reset.assert_called_once_with(ctx)
