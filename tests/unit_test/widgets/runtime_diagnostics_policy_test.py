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

from contextlib import nullcontext
from unittest.mock import Mock

import pytest

from nvflare.apis.analytix import AnalyticsData, AnalyticsDataType
from nvflare.apis.event_type import EventType
from nvflare.apis.fl_constant import EventScope, FLContextKey
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.apis.signal import Signal
from nvflare.widgets import fed_event as fed
from nvflare.widgets.info_collector import GroupInfoCollector, InfoCollector
from tests.timing_utils import isolate_time


@pytest.mark.parametrize(
    "event_type, key",
    [
        (EventType.CRITICAL_LOG_AVAILABLE, "critical"),
        (EventType.ERROR_LOG_AVAILABLE, "error"),
        (EventType.WARNING_LOG_AVAILABLE, "warning"),
        (EventType.EXCEPTION_LOG_AVAILABLE, "exception"),
    ],
)
def test_analytics_log_reports_are_grouped_by_origin_and_severity(event_type, key):
    collector = InfoCollector()
    ctx = FLContext()
    ctx.set_prop(FLContextKey.EVENT_ORIGIN, "trainer")
    ctx.set_prop(
        FLContextKey.EVENT_DATA, AnalyticsData("log", "failed update", AnalyticsDataType.TEXT).to_dxo().to_shareable()
    )
    collector.handle_event(event_type, ctx)
    assert collector.get_errors()["trainer"][key].endswith(": failed update")
    collector.reset_errors()
    assert collector.get_errors() == {}


def test_malformed_diagnostics_do_not_fire_recursive_logging_events(monkeypatch):
    collector = InfoCollector()
    error, exception = Mock(), Mock()
    monkeypatch.setattr(collector, "log_error", error)
    monkeypatch.setattr(collector, "log_exception", exception)
    ctx = FLContext()
    collector.handle_event(EventType.ERROR_LOG_AVAILABLE, ctx)
    assert error.call_args.kwargs["fire_event"] is False
    ctx.set_prop(FLContextKey.EVENT_DATA, Shareable({"invalid": "DXO"}))
    collector.handle_event(EventType.ERROR_LOG_AVAILABLE, ctx)
    assert exception.call_args.kwargs["fire_event"] is False
    assert collector.get_errors() is None


def test_run_stats_use_fresh_context_and_start_end_reset_state(monkeypatch):
    collector = InfoCollector()
    engine = Mock()
    ctx, stats_ctx = FLContext(), FLContext()
    monkeypatch.setattr(ctx, "get_engine", lambda: engine)
    engine.new_context.side_effect = lambda: nullcontext(stats_ctx)

    def report(event_type, fl_ctx):
        coll = fl_ctx.get_prop(InfoCollector.CTX_KEY_STATS_COLLECTOR)
        coll.add_info("controller", {"round": 2})
        coll.add_info("controller", {"rounds": 4})

    engine.fire_event.side_effect = report
    collector.add_info("old", "group", "value", 1)
    collector.handle_event(EventType.START_RUN, ctx)
    assert collector.categories == {}
    assert collector.get_run_stats() == {"controller": {"round": 2, "rounds": 4}}
    assert ctx.get_prop(InfoCollector.CTX_KEY_STATS_COLLECTOR) is None
    collector.handle_event(EventType.END_RUN, ctx)
    assert collector.get_run_stats() == {}


def test_category_replacement_and_missing_group_access():
    collector = InfoCollector()
    collector.add_info("stats", "worker", "first", 1)
    collector.add_info("stats", "worker", "second", 2)
    assert collector.get_group("stats", "worker") == {"first": 1, "second": 2}
    collector.set_info("stats", "worker", {"replacement": 3})
    assert collector.get_group("stats", "worker") == {"replacement": 3}
    collector.set_info("new", "group", {"value": 4})
    assert collector.get_category("new") == {"group": {"value": 4}}
    assert collector.get_group("missing", "missing") is None
    collector.reset_group("missing", "missing")
    collector.reset_all()
    group = GroupInfoCollector()
    group.set_info("task", {"count": 1})
    assert group.info == {"task": {"count": 1}}


def test_outgoing_federated_event_stamps_identity_and_ignores_incoming_echo(monkeypatch):
    runner = fed.FedEventRunner()
    send = Mock()
    monkeypatch.setattr(runner, "fire_and_forget_request", send)
    ctx = FLContext()
    monkeypatch.setattr(ctx, "get_identity_name", lambda: "site1")
    ctx.set_prop(FLContextKey.EVENT_SCOPE, EventScope.FEDERATION)
    data = Shareable({"metric": 3})
    data.set_header(fed.FedEventHeader.TARGETS, ["server"])
    ctx.set_prop(FLContextKey.EVENT_DATA, data)
    runner.handle_event("metric", ctx)
    assert data.get_header(fed.FedEventHeader.EVENT_TYPE) == "metric"
    assert data.get_header(fed.FedEventHeader.ORIGIN) == "site1"
    assert len(data.get_header(fed.FedEventHeader.EVENT_ID)) == 32
    send.assert_called_once_with(request=data, fl_ctx=ctx, targets=["server"], secure=False)
    data.set_header(fed.FedEventHeader.DIRECTION, "in")
    runner.handle_event("metric", ctx)
    assert send.call_count == 1


@pytest.mark.parametrize("fire_error", [False, True])
def test_event_poster_drains_queue_during_shutdown_with_bounded_waits(monkeypatch, fire_error):
    runner = fed.FedEventRunner(regular_interval=0.01, grace_period=0.1)
    warning = Mock()
    monkeypatch.setattr(runner, "log_warning", warning)
    runner.abort_signal = Signal()
    runner.asked_to_stop = True
    event = Shareable({"value": 3})
    event.set_header(fed.FedEventHeader.EVENT_TYPE, "metric")
    runner.in_events = [event]
    ctx, engine = FLContext(), Mock()
    engine.new_context.side_effect = lambda: nullcontext(ctx)
    if fire_error:
        engine.fire_event.side_effect = RuntimeError("listener stopped")
    runner.engine = engine
    local_time = isolate_time(monkeypatch, fed)
    pauses = []

    def sleep(interval):
        pauses.append(interval)
        assert len(pauses) <= 3, "event poster failed to finish after draining shutdown queue"

    monkeypatch.setattr(local_time, "sleep", sleep)
    runner._post()
    assert runner.in_events == []
    engine.fire_event.assert_called_once_with(event_type="metric", fl_ctx=ctx)
    assert ctx.get_prop(FLContextKey.EVENT_DATA) is event
    assert ctx.get_prop(FLContextKey.EVENT_SCOPE) == EventScope.FEDERATION
    assert pauses == [0.01, 0.0, 0.1]
    assert warning.call_count == (2 if fire_error else 1)


def test_readiness_waits_for_pending_events_and_quiet_period(monkeypatch):
    runner = fed.FedEventRunner(queue_empty_period=2.0)
    local_time = isolate_time(monkeypatch, fed)
    monkeypatch.setattr(local_time, "time", lambda: 10.0)
    runner.last_queue_empty_time = 9.0
    ctx = FLContext()
    runner.handle_event(EventType.CHECK_END_RUN_READINESS, ctx)
    assert ctx.get_prop(FLContextKey.NOT_READY_TO_END_RUN) is True
    runner.last_queue_empty_time = 5.0
    ready_ctx = FLContext()
    runner.handle_event(EventType.CHECK_END_RUN_READINESS, ready_ctx)
    assert ready_ctx.get_prop(FLContextKey.NOT_READY_TO_END_RUN) is None
    runner.in_events = [Shareable()]
    runner.handle_event(EventType.CHECK_END_RUN_READINESS, ready_ctx)
    assert ready_ctx.get_prop(FLContextKey.NOT_READY_TO_END_RUN) is True
