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
from nvflare.apis.shareable import ReservedHeaderKey, Shareable
from nvflare.apis.signal import Signal
from nvflare.app_common.app_constant import AppConstants
from nvflare.app_common.widgets import job_stats_reporter as subject
from tests.timing_utils import isolate_time


def test_unknown_payload_sizes_and_cycles_are_bounded_without_materializing_objects():
    class Unmeasurable:
        def __sizeof__(self):
            raise TypeError("not materialized")

    cycle = []
    cycle.append(cycle)
    assert subject._estimate_size_bytes(None) == 0
    assert subject._estimate_size_bytes(cycle) == 0
    assert subject._estimate_size_bytes(Unmeasurable()) == 0
    assert subject._round_from_shareable(object()) is None
    assert subject._round_sort_key("warmup") == (1, 0.0, "warmup")


def test_client_task_telemetry_owns_one_sampler_and_attaches_error_before_cleanup(monkeypatch):
    reporter = subject.JobStatsReporter()
    reporter._is_client = True
    sampler = Mock()
    sampler.stop.return_value = {"cpu": 1}
    factory = Mock(return_value=sampler)
    monkeypatch.setattr(subject, "_ResourceSampler", factory)
    clock = isolate_time(monkeypatch, subject)
    monkeypatch.setattr(clock, "monotonic", Mock(side_effect=range(10, 30)))
    ctx = FLContext()
    ctx.set_prop(FLContextKey.TASK_ID, "task")
    ctx.set_prop(FLContextKey.TASK_NAME, "train")
    ctx.set_prop(FLContextKey.TASK_DATA, Shareable())
    reporter._handle_client_task_received(ctx)
    reporter._handle_client_task_start(ctx)
    result = Shareable()
    result.set_header(ReservedHeaderKey.ERROR, "executor failed")
    ctx.set_prop(FLContextKey.TASK_RESULT, result)
    reporter._handle_client_task_completed(ctx)
    telemetry = result.get_header(AppConstants.JOB_STATS_CLIENT_TELEMETRY)
    assert telemetry["error"] == "executor failed" and telemetry["resources"] == {"cpu": 1}
    assert telemetry["execution_time"] >= 0 and telemetry["client_processing_time"] >= telemetry["execution_time"]
    sampler.start.assert_called_once_with()
    sampler.stop.assert_called_once_with()
    assert reporter._client_task_info == reporter._client_task_telemetry == {}
    reporter.handle_event(EventType.AFTER_SEND_TASK_RESULT, ctx)
    reporter.handle_event(EventType.END_RUN, ctx)
    assert sampler.stop.call_count == 1


def test_telemetry_dispatch_contains_local_failure_without_recursive_logging(monkeypatch):
    reporter = subject.JobStatsReporter()
    reporter._is_client = True
    monkeypatch.setattr(reporter, "_handle_client_task_start", Mock(side_effect=ValueError("sampler failed")))
    error = Mock()
    monkeypatch.setattr(reporter, "log_exception", error)
    reporter.handle_event(EventType.BEFORE_TASK_EXECUTION, FLContext())
    assert error.call_args.kwargs["fire_event"] is False


def test_rejected_contribution_creates_round_and_clears_only_transient_disconnect_error():
    reporter = subject.JobStatsReporter(collect_resources=False)
    ctx, peer = FLContext(), FLContext()
    peer.set_prop(ReservedKey.IDENTITY_NAME, "site")
    ctx.set_peer_context(peer)
    ctx.set_prop(AppConstants.CURRENT_ROUND, 4)
    ctx.set_prop(AppConstants.AGGREGATION_ACCEPTED, False)
    reporter._handle_contribution_accept(ctx)
    record = reporter._rounds[(reporter._workflow_seq, 4)]
    assert record.rejected_clients == {"site"} and record.accepted_clients == set()
    assert reporter._client_errors["site"] == ["CONTRIBUTION_REJECTED"]
    ctx.set_prop(FLContextKey.DISCONNECTED_CLIENT_NAME, "site")
    reporter._handle_disconnect(ctx)
    ctx.set_prop(FLContextKey.RECONNECTED_CLIENT_NAME, "site")
    reporter._handle_reconnect(ctx)
    assert reporter._client_errors["site"] == ["CONTRIBUTION_REJECTED"]
    assert "site" not in reporter._disconnected_clients


@pytest.mark.parametrize("failure", ["fatal", "aborted"])
def test_job_outcome_reports_fatal_context_or_run_abort(failure):
    reporter = subject.JobStatsReporter(collect_resources=False)
    ctx = FLContext()
    if failure == "fatal":
        ctx.set_prop(FLContextKey.FATAL_SYSTEM_ERROR, True)
    else:
        signal = Signal()
        signal.trigger(True)
        ctx.set_prop(FLContextKey.RUN_ABORT_SIGNAL, signal)
    status, reason = reporter._determine_status(ctx)
    assert status == subject.JobStatusCode.FAILURE
    assert reason == ("fatal system error" if failure == "fatal" else "job was aborted")
