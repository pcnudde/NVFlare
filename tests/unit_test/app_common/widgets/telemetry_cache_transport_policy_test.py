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

from nvflare.apis.event_type import EventType
from nvflare.apis.fl_constant import FLContextKey
from nvflare.apis.fl_context import FLContext
from nvflare.app_common.widgets import job_stats_reporter as subject


def test_client_send_completion_and_end_run_release_owned_sampler_state():
    reporter, context = subject.JobStatsReporter(), FLContext()
    reporter._is_client = True
    context.set_prop(FLContextKey.TASK_ID, "task")
    sampler = Mock()
    reporter._client_task_info["task"] = {"sampler": sampler}
    reporter._client_task_telemetry["task"] = {"cached": True}
    reporter.handle_event(EventType.AFTER_SEND_TASK_RESULT, context)
    sampler.stop.assert_called_once_with()
    assert reporter._client_task_info == reporter._client_task_telemetry == {}
    reporter.handle_event(EventType.END_RUN, context)
    assert reporter._client_task_info == reporter._client_task_telemetry == {}


def test_concurrent_finalizer_uses_existing_telemetry_after_sampler_release():
    reporter = subject.JobStatsReporter()
    sampler, cached = Mock(), {"execution_time": 3}
    sampler.stop.side_effect = lambda: reporter._client_task_telemetry.setdefault("task", cached)
    reporter._client_task_info["task"] = {"sampler": sampler, "execution_start_time": 1}
    assert reporter._finalize_client_task("task") is cached
    assert reporter._finalize_client_task("task") is cached
    sampler.stop.assert_called_once_with()
    assert reporter._client_task_info["task"]["sampler"] is None


def test_battery_summary_preserves_explicit_plugged_state():
    reporter = subject.JobStatsReporter()
    reporter._completed_tasks = [
        SimpleNamespace(
            client="site", resources={"battery_start_percent": 80, "battery_end_percent": 70, "battery_plugged": True}
        )
    ]
    summary = reporter._build_resource_summary()
    assert summary["site"]["battery_used_percent"] == 10
    assert summary["site"]["battery_plugged"] is True


def test_transport_stats_support_legacy_histograms_without_lock_and_empty_pool_registry(monkeypatch):
    monkeypatch.setattr(subject.StatsPoolManager, "pools", {})
    assert subject.JobStatsReporter._get_transport_comm_stats() == {}
    pool = Mock(spec=subject.HistPool)
    pool.update_lock = None
    pool.cat_bins = {"site": [SimpleNamespace(total=4.5, count=2)]}
    monkeypatch.setattr(subject.StatsPoolManager, "pools", {"sent_msg_sizes": pool})
    result = subject.JobStatsReporter._get_transport_comm_stats()
    assert result["total_sent_mb"] == 4.5 and result["item_count_sent"] == 2
    assert result["total_received_mb"] == 0
