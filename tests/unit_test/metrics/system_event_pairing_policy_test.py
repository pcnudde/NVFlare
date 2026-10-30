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

from unittest.mock import Mock

from nvflare.apis.event_type import EventType
from nvflare.apis.fl_context import FLContext
from nvflare.metrics import metrics_collector as base
from nvflare.metrics.metrics_keys import MetricKeys, MetricTypes
from nvflare.metrics.system_metrics_collector import SysMetricsCollector
from tests.timing_utils import isolate_time


def test_system_pair_records_one_duration_and_single_event_keeps_static_tags(monkeypatch):
    monkeypatch.setattr(base, "DataBus", Mock())
    collector = SysMetricsCollector({"job_id": "job", "site": "one"})
    publish = Mock()
    monkeypatch.setattr(collector, "publish_metrics", publish)
    clock = isolate_time(monkeypatch, base)
    monkeypatch.setattr(clock, "time", Mock(side_effect=[2.0, 7.0, 8.0, 9.0]))
    ctx = FLContext()
    for event in [EventType.SYSTEM_START, EventType.SYSTEM_END, EventType.CLIENT_DISCONNECTED, "unrelated"]:
        collector.handle_event(event, ctx)
    assert publish.call_count == 4
    duration = [call for call in publish.call_args_list if call.args[0][MetricKeys.type] == MetricTypes.GAUGE]
    assert len(duration) == 1
    assert duration[0].args == (
        {MetricKeys.time_taken: 5.0, MetricKeys.type: MetricTypes.GAUGE},
        "_system",
        collector.tags,
        ctx,
    )
    assert collector.event_start_time == {}
    assert collector.get_pair_events()[EventType.SYSTEM_START] == "_system"
    assert EventType.SYSTEM_START in collector.get_pair_start_events()
    assert EventType.CLIENT_DISCONNECTED in collector.get_single_events()
