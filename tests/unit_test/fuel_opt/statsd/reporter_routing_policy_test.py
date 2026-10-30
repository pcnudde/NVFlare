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

import pytest

pytest.importorskip("datadog")

from nvflare.apis.fl_constant import ReservedTopic
from nvflare.fuel_opt.statsd import statsd_reporter as subject
from nvflare.metrics.metrics_keys import MetricKeys, MetricTypes


@pytest.fixture
def reporter(monkeypatch):
    bus, client, initialize = Mock(), Mock(), Mock()
    monkeypatch.setattr(subject, "DataBus", Mock(return_value=bus))
    monkeypatch.setattr(subject, "statsd", client)
    monkeypatch.setattr(subject, "initialize", initialize)
    obj = subject.StatsDReporter("site", host="metrics", port=8125)
    monkeypatch.setattr(obj, "logger", Mock())
    return obj, client, bus, initialize


def test_counter_and_gauge_keep_values_and_convert_tags_without_mutating_input(reporter):
    obj, client, bus, initialize = reporter
    initialize.assert_called_once_with(statsd_host="metrics", statsd_port=8125)
    bus.subscribe.assert_called_once_with([ReservedTopic.APP_METRICS], obj.process_metrics)
    tags = {"site": "a", "round": 3}
    obj.process_metrics(
        ReservedTopic.APP_METRICS,
        [
            {
                MetricKeys.metric_name: "count",
                MetricKeys.value: 2,
                MetricKeys.type: MetricTypes.COUNTER,
                MetricKeys.tags: tags,
            },
            {MetricKeys.metric_name: "load", MetricKeys.value: 0.25, MetricKeys.type: MetricTypes.GAUGE},
        ],
        bus,
    )
    client.increment.assert_called_once_with("count", 2, tags=["site:a", "round:3"])
    client.gauge.assert_called_once_with("load", 0.25, tags=[])
    assert tags == {"site": "a", "round": 3}
    obj.logger.warning.assert_not_called()


def test_unsubscribed_and_unsupported_metric_types_cannot_emit_counter_or_gauge(reporter):
    obj, client, bus, _ = reporter
    obj.process_metrics("unrelated", [{MetricKeys.type: MetricTypes.COUNTER}], bus)
    obj.process_metrics(
        ReservedTopic.APP_METRICS,
        [
            {MetricKeys.type: MetricTypes.HISTOGRAM},
            {MetricKeys.type: MetricTypes.SUMMARY},
            {MetricKeys.type: "unknown"},
        ],
        bus,
    )
    assert client.method_calls == []
    obj.logger.warning.assert_called_once()
    assert "Unknown metric type" in obj.logger.warning.call_args.args[0]


def test_one_bad_sink_call_is_contained_and_warns(reporter):
    obj, client, bus, _ = reporter
    client.increment.side_effect = OSError("sink unavailable")
    obj.process_metrics(
        ReservedTopic.APP_METRICS,
        [{MetricKeys.type: MetricTypes.COUNTER, MetricKeys.metric_name: "n", MetricKeys.value: 1}],
        bus,
    )
    obj.logger.warning.assert_called_once()
    assert "sink unavailable" in obj.logger.warning.call_args.args[0]
