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

from nvflare.apis.fl_constant import FLContextKey, ReservedTopic
from nvflare.apis.fl_context import FLContext
from nvflare.metrics import metrics_publisher as subject
from nvflare.metrics.metrics_keys import METRICS_EVENT_TYPE, MetricKeys, MetricTypes


@pytest.mark.parametrize("stream", [True, False])
def test_metrics_use_exactly_one_destination_and_do_not_publish_type_as_value(stream, monkeypatch):
    comp, bus, engine, ctx, event_ctx = Mock(), Mock(), Mock(), FLContext(), FLContext()
    monkeypatch.setattr(ctx, "get_engine", lambda: engine)
    engine.new_context.side_effect = lambda: nullcontext(event_ctx)
    metrics, tags = {"duration": 2, MetricKeys.type: MetricTypes.GAUGE}, {"site": "one"}
    subject.collect_metrics(comp, stream, metrics, "task", tags, bus, ctx)
    if stream:
        bus.publish.assert_not_called()
        comp.fire_event.assert_called_once_with(event_type=METRICS_EVENT_TYPE, fl_ctx=event_ctx)
        data = event_ctx.get_prop(FLContextKey.EVENT_DATA)
        assert data["METRICS"] == {MetricKeys.metric_name: "task", MetricKeys.value: metrics, MetricKeys.tags: tags}
    else:
        comp.fire_event.assert_not_called()
        bus.publish.assert_called_once_with(
            [ReservedTopic.APP_METRICS],
            [
                {
                    MetricKeys.metric_name: "task_duration",
                    MetricKeys.value: 2,
                    MetricKeys.type: MetricTypes.GAUGE,
                    MetricKeys.tags: tags,
                }
            ],
        )
