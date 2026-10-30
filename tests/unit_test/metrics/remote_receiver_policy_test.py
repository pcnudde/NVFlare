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

from nvflare.apis.fl_constant import EventScope, FLContextKey, ReservedKey
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.metrics import remote_metrics_receiver as subject
from nvflare.metrics.metrics_keys import METRICS_EVENT_TYPE, MetricKeys


@pytest.fixture
def receiver(monkeypatch):
    bus, publish = Mock(), Mock()
    monkeypatch.setattr(subject, "DataBus", Mock(return_value=bus))
    monkeypatch.setattr(subject, "publish_app_metrics", publish)
    obj = subject.RemoteMetricsReceiver()
    monkeypatch.setattr(obj, "log_error", Mock())
    return obj, bus, publish


@pytest.mark.parametrize("federated", [False, True])
def test_metrics_payload_is_published_only_for_known_local_or_federated_origin(receiver, federated, monkeypatch):
    obj, bus, publish = receiver
    ctx = FLContext()
    monkeypatch.setattr(ctx, "get_identity_name", lambda: "site-one")
    ctx.set_prop(FLContextKey.EVENT_SCOPE, EventScope.FEDERATION if federated else EventScope.LOCAL)
    metrics, tags = [{"latency": 2}], {"task": "train"}
    data = Shareable({"METRICS": {MetricKeys.metric_name: "timing", MetricKeys.value: metrics, MetricKeys.tags: tags}})
    data.set_peer_props({ReservedKey.IDENTITY_NAME: "remote"})
    ctx.set_prop(FLContextKey.EVENT_DATA, data)
    obj.handle_event(METRICS_EVENT_TYPE, ctx)
    publish.assert_called_once_with(metrics=metrics, metric_name="timing", tags=tags, data_bus=bus)
    obj.log_error.assert_not_called()


@pytest.mark.parametrize("fault", ["no-event", "bad-type", "no-origin", "no-metrics"])
def test_bad_remote_metrics_are_rejected_without_recursive_log_events(receiver, monkeypatch, fault):
    obj, _, publish = receiver
    ctx = FLContext()
    monkeypatch.setattr(ctx, "get_identity_name", lambda: None if fault == "no-origin" else "site")
    if fault != "no-event":
        data = [] if fault == "bad-type" else Shareable()
        ctx.set_prop(FLContextKey.EVENT_DATA, data)
    obj.handle_event(METRICS_EVENT_TYPE, ctx)
    publish.assert_not_called()
    obj.log_error.assert_called_once()
    assert obj.log_error.call_args.kwargs["fire_event"] is False


def test_custom_event_filter_ignores_unsubscribed_event(receiver):
    obj, _, publish = receiver
    obj.events = ["selected"]
    obj.handle_event(METRICS_EVENT_TYPE, FLContext())
    publish.assert_not_called()
    obj.log_error.assert_not_called()
