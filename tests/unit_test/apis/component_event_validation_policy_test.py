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

from nvflare.apis import fl_component as subject
from nvflare.apis.dxo import from_shareable
from nvflare.apis.event_type import EventType
from nvflare.apis.fl_constant import FLContextKey, LogMessageTag, ReservedKey
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable


@pytest.mark.parametrize(
    "event, context, error", [(1, FLContext(), TypeError), ("", FLContext(), ValueError), ("event", None, TypeError)]
)
def test_invalid_local_event_is_rejected_before_dispatch(event, context, error):
    with pytest.raises(error):
        subject.FLComponent().fire_event(event, context)


@pytest.mark.parametrize("data, context", [(Shareable(), None), ({}, FLContext())])
def test_federated_event_rejects_untyped_envelopes(data, context):
    with pytest.raises(TypeError):
        subject.FLComponent().fire_fed_event("event", data, context)


def test_critical_and_exception_logs_publish_tagged_analytics(monkeypatch):
    component = subject.FLComponent()
    component.logger = Mock()
    monkeypatch.setattr(subject, "secure_format_traceback", lambda: "safe traceback")
    ctx = FLContext()
    events = []

    def dispatch(kind, context):
        events.append((kind, from_shareable(context.get_prop(FLContextKey.EVENT_DATA)).data.copy()))

    ctx.set_prop(ReservedKey.ENGINE, SimpleNamespace(fire_event=dispatch))
    component.log_critical(ctx, "critical", fire_event=True)
    component.log_exception(ctx, "failure", fire_event=True)
    assert [kind for kind, _ in events] == [EventType.CRITICAL_LOG_AVAILABLE, EventType.EXCEPTION_LOG_AVAILABLE]
    assert events[0][1]["track_key"] == LogMessageTag.CRITICAL
    assert "critical" in events[0][1]["track_value"]
    assert events[1][1]["track_key"] == LogMessageTag.EXCEPTION
    assert events[1][1]["track_value"].endswith("failure\nsafe traceback")
    assert component.logger.critical.call_count == 1
    assert component.logger.error.call_count == 2
    # Context-free logging must not dispatch an event.
    component._fire_log_event("unused", "tag", "message", None)
    assert len(events) == 2


def test_duplicate_handler_registration_preserves_first_parameters():
    component = subject.FLComponent()
    handler = Mock()
    component.register_event_handler("event", handler, priority=1)
    component.register_event_handler(["event"], handler, priority=2)
    assert component.get_event_handlers() == {"event": [(handler, {"priority": 1})]}
    with pytest.raises(ValueError, match="event_types must be"):
        component.register_event_handler(1, handler)
