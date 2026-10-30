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

from nvflare.apis.event_type import EventType
from nvflare.apis.fl_context import FLContext
from nvflare.widgets.comp_caller import ComponentCaller


def test_call_uses_engine_snapshot_and_fresh_context_even_if_run_ends_during_creation(monkeypatch):
    widget, ctx, call_ctx, engine = ComponentCaller(), FLContext(), FLContext(), Mock()
    monkeypatch.setattr(ctx, "get_engine", lambda: engine)
    assert widget.call_components("component", "reset", {}) is None
    widget.handle_event(EventType.START_RUN, ctx)

    def new_context():
        widget.handle_event(EventType.END_RUN, ctx)
        return nullcontext(call_ctx)

    engine.new_context.side_effect = new_context

    def event(event_type, fl_ctx):
        assert event_type == widget.EVENT_TYPE_CALL_COMPONENT and fl_ctx is call_ctx
        info = fl_ctx.get_prop(widget.CTX_KEY_CALL_INFO)
        assert (info.target, info.action, info.params) == ("component", "reset", {"round": 2})
        info.record_result("first")
        info.record_result("second", "rejected")

    engine.fire_event.side_effect = event
    assert widget.call_components("component", "reset", {"round": 2}) == {"first": "OK", "second": "rejected"}
    assert ctx.get_prop(widget.CTX_KEY_CALL_INFO) is None
    assert widget.engine is None and widget.call_components("component", "reset", {}) is None
    engine.fire_event.assert_called_once()
