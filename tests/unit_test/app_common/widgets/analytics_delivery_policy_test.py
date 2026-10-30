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

from nvflare.apis.analytix import ANALYTIC_EVENT_TYPE, AnalyticsDataType, LogWriterName, TrackConst
from nvflare.apis.dxo import from_shareable
from nvflare.apis.event_type import EventType
from nvflare.apis.fl_constant import EventScope, FLContextKey, ReservedKey
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.app_common.widgets import streaming as subject


class RecordingReceiver(subject.AnalyticsReceiver):
    def __init__(self):
        super().__init__()
        self.initialize = Mock()
        self.save = Mock()
        self.finalize = Mock()
        self.log_error = Mock()

    def initialize(self, fl_ctx):
        pass

    def save(self, fl_ctx, shareable, record_origin):
        pass

    def finalize(self, fl_ctx):
        pass


def analytic_context():
    ctx = FLContext()
    ctx.set_prop(ReservedKey.IDENTITY_NAME, "local", private=False, sticky=True)
    return ctx


@pytest.mark.parametrize("scope", [EventScope.LOCAL, EventScope.FEDERATION])
def test_receiver_stores_local_and_federated_origins_then_drops_post_end_data(scope):
    receiver, ctx = RecordingReceiver(), analytic_context()
    payload = Shareable()
    payload.set_peer_props({ReservedKey.IDENTITY_NAME: "remote"})
    ctx.set_prop(FLContextKey.EVENT_DATA, payload, private=True, sticky=False)
    ctx.set_prop(FLContextKey.EVENT_SCOPE, scope, private=True, sticky=False)
    receiver.handle_event(ANALYTIC_EVENT_TYPE, ctx)
    receiver.save.assert_not_called()
    receiver.handle_event(EventType.START_RUN, ctx)
    receiver.handle_event(ANALYTIC_EVENT_TYPE, ctx)
    receiver.save.assert_called_once_with(
        shareable=payload, fl_ctx=ctx, record_origin="local" if scope == EventScope.LOCAL else "remote"
    )
    receiver.handle_event(EventType.END_RUN, ctx)
    receiver.finalize.assert_called_once_with(ctx)
    receiver.handle_event(ANALYTIC_EVENT_TYPE, ctx)
    assert receiver.save.call_count == 1


@pytest.mark.parametrize("failure", ["initialize", "missing-data", "wrong-data", "missing-origin", "save", "finalize"])
def test_receiver_contains_invalid_data_and_extension_failures(failure):
    receiver, ctx = RecordingReceiver(), analytic_context()
    if failure == "initialize":
        receiver.initialize.side_effect = RuntimeError("initialize")
    receiver.handle_event(EventType.START_RUN, ctx)
    if failure == "initialize":
        assert not receiver._initialized
    else:
        payload = Shareable()
        if failure != "missing-data":
            ctx.set_prop(FLContextKey.EVENT_DATA, 7 if failure == "wrong-data" else payload, private=True, sticky=False)
        if failure == "missing-origin":
            ctx.set_prop(FLContextKey.EVENT_SCOPE, EventScope.FEDERATION, private=True, sticky=False)
        if failure == "save":
            receiver.save.side_effect = RuntimeError("save")
        if failure == "finalize":
            receiver.finalize.side_effect = RuntimeError("finalize")
            receiver.handle_event(EventType.END_RUN, ctx)
        else:
            receiver.handle_event(ANALYTIC_EVENT_TYPE, ctx)
    receiver.log_error.assert_called_once()
    assert receiver.log_error.call_args.kwargs["fire_event"] is False


def test_sender_publishes_analytic_dxo_in_engine_context_and_closes(monkeypatch):
    sender = subject.AnalyticsSender(event_type="custom", writer_name=LogWriterName.TORCH_TB)
    ctx = analytic_context()
    engine = Mock()
    engine.new_context.return_value = ctx
    ctx.set_prop(ReservedKey.ENGINE, engine, private=True, sticky=True)
    send = Mock()
    monkeypatch.setattr(subject, "send_analytic_dxo", send)
    sender.handle_event(EventType.ABOUT_TO_START_RUN, ctx)
    sender.add("loss", 0.25, AnalyticsDataType.SCALAR, global_step=3)
    assert sender.get_writer_name() == LogWriterName.TORCH_TB
    call = send.call_args
    assert call.args == (sender,)
    assert call.kwargs["event_type"] == "custom"
    assert from_shareable(call.kwargs["dxo"].to_shareable()).data[TrackConst.TRACK_KEY] == "loss"
    assert call.kwargs["dxo"].data[TrackConst.GLOBAL_STEP_KEY] == 3
    with pytest.raises(TypeError, match="global step"):
        sender.add("loss", 1, AnalyticsDataType.SCALAR, global_step="3")
    assert send.call_count == 1
    sender.close()
    sender.close()
    assert sender.engine is None
