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

from nvflare.apis.fl_constant import EventScope, FLContextKey
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.app_common.widgets.convert_to_fed_event import ConvertToFedEvent


@pytest.mark.parametrize("scenario", ["unselected", "already-federated", "missing", "wrong-type", "valid"])
def test_conversion_requires_selected_local_shareable_and_preserves_event_payload(monkeypatch, scenario):
    widget = ConvertToFedEvent(["metric"], fed_event_prefix="forward.")
    send, error = Mock(), Mock()
    monkeypatch.setattr(widget, "fire_fed_event", send)
    monkeypatch.setattr(widget, "log_error", error)
    ctx = FLContext()
    data = Shareable({"value": 0.25})
    if scenario != "missing":
        ctx.set_prop(FLContextKey.EVENT_DATA, "invalid" if scenario == "wrong-type" else data)
    if scenario == "already-federated":
        ctx.set_prop(FLContextKey.EVENT_SCOPE, EventScope.FEDERATION)
    widget.handle_event("other" if scenario == "unselected" else "metric", ctx)
    assert error.call_count == int(scenario in ("missing", "wrong-type"))
    assert send.call_count == int(scenario == "valid")
    if scenario == "valid":
        send.assert_called_once_with("forward.metric", data, ctx)
