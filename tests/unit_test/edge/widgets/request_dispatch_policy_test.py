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
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from nvflare.apis.event_type import EventType
from nvflare.apis.fl_constant import FLContextKey, ReservedKey
from nvflare.apis.fl_context import FLContext
from nvflare.edge.constants import EdgeContextKey, EdgeEventType
from nvflare.edge.web.models.device_info import DeviceInfo
from nvflare.edge.web.models.job_request import JobRequest
from nvflare.edge.web.models.result_report import ResultReport
from nvflare.edge.web.models.selection_request import SelectionRequest
from nvflare.edge.web.models.task_request import TaskRequest
from nvflare.edge.widgets import etr, runner
from nvflare.fuel.f3.cellnet.defs import MessageHeaderKey
from nvflare.fuel.f3.message import Message


@pytest.mark.parametrize("outcome", ["reply", "exception-property", "missing", "raised"])
def test_cell_request_receiver_preserves_payload_and_rejects_incomplete_event_handling(monkeypatch, outcome):
    receiver, ctx, engine = etr.EdgeTaskReceiver(), FLContext(), Mock()
    engine.new_context.side_effect = lambda: nullcontext(ctx)
    monkeypatch.setattr(ctx, "get_engine", lambda: engine)
    monkeypatch.setattr(receiver, "log_error", Mock())
    receiver._handle_start_run(EventType.START_RUN, ctx)
    callbacks = engine.get_cell.return_value.register_request_cb.call_args_list
    assert len(callbacks) == 3 and len({c.kwargs["topic"] for c in callbacks}) == 3

    def process(event_type, context):
        assert event_type == "edge-event"
        assert context.get_prop(EdgeContextKey.REQUEST_FROM_EDGE) == {"weights": [1]}
        if outcome == "reply":
            context.set_prop(EdgeContextKey.REPLY_TO_EDGE, {"status": "OK"})
        elif outcome == "exception-property":
            context.set_prop(FLContextKey.EXCEPTIONS, ["handler failed"])
        elif outcome == "raised":
            raise ValueError("handler failed")

    engine.fire_event.side_effect = process
    reply = receiver._receive_edge_request(Message(payload={"weights": [1]}), "edge-event")
    assert reply.get_header(MessageHeaderKey.RETURN_CODE) == (
        etr.CellReturnCode.OK if outcome == "reply" else etr.CellReturnCode.PROCESS_EXCEPTION
    )
    if outcome == "reply":
        assert reply.payload == {"status": "OK"}


class FakeRunner(runner.SimulationRunner):
    def create_simulator(self, fl_ctx):
        return Mock()


@pytest.mark.parametrize(
    "kind,event",
    [
        ("task", EdgeEventType.EDGE_TASK_REQUEST_RECEIVED),
        ("selection", EdgeEventType.EDGE_SELECTION_REQUEST_RECEIVED),
        ("result", EdgeEventType.EDGE_RESULT_REPORT_RECEIVED),
        ("job", EdgeEventType.EDGE_JOB_REQUEST_RECEIVED),
    ],
)
def test_simulation_request_dispatches_by_typed_payload_and_returns_reply(monkeypatch, kind, event):
    widget, ctx, engine = FakeRunner(), FLContext(), Mock()
    engine.new_context.side_effect = lambda: nullcontext(ctx)
    device = DeviceInfo("device")
    request = {
        "task": TaskRequest(device, None, "job", {}),
        "selection": SelectionRequest(device, "job"),
        "result": ResultReport(device, None, "job", "task"),
        "job": JobRequest("name", device, None),
    }[kind]

    def process(event_type, context):
        assert event_type == event and context.get_prop(EdgeContextKey.REQUEST_FROM_EDGE) is request
        context.set_prop(EdgeContextKey.REPLY_TO_EDGE, {"accepted": True})

    monkeypatch.setattr(widget, "fire_event", process)
    assert widget._post_request(request, None, engine) == {"accepted": True}
    with pytest.raises(RuntimeError, match="invalid request type"):
        widget._post_request(object(), None, engine)


def test_simulation_runner_starts_only_on_leaf_and_stops_owned_simulator(monkeypatch):
    widget, ctx = FakeRunner(), FLContext()
    thread = Mock()
    monkeypatch.setattr(runner, "threading", SimpleNamespace(Thread=Mock(return_value=thread)))
    widget._sr_start_run(EventType.START_RUN, ctx)
    assert widget.simulator is None
    ctx.set_prop(ReservedKey.IS_LEAF, True)
    widget._sr_start_run(EventType.START_RUN, ctx)
    thread.start.assert_called_once_with()
    widget.simulator.set_send_func.assert_called_once_with(widget._post_request, engine=None)
    widget._run()
    widget.simulator.start.assert_called_once_with()
    widget._sr_end_run(EventType.END_RUN, ctx)
    widget.simulator.stop.assert_called_once_with()
