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

from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import ReturnCode, Shareable, make_reply
from nvflare.apis.signal import Signal
from nvflare.apis.utils import reliable_message as subject
from tests.timing_utils import isolate_time


@pytest.mark.parametrize("scenario", ["ready-before", "ready-after", "status", "reply", "retry", "abort", "deadline"])
def test_reliable_send_admission_retries_are_bounded_and_result_ownership_is_preserved(monkeypatch, scenario):
    clock = isolate_time(monkeypatch, subject)
    ticks = []

    def now():
        ticks.append(True)
        assert len(ticks) <= 30, "reliable send failed to finish within a finite clock budget"
        return len(ticks) * 0.1

    monkeypatch.setattr(clock, "time", now)
    monkeypatch.setattr(clock, "sleep", Mock())
    cls = subject.ReliableMessage
    for name in ("debug", "info", "error"):
        monkeypatch.setattr(cls, name, Mock())
    monkeypatch.setattr(cls, "_query_interval", 0.3)
    result = make_reply(ReturnCode.OK)
    query = Mock(return_value=result)
    monkeypatch.setattr(cls, "_query_result", query)
    ctx, engine, signal = FLContext(), Mock(), Signal()
    monkeypatch.setattr(ctx, "get_engine", lambda: engine)
    receiver = subject._ReplyReceiver("tx", 0.1, 0.4 if scenario == "deadline" else 10)
    receiver.tx_start_time = 0
    status = make_reply(ReturnCode.OK)
    status.set_header(subject.HEADER_STATUS, subject.STATUS_IN_PROCESS)
    direct = make_reply(ReturnCode.OK)
    direct.set_header(subject.HEADER_OP, subject.OP_REPLY)
    if scenario == "ready-before":
        receiver.result_ready.set()

    def send(**kwargs):
        assert kwargs["targets"] == ["peer"] and kwargs["topic"] == subject.TOPIC_RELIABLE_REQUEST
        if scenario == "ready-after":
            receiver.result_ready.set()
        if scenario == "abort":
            signal.trigger(True)
        if scenario in ("deadline", "abort") or (scenario == "retry" and engine.send_aux_request.call_count == 1):
            return {"peer": make_reply(ReturnCode.COMMUNICATION_ERROR)}
        return {"peer": direct if scenario in ("reply", "retry") else status}

    engine.send_aux_request.side_effect = send
    reply = cls._send_request("peer", Shareable(), signal, ctx, receiver)
    if scenario == "deadline":
        assert reply.get_return_code() == ReturnCode.COMMUNICATION_ERROR
        query.assert_not_called()
    elif scenario == "abort":
        assert reply.get_return_code() == ReturnCode.TASK_ABORTED
        query.assert_not_called()
    elif scenario in ("reply", "retry"):
        assert reply is direct
        assert engine.send_aux_request.call_count == (2 if scenario == "retry" else 1)
        query.assert_not_called()
    else:
        assert reply is result
        query.assert_called_once_with("peer", signal, ctx, receiver)
    if scenario == "ready-before":
        engine.send_aux_request.assert_not_called()
