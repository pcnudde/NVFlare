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

import threading
from unittest.mock import Mock

import pytest

from nvflare.fuel.f3.message import Message
from nvflare.fuel.f3.streaming.tools import receiver, sender
from tests.timing_utils import isolate_time


@pytest.mark.parametrize("outcome", ["success", "no-reply", "error"])
def test_sender_probe_waits_for_receiver_then_sends_buffer_and_stops_owned_cell(monkeypatch, outcome):
    cell = Mock()
    factory = Mock(return_value=cell)
    monkeypatch.setattr(sender, "Cell", factory)
    ready = Mock()
    monkeypatch.setattr(sender, "receiver_ready", ready)
    buffer = b"small test payload"
    monkeypatch.setattr(sender, "make_buffer", Mock(return_value=buffer))
    order = []
    ready.wait.side_effect = lambda: order.append("ready")

    def send(**kwargs):
        order.append("send")
        if outcome == "error":
            raise OSError("connection closed")
        return Message({sender.TIMESTAMP: 2}, "received") if outcome == "success" else None

    cell.send_request.side_effect = send
    clock = isolate_time(monkeypatch, sender)
    monkeypatch.setattr(clock, "time", Mock(side_effect=[1, 3]))
    sender.send_blob("tcp://server:1234", len(buffer))
    factory.assert_called_once_with(fqcn=sender.TX_CELL, root_url="tcp://server:1234", secure=False, credentials={})
    cell.core_cell.set_cell_connected_cb.assert_called_once_with(sender.cell_connected)
    cell.start.assert_called_once_with()
    assert order == ["ready", "send"]
    call = cell.send_request.call_args.kwargs
    assert call["target"] == sender.RX_CELL
    assert call["channel"] == sender.TEST_CHANNEL and call["topic"] == sender.TEST_TOPIC
    assert call["request"].payload is buffer
    assert call["request"].get_header(sender.TIMESTAMP) == 1
    cell.stop.assert_called_once_with()


def test_connected_callback_signals_only_its_owned_readiness_event(monkeypatch):
    ready = threading.Event()
    monkeypatch.setattr(sender, "receiver_ready", ready)
    peer = Mock()
    peer.get_fqcn.return_value = "receiver"
    assert not ready.is_set()
    sender.cell_connected(peer)
    assert ready.is_set()


def test_receiver_callback_acknowledges_byte_count_timestamp_and_completion_event(monkeypatch):
    event = threading.Event()
    monkeypatch.setattr(receiver, "event", event)
    clock = isolate_time(monkeypatch, receiver)
    monkeypatch.setattr(clock, "time", Mock(side_effect=[3, 4]))
    reply = receiver.request_cb(Message({receiver.TIMESTAMP: 1}, b"payload"))
    assert reply.payload == "Received 7 bytes"
    assert reply.get_header(receiver.TIMESTAMP) == 4
    assert event.is_set()


def test_receiver_probe_registers_callback_waits_for_completion_then_stops_cell(monkeypatch):
    cell = Mock()
    factory = Mock(return_value=cell)
    monkeypatch.setattr(receiver, "Cell", factory)
    event = Mock()
    monkeypatch.setattr(receiver, "event", event)
    order = []
    event.wait.side_effect = lambda: order.append("received")
    clock = isolate_time(monkeypatch, receiver)
    monkeypatch.setattr(clock, "sleep", lambda delay: order.append(("drain", delay)))
    cell.stop.side_effect = lambda: order.append("stop")
    receiver.receive_blob("tcp://server:1234")
    factory.assert_called_once_with(fqcn=receiver.RX_CELL, root_url="tcp://server:1234", secure=False, credentials={})
    cell.register_request_cb.assert_called_once_with(
        channel=receiver.TEST_CHANNEL, topic=receiver.TEST_TOPIC, cb=receiver.request_cb
    )
    assert order == ["received", ("drain", 1), "stop"]
    cell.start.assert_called_once_with()
