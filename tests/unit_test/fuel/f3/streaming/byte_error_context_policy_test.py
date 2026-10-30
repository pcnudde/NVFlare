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

from collections import OrderedDict
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from nvflare.fuel.f3.cellnet.defs import MessageHeaderKey, ReturnCode
from nvflare.fuel.f3.message import Message
from nvflare.fuel.f3.streaming import byte_streamer as module


@pytest.fixture
def streamer(monkeypatch):
    monkeypatch.setattr(module.ByteStreamer, "tx_task_map", {})
    monkeypatch.setattr(module.ByteStreamer, "error_context_map", OrderedDict())
    monkeypatch.setattr(module, "time", SimpleNamespace(monotonic=lambda: 0))
    subject = module.ByteStreamer.__new__(module.ByteStreamer)
    subject.cell = Mock()
    subject.cell.my_info.fqcn = "sender"
    subject.error_callbacks = []
    subject.chunk_size = 1024
    return subject


def test_retry_payload_snapshots_are_immutable_and_account_for_scatter_gather_buffers():
    first, second = bytearray(b"a"), bytearray(b"bc")
    snapshot = module._snapshot_payload([first, memoryview(second)])
    assert snapshot == [b"a", b"bc"] and module._payload_size(snapshot) == 3
    first[:] = b"z"
    assert snapshot == [b"a", b"bc"]
    assert module._payload_size(None) == 0 and module._snapshot_payload(None) is None


def test_error_callbacks_continue_after_failure_and_invalid_callback_is_rejected(streamer, caplog):
    with pytest.raises(module.StreamError, match="not callable"):
        streamer.register_error_callback(7)
    first, second = Mock(side_effect=ValueError("callback failed")), Mock()
    streamer.register_error_callback(first)
    streamer.register_error_callback(second)
    message = Message()
    streamer._notify_error_callbacks(message)
    second.assert_called_once_with(message)
    assert "callback failed" in caplog.text
    assert streamer.get_chunk_size() == 1024


@pytest.mark.parametrize("missing", [MessageHeaderKey.ORIGIN, MessageHeaderKey.DESTINATION, None])
def test_forward_error_requires_peer_identity_and_keeps_stream_request_correlation(streamer, missing, caplog):
    caplog.set_level("DEBUG", logger=module.log.name)
    headers = {
        MessageHeaderKey.OPTIONAL: True,
        MessageHeaderKey.ORIGIN: "sender",
        MessageHeaderKey.DESTINATION: "target",
        module.StreamHeaderKey.STREAM_ID: "stream",
        module.StreamHeaderKey.STREAM_REQ_ID: "request",
    }
    if missing:
        headers.pop(missing)
    streamer.cell.fire_and_forget.return_value = {"sender": ReturnCode.TARGET_UNREACHABLE}
    streamer._forward_error_handler(Message(headers=headers), ReturnCode.TARGET_UNREACHABLE)
    assert streamer.cell.fire_and_forget.call_count == int(missing is None)
    if missing is None:
        reply = streamer.cell.fire_and_forget.call_args.args[3]
        assert reply.get_header(module.StreamHeaderKey.STREAM_REQ_ID) == "request"
        assert reply.get_header(module.StreamHeaderKey.ERROR_TYPE) == module.StreamTargetUnreachable.__name__
        assert "failed to report stream routing error" in caplog.text


def test_late_error_context_cache_discards_oldest_record_at_capacity(monkeypatch, streamer):
    monkeypatch.setattr(module, "MAX_STREAM_ERROR_CONTEXTS", 2)
    for identifier in ["first", "second", "third"]:
        task = SimpleNamespace(
            sid=identifier, cell=streamer.cell, target="target", channel="app", topic="topic", headers={}
        )
        module.ByteStreamer._retain_error_context(task)
    assert list(module.ByteStreamer.error_context_map) == ["second", "third"]


@pytest.mark.parametrize("optional", [True, False])
def test_transmit_failure_always_notifies_task_stop_with_contextual_error(optional, caplog):
    caplog.set_level("DEBUG", logger=module.log.name)
    task = Mock(optional=optional)
    task.send_loop.side_effect = ValueError("read failed")
    module.ByteStreamer._transmit_task(task)
    error, notify = task.stop.call_args.args
    assert isinstance(error, module.StreamError) and "read failed" in str(error) and notify
    assert "Error while sending" in caplog.text


@pytest.mark.parametrize("mode", ["foreign_task", "foreign_context", "wrong_context", "valid"])
def test_stream_error_is_correlated_to_own_cell_and_exact_receiver_context_before_stopping(streamer, mode, caplog):
    task = SimpleNamespace(
        cell=object() if mode == "foreign_task" else streamer.cell,
        target="receiver",
        channel="app",
        topic="topic",
        headers={module.StreamHeaderKey.STREAM_REQ_ID: "request"},
        stop=Mock(),
    )
    if mode == "foreign_context":
        module.ByteStreamer.error_context_map["stream"] = module._TxTaskContext(
            object(), "receiver", "app", "topic", "request", 10
        )
    else:
        module.ByteStreamer.tx_task_map["stream"] = task
    message = Message(
        headers={
            MessageHeaderKey.ORIGIN: "wrong" if mode == "wrong_context" else "receiver",
            module.StreamHeaderKey.STREAM_ID: "stream",
            module.StreamHeaderKey.CHANNEL: "app",
            module.StreamHeaderKey.TOPIC: "topic",
            module.StreamHeaderKey.STREAM_REQ_ID: "request",
        }
    )
    callback = Mock()
    streamer.register_error_callback(callback)
    streamer._error_handler(message)
    assert task.stop.call_count == callback.call_count == int(mode == "valid")
    if mode != "valid":
        assert "Ignored" in caplog.text
