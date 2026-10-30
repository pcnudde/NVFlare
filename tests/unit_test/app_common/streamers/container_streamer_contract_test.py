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

from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import ReturnCode, Shareable, make_reply
from nvflare.apis.streaming import StreamableEngine, StreamContextKey
from nvflare.app_common.streamers import container_streamer as cs
from nvflare.app_common.streamers.container_retriever import ContainerRetriever


@pytest.mark.parametrize("container", [[1, 2], {"x": 1, "y": 2}, {1, 2}])
def test_entries_round_trip_preserves_container_type_and_completion(container):
    ctx = {cs._CTX_TYPE: f"builtins.{type(container).__name__}", cs._CTX_SIZE: len(container)}
    producer = cs._EntryProducer(container, 7)
    consumer = cs._EntryConsumerFactory().get_consumer(ctx, FLContext())
    for index in range(len(container)):
        item, timeout = producer.produce(ctx, FLContext())
        assert timeout == 7
        keep_going, reply = consumer.consume(item, ctx, FLContext())
        assert keep_going is (index < len(container) - 1)
        assert reply.get_return_code() == ReturnCode.OK
        assert producer.process_replies({"site": reply}, ctx, FLContext()) is (None if keep_going else True)
    consumer.finalize(ctx, FLContext())
    assert type(cs.ContainerStreamer.get_result(ctx)) is type(container)
    assert cs.ContainerStreamer.get_result(ctx) == container
    assert producer.produce(ctx, FLContext()) == (None, 0.0)


@pytest.mark.parametrize("container", [[], {}, set()])
def test_empty_container_cannot_begin_stream(container):
    with pytest.raises(ValueError, match="empty container"):
        cs._EntryProducer(container, 1)


def test_bad_entry_and_incomplete_size_are_rejected_without_publishing_result():
    ctx = {cs._CTX_TYPE: "builtins.dict", cs._CTX_SIZE: 2}
    consumer = cs._EntryConsumer(ctx)
    with pytest.raises(ValueError, match="Unable to add entry"):
        consumer.consume(Shareable({cs._KEY_ENTRY: 3}), ctx, FLContext())
    with pytest.raises(ValueError, match="does not match expected size"):
        consumer.consume(Shareable({cs._KEY_ENTRY: ("x", 1), cs._KEY_LAST: True}), ctx, FLContext())
    assert cs.ContainerStreamer.get_result(ctx) is None


@pytest.mark.parametrize("last", [False, True])
def test_any_failed_receiver_stops_producer_even_on_last_entry(last):
    producer = cs._EntryProducer([1], 1)
    producer.last = last
    assert (
        producer.process_replies({"good": make_reply(ReturnCode.OK), "bad": make_reply(ReturnCode.ERROR)}, {}, None)
        is False
    )


def test_stream_registration_and_send_forward_routing_and_security_options():
    engine = Mock(spec=StreamableEngine)
    engine.stream_objects.return_value = True
    ctx = Mock(spec=FLContext)
    ctx.get_engine.return_value = engine
    callback = Mock()
    cs.ContainerStreamer.register_stream_processing(ctx, "updates", "weights", callback, tag="batch")
    registered = engine.register_stream_processing.call_args.kwargs
    assert registered["channel"] == "updates"
    assert registered["topic"] == "weights"
    assert registered["stream_done_cb"] is callback
    assert registered["tag"] == "batch"
    stream_ctx = {"job": "one"}
    assert cs.ContainerStreamer.stream_container(
        "updates", "weights", stream_ctx, ["site"], [4], ctx, optional=True, secure=True
    )
    sent = engine.stream_objects.call_args.kwargs
    assert sent["stream_ctx"] is stream_ctx
    assert stream_ctx[cs._CTX_TYPE] == "list"
    assert stream_ctx[cs._CTX_SIZE] == 1
    assert sent["producer"].entry_timeout == 60
    assert sent["targets"] == ["site"]
    assert sent["optional"] and sent["secure"]


def test_non_streamable_engine_and_invalid_timeout_fail_before_send():
    ctx = Mock(spec=FLContext)
    ctx.get_engine.return_value = object()
    with pytest.raises(RuntimeError, match="StreamableEngine"):
        cs.ContainerStreamer.register_stream_processing(ctx, "c", "t")
    with pytest.raises(RuntimeError, match="StreamableEngine"):
        cs.ContainerStreamer.stream_container("c", "t", None, ["s"], [1], ctx)
    with pytest.raises(ValueError, match="entry_timeout must > 0"):
        cs.ContainerStreamer.stream_container("c", "t", None, ["s"], [1], ctx, entry_timeout=-1)


def test_retriever_validates_names_and_extracts_stream_result(monkeypatch):
    retriever = ContainerRetriever("models", stream_msg_optional=True, stream_msg_secure=True)
    container = {"w": [1, 2]}
    retriever.add_container("weights", container)
    for request in (Shareable(), Shareable({"name": "missing"})):
        assert retriever.validate_request(request, FLContext()) == (ReturnCode.BAD_REQUEST_DATA, None)
    assert retriever.validate_request(Shareable({"name": "weights"}), FLContext()) == (ReturnCode.OK, container)
    send = Mock()
    monkeypatch.setattr(cs.ContainerStreamer, "stream_container", send)
    retriever.do_stream("site", Shareable(), None, {"tx": 1}, container)
    assert send.call_args.kwargs["container"] is container
    assert send.call_args.kwargs["targets"] == ["site"]
    assert send.call_args.kwargs["optional"] and send.call_args.kwargs["secure"]
    stream_ctx = {StreamContextKey.RC: ReturnCode.OK, cs._CTX_RESULT: container}
    assert retriever.get_result(stream_ctx) == (ReturnCode.OK, container)
    register = Mock()
    monkeypatch.setattr(cs.ContainerStreamer, "register_stream_processing", register)
    retriever.register_stream_processing("channel", "topic", None, send, token=9)
    assert register.call_args.kwargs["stream_done_cb"] is send
    assert register.call_args.kwargs["token"] == 9
    retrieve = Mock(return_value=(ReturnCode.OK, container))
    monkeypatch.setattr(retriever, "retrieve", retrieve)
    assert retriever.retrieve_container("site", None, 7, "weights") == (ReturnCode.OK, container)
    retrieve.assert_called_once_with(from_site="site", fl_ctx=None, timeout=7, name="weights")
