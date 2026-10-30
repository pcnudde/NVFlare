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

from nvflare.apis.fl_constant import ReservedKey, ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.apis.signal import Signal
from nvflare.apis.streaming import ConsumerFactory, ObjectConsumer, ObjectProducer, StreamContextKey
from nvflare.private import stream_runner as module


@pytest.fixture
def streamer(monkeypatch):
    monkeypatch.setattr(module, "ThreadPoolExecutor", Mock(return_value=Mock()))
    subject = module.ObjectStreamer(Mock())
    subject.logger = Mock()
    yield subject
    subject.shutdown()


@pytest.mark.parametrize("stage", ["finalize", "done", "return"])
def test_consumer_completion_contains_each_cleanup_failure_and_still_returns_consumer(stage):
    factory, consumer, logger = Mock(spec=ConsumerFactory), Mock(spec=ObjectConsumer), Mock()
    if stage == "finalize":
        consumer.finalize.side_effect = ValueError("finalize failed")
    if stage == "return":
        factory.return_consumer.side_effect = ValueError("return failed")

    def done(context, fl_context, **kwargs):
        if stage == "done":
            raise ValueError("done failed")
        return {"complete": True}

    context = {}
    info = module._ConsumerInfo(logger, context, factory, consumer, done, None, {})
    result = info.stream_done(ReturnCode.OK, FLContext())
    assert result == (None if stage == "done" else {"complete": True})
    assert context[StreamContextKey.RC] == (ReturnCode.EXECUTION_EXCEPTION if stage == "finalize" else ReturnCode.OK)
    factory.return_consumer.assert_called_once()
    assert stage + " failed" in logger.error.call_args.args[0]


@pytest.mark.parametrize(
    "mode, code",
    [
        ("abort_before", ReturnCode.TASK_ABORTED),
        ("unregistered", ReturnCode.EXECUTION_EXCEPTION),
        ("missing_initial", ReturnCode.BAD_REQUEST_DATA),
        ("missing_context", ReturnCode.BAD_REQUEST_DATA),
        ("consume_error", ReturnCode.EXECUTION_EXCEPTION),
        ("abort_during", ReturnCode.TASK_ABORTED),
        ("empty_reply", ReturnCode.OK),
    ],
)
def test_incoming_stream_errors_do_not_retain_transaction_or_owned_consumer(streamer, mode, code):
    context, abort = FLContext(), Signal()
    context.set_prop(ReservedKey.RUN_ABORT_SIGNAL, abort)
    factory, consumer = Mock(spec=ConsumerFactory), Mock(spec=ObjectConsumer)
    factory.get_consumer.return_value = consumer

    def consume(*args):
        if mode == "consume_error":
            raise ValueError("consumer failed")
        if mode == "abort_during":
            abort.trigger(True)
        return False, None

    consumer.consume.side_effect = consume
    if mode != "unregistered":
        streamer.register_stream_processing("channel", "topic", factory)
    request = Shareable()
    for key, value in [
        (module.HeaderKey.TX_ID, "tx"),
        (module.HeaderKey.TOPIC, "topic"),
        (module.HeaderKey.CHANNEL, "channel"),
        (module.HeaderKey.SEQ, 1 if mode == "missing_initial" else 0),
        (module.HeaderKey.CTX, {} if mode == "missing_context" else {"metadata": True}),
    ]:
        request.set_header(key, value)
    if mode == "abort_before":
        abort.trigger(True)
    reply = streamer._handle_request("stream", request, context)
    assert reply.get_return_code() == code
    assert streamer.tx_table == {}
    acquired = mode in ["consume_error", "abort_during", "empty_reply"]
    assert factory.get_consumer.call_count == consumer.finalize.call_count == int(acquired)


@pytest.mark.parametrize("result", [None, {"complete": True}])
def test_sender_abort_after_first_reply_closes_remote_session_without_producing_more(streamer, result):
    context, abort = FLContext(), Signal()
    context.set_prop(ReservedKey.RUN_ABORT_SIGNAL, abort)
    producer = Mock(spec=ObjectProducer)
    producer.produce.side_effect = [(Shareable(), 1.0), AssertionError("produced after abort")]

    def replies(*args):
        abort.trigger(True)
        return result

    producer.process_replies.side_effect = replies
    assert streamer.stream("channel", "topic", {}, [], producer, context) == (ReturnCode.TASK_ABORTED, result)
    assert producer.produce.call_count == 1
    assert streamer.aux_runner.send_aux_request.call_count == 2
    assert streamer.aux_runner.send_aux_request.call_args.kwargs["topic"] == module.TOPIC_STREAM_ABORT


def test_async_stream_submission_uses_owned_executor_and_rejects_shutdown(streamer):
    producer = Mock(spec=ObjectProducer)
    context = FLContext()
    executor = streamer.streaming_executor
    result = streamer.stream_no_wait("channel", "topic", {}, [], producer, context, secure=True, optional=True)
    assert result is executor.submit.return_value
    assert executor.submit.call_args.args == (streamer.stream,)
    assert executor.submit.call_args.kwargs["secure"] and executor.submit.call_args.kwargs["optional"]
    streamer.shutdown()
    executor.shutdown.assert_called_once_with(wait=False, cancel_futures=True)
    with pytest.raises(RuntimeError, match="shut down"):
        streamer.stream_no_wait("channel", "topic", {}, [], producer, context)
