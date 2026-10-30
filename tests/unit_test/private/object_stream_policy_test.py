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
from nvflare.apis.shareable import Shareable, make_reply
from nvflare.apis.signal import Signal
from nvflare.apis.streaming import ConsumerFactory, ObjectConsumer, ObjectProducer, StreamContextKey
from nvflare.private.stream_runner import TOPIC_STREAM_ABORT, TOPIC_STREAM_REQUEST, HeaderKey, ObjectStreamer


@pytest.fixture
def streamer():
    runner = ObjectStreamer(Mock())
    yield runner
    runner.shutdown()


@pytest.mark.parametrize("result,expected", [({"complete": True}, ReturnCode.OK), (False, ReturnCode.ERROR)])
def test_producer_success_and_failure_results_close_remote_sessions(streamer, result, expected):
    producer = Mock(spec=ObjectProducer)
    request = Shareable()
    producer.produce.return_value = (request, 2.0)
    producer.process_replies.return_value = result
    context, stream_context, targets = FLContext(), {"custom": 7}, [object()]
    assert streamer.stream(
        "channel", "topic", stream_context, targets, producer, context, secure=True, optional=True
    ) == (expected, result)
    calls = streamer.aux_runner.send_aux_request.call_args_list
    assert calls[0].kwargs["topic"] == TOPIC_STREAM_REQUEST
    assert calls[0].kwargs["secure"] is True and calls[0].kwargs["optional"] is True
    assert calls[0].kwargs["timeout"] == 2.0
    assert request.get_header(HeaderKey.CHANNEL) == "channel" and request.get_header(HeaderKey.TOPIC) == "topic"
    assert request.get_header(HeaderKey.SEQ) == 0 and request.get_header(HeaderKey.CTX) is stream_context
    assert stream_context[StreamContextKey.CHANNEL] == "channel"
    assert calls[1].kwargs["topic"] == TOPIC_STREAM_ABORT
    assert calls[1].kwargs["request"].get_header(HeaderKey.TX_ID) == request.get_header(HeaderKey.TX_ID)


def test_multi_message_stream_sends_metadata_once_and_cancels_on_exhaustion(streamer):
    producer = Mock(spec=ObjectProducer)
    first, second = Shareable(), Shareable()
    producer.produce.side_effect = [(first, 1.0), (second, 1.0), (None, 1.0)]
    producer.process_replies.return_value = None
    assert streamer.stream("channel", "topic", None, [], producer, FLContext()) == (ReturnCode.OK, None)
    assert first.get_header(HeaderKey.SEQ) == 0 and second.get_header(HeaderKey.SEQ) == 1
    assert first.get_header(HeaderKey.TX_ID) == second.get_header(HeaderKey.TX_ID)
    assert first.get_header(HeaderKey.CTX) is not None and second.get_header(HeaderKey.CTX) is None
    assert streamer.aux_runner.send_aux_request.call_count == 3


@pytest.mark.parametrize(
    "bad_request,timeout,error",
    [
        ("invalid", 1.0, "must produce Shareable"),
        (Shareable(), 1, "float timeout"),
        (Shareable(), 0.0, "positive float"),
        (Shareable(), -1.0, "positive float"),
    ],
)
def test_invalid_producer_output_after_first_message_aborts_existing_session(streamer, bad_request, timeout, error):
    producer = Mock(spec=ObjectProducer)
    producer.produce.side_effect = [(Shareable(), 1.0), (bad_request, timeout)]
    producer.process_replies.return_value = None
    with pytest.raises(ValueError, match=error):
        streamer.stream("channel", "topic", {}, [], producer, FLContext())
    assert streamer.aux_runner.send_aux_request.call_args.kwargs["topic"] == TOPIC_STREAM_ABORT


def test_producer_exception_aborts_existing_session_and_retains_exception(streamer):
    producer = Mock(spec=ObjectProducer)
    failure = RuntimeError("cannot read next chunk")
    producer.produce.side_effect = [(Shareable(), 1.0), failure]
    producer.process_replies.return_value = None
    with pytest.raises(RuntimeError) as error:
        streamer.stream("channel", "topic", {}, [], producer, FLContext())
    assert error.value is failure
    assert streamer.aux_runner.send_aux_request.call_args.kwargs["topic"] == TOPIC_STREAM_ABORT


def test_abort_before_production_does_not_open_remote_session(streamer):
    context = FLContext()
    abort = Signal()
    abort.trigger(True)
    context.set_prop(ReservedKey.RUN_ABORT_SIGNAL, abort)
    producer = Mock(spec=ObjectProducer)
    assert streamer.stream("channel", "topic", {}, [], producer, context) == (ReturnCode.TASK_ABORTED, None)
    producer.produce.assert_not_called()
    streamer.aux_runner.send_aux_request.assert_not_called()


@pytest.mark.parametrize("field", [HeaderKey.TX_ID, HeaderKey.TOPIC, HeaderKey.CHANNEL, HeaderKey.SEQ])
def test_receiver_rejects_missing_protocol_headers(streamer, field):
    request = Shareable()
    for key, value in (
        (HeaderKey.TX_ID, "tx"),
        (HeaderKey.TOPIC, "topic"),
        (HeaderKey.CHANNEL, "channel"),
        (HeaderKey.SEQ, 0),
    ):
        if key != field:
            request.set_header(key, value)
    assert streamer._handle_request("ignored", request, FLContext()).get_return_code() == ReturnCode.BAD_REQUEST_DATA
    assert streamer.tx_table == {}


@pytest.mark.parametrize("factory_result", [None, "invalid", RuntimeError("factory failed")])
def test_receiver_factory_failure_does_not_leave_a_live_transaction(streamer, factory_result):
    factory = Mock(spec=ConsumerFactory)
    if isinstance(factory_result, Exception):
        factory.get_consumer.side_effect = factory_result
    else:
        factory.get_consumer.return_value = factory_result
    streamer.register_stream_processing("channel", "topic", factory)
    request = Shareable()
    for key, value in (
        (HeaderKey.TX_ID, "tx"),
        (HeaderKey.TOPIC, "topic"),
        (HeaderKey.CHANNEL, "channel"),
        (HeaderKey.SEQ, 0),
        (HeaderKey.CTX, {"custom": 1}),
    ):
        request.set_header(key, value)
    assert streamer._handle_request("ignored", request, FLContext()).get_return_code() == ReturnCode.EXECUTION_EXCEPTION
    assert streamer.tx_table == {}


def test_receiver_finalizes_consumer_and_returns_end_result_once(streamer):
    consumer, factory = Mock(spec=ObjectConsumer), Mock(spec=ConsumerFactory)
    consumer.consume.return_value = (False, make_reply(ReturnCode.OK))
    factory.get_consumer.return_value = consumer
    done, consumed = Mock(return_value={"received": 1}), Mock()
    streamer.register_stream_processing("channel", "topic", factory, done, consumed, marker="kept")
    request, context, stream_context = Shareable(), FLContext(), {"custom": 1}
    for key, value in (
        (HeaderKey.TX_ID, "tx"),
        (HeaderKey.TOPIC, "topic"),
        (HeaderKey.CHANNEL, "channel"),
        (HeaderKey.SEQ, 0),
        (HeaderKey.CTX, stream_context),
    ):
        request.set_header(key, value)
    reply = streamer._handle_request("ignored", request, context)
    assert reply.get_return_code() == ReturnCode.OK and reply.get_header(HeaderKey.END_RESULT) == {"received": 1}
    assert streamer.tx_table == {} and stream_context[StreamContextKey.RC] == ReturnCode.OK
    consumer.finalize.assert_called_once_with(stream_context, context)
    done.assert_called_once_with(stream_context, context, marker="kept")
    factory.return_consumer.assert_called_once_with(consumer=consumer, stream_ctx=stream_context, fl_ctx=context)
    assert callable(stream_context[StreamContextKey.END_STREAM])
    assert stream_context[StreamContextKey.END_STREAM](ReturnCode.OK, context) is None


def test_abort_requires_transaction_id_and_finalizes_registered_transaction(streamer):
    assert streamer._handle_abort("ignored", Shareable(), FLContext()).get_return_code() == ReturnCode.BAD_REQUEST_DATA
    info = Mock()
    streamer.tx_table["tx"] = info
    request, context = Shareable(), FLContext()
    request.set_header(HeaderKey.TX_ID, "tx")
    assert streamer._handle_abort("ignored", request, context).get_return_code() == ReturnCode.OK
    info.stream_done.assert_called_once_with(ReturnCode.TASK_ABORTED, context)
    assert streamer.tx_table == {}
