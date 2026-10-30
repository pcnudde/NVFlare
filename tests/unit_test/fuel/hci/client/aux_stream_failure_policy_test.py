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

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.fuel.f3.cellnet.defs import MessageHeaderKey, ReturnCode
from nvflare.fuel.f3.message import Message
from nvflare.fuel.hci.client import api as subject
from nvflare.fuel.hci.client.api_spec import CommandContext


@pytest.fixture
def api():
    value = object.__new__(subject.AdminAPI)
    value._debug = False
    value._print_hci = Mock()
    value.fire_session_event = Mock()
    return value


@pytest.mark.parametrize("has_reply", [False, True])
def test_aux_dispatch_preserves_message_payload_topic_and_context(api, has_reply):
    ctx, payload = FLContext(), Shareable({"request": True})
    api.new_context = lambda: nullcontext(ctx)
    api.aux_runner = Mock()
    response = Shareable({"reply": True}) if has_reply else None
    api.aux_runner.dispatch.return_value = response
    reply = api._handle_aux_message(Message({MessageHeaderKey.TOPIC: "inspect"}, payload))
    api.aux_runner.dispatch.assert_called_once_with(topic="inspect", request=payload, fl_ctx=ctx)
    assert reply.payload is response
    assert reply.get_header(MessageHeaderKey.RETURN_CODE) == (ReturnCode.OK if has_reply else None)
    with pytest.raises(AssertionError, match="CellMessage"):
        api._handle_aux_message({})


@pytest.mark.parametrize("failure", ["event", "no-attempts", "certificate", "unexpected"])
def test_login_fails_promptly_on_handler_certificate_or_unexpected_error(api, failure):
    api.auto_login_max_tries = 0 if failure == "no-attempts" else 1
    api._user_login = Mock(return_value={subject.ResultKey.STATUS: subject.APIStatus.ERROR_CERT})
    if failure == "event":
        api.fire_session_event.side_effect = ValueError("handler failed")
    if failure == "unexpected":
        api._user_login.side_effect = ValueError("unexpected login failure")
    result = api.login()
    assert result[subject.ResultKey.STATUS] == (
        subject.APIStatus.ERROR_CERT if failure == "certificate" else subject.APIStatus.ERROR_RUNTIME
    )
    assert api._user_login.call_count == int(failure in ("certificate", "unexpected"))
    if failure in ("certificate", "no-attempts"):
        assert api.fire_session_event.call_args.args[0] == subject.EventType.LOGIN_FAILURE


def test_aux_streaming_delegates_only_to_server_and_preserves_callbacks(api):
    api.object_streamer = Mock(spec=subject.ObjectStreamer)
    context, producer, factory = FLContext(), Mock(), Mock()
    assert (
        api.stream_objects("channel", "topic", {}, [], producer, context, optional=True, secure=True)
        is api.object_streamer.stream.return_value
    )
    call = api.object_streamer.stream.call_args.kwargs
    assert call["producer"] is producer and call["fl_ctx"] is context
    assert call["optional"] is True and call["secure"] is True
    assert len(call["targets"]) == 1
    assert call["targets"][0].name == subject.AuxMsgTarget.server_target().name
    callback = Mock()
    api.register_stream_processing("channel", "topic", factory, stream_done_cb=callback, marker=3)
    api.object_streamer.register_stream_processing.assert_called_once_with(
        channel="channel", topic="topic", factory=factory, stream_done_cb=callback, consumed_cb=None, marker=3
    )
    api.shutdown_streamer()
    api.object_streamer.shutdown.assert_called_once_with()


def test_login_reply_processor_sets_token_on_the_command_api():
    api = SimpleNamespace(token=None)
    ctx = CommandContext()
    ctx.set_api(api)
    subject._LoginReplyProcessor().process_token(ctx, "issued-token")
    assert api.token == "issued-token"
    with pytest.raises(TypeError, match="CommandContext"):
        subject._ServerReplyJsonProcessor(object())


@pytest.mark.parametrize("value", ["invalid", 0])
def test_timeout_validation_does_not_replace_previous_command_setting(api, value):
    api.cmd_timeout = 3
    with pytest.raises(TypeError if isinstance(value, str) else ValueError):
        api.set_command_timeout(value)
    assert api.cmd_timeout == 3
    api.unset_command_timeout()
    assert api.cmd_timeout is None
