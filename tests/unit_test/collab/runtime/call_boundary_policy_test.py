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

from nvflare.apis.signal import Signal
from nvflare.collab import collab
from nvflare.collab.api.app import ClientApp
from nvflare.collab.api.call_opt import CallOption
from nvflare.collab.api.context import get_call_context, set_call_context
from nvflare.collab.api.exceptions import CollabCallError
from nvflare.collab.runtime import dispatch
from nvflare.collab.runtime.cell_dispatcher import CellDispatcher
from nvflare.collab.runtime.defs import CallHeaderKey, CallReplyKey, ObjectCallKey
from nvflare.fuel.f3.cellnet.defs import MessageHeaderKey, ReturnCode
from nvflare.fuel.f3.cellnet.utils import new_cell_message


@pytest.fixture(autouse=True)
def restore_context():
    previous = get_call_context()
    try:
        yield
    finally:
        set_call_context(previous)


@pytest.mark.parametrize("kind", ["ok", "timeout", "invalid-message", "invalid-payload", "remote-error", "one-way"])
def test_cell_dispatcher_validates_reply_and_preserves_authenticated_envelope_without_transport(kind):
    cell = Mock()
    cell.get_fqcn.return_value = "server.job"
    message = new_cell_message({MessageHeaderKey.RETURN_CODE: ReturnCode.OK}, {CallReplyKey.RESULT: 12})
    cell.send_request.return_value = message
    if kind == "timeout":
        message.set_header(MessageHeaderKey.RETURN_CODE, ReturnCode.TIMEOUT)
    elif kind == "invalid-message":
        cell.send_request.return_value = None
    elif kind == "invalid-payload":
        message.payload = "wrong"
    elif kind == "remote-error":
        message.payload = {
            CallReplyKey.ERROR: "bad input",
            CallReplyKey.ERROR_TYPE: "ValueError",
            CallReplyKey.ERROR_TRACEBACK: "remote trace",
        }
    dispatcher = CellDispatcher(Mock(), Mock(), "server", cell, "site.job", Signal(), Mock())
    options = CallOption(timeout=0.2, expect_result=kind != "one-way", optional=True)
    if kind in ("timeout", "remote-error"):
        with pytest.raises(CollabCallError) as caught:
            dispatcher._call_target(None, "site.helper", options, "work", value=3)
        assert caught.value.target_name == "site.helper"
        if kind == "timeout":
            assert isinstance(caught.value.cause, TimeoutError)
        else:
            assert caught.value.cause_type == "ValueError"
            assert caught.value.remote_traceback == "remote trace"
    elif kind.startswith("invalid"):
        with pytest.raises(RuntimeError, match="internal error|reply must be dict"):
            dispatcher._call_target(None, "site.helper", options, "work", value=3)
    else:
        assert dispatcher._call_target(None, "site.helper", options, "work", value=3) == (
            None if kind == "one-way" else 12
        )
    if kind == "one-way":
        request = cell.fire_and_forget.call_args.kwargs["message"]
        cell.send_request.assert_not_called()
    else:
        request = cell.send_request.call_args.kwargs["request"]
    assert request.get_header(CallHeaderKey.TARGET_NAME) == "site.helper"
    assert request.get_header(CallHeaderKey.METHOD_NAME) == "work"
    assert request.payload[ObjectCallKey.KWARGS] == {"value": 3}


class Program:
    @collab.publish
    def work(self, value):
        return value + 1


def test_dispatch_to_app_resolves_published_primary_method_and_restores_context():
    class PublishedApp(ClientApp):
        @collab.publish
        def work(self, value):
            return value + 1

    app = PublishedApp(object(), "site")
    previous = object()
    set_call_context(previous)
    request = new_cell_message(
        {
            CallHeaderKey.AUTHENTICATED_CALLER: "server",
            CallHeaderKey.METHOD_NAME: "work",
            CallHeaderKey.TARGET_NAME: "site",
        },
        {ObjectCallKey.KWARGS: {"value": 3}},
    )
    reply = dispatch._call_app_method(request, app, Mock())
    assert reply.payload[CallReplyKey.RESULT] == 4
    assert get_call_context() is previous


@pytest.mark.parametrize(
    "change,expected",
    [
        ({"caller": None}, "authenticated caller"),
        ({"method": None}, "missing or invalid"),
        ({"target": 3}, "expect str"),
        ({"target": "other"}, "does not match receiving app"),
        ({"target": "site.absent"}, "no object named"),
        ({"method": "private"}, "no method named"),
        ({"payload_method": "other"}, "method name does not match"),
        ({"payload_target": "other"}, "target name does not match"),
        ({"args": {"wrong": 1}}, "should be list/tuple"),
        ({"kwargs": [3]}, "should be dict"),
    ],
)
def test_dispatch_rejects_invalid_authenticated_call_contract_before_user_execution(change, expected):
    app = ClientApp(Program(), "site")
    headers = {
        CallHeaderKey.AUTHENTICATED_CALLER: change.get("caller", "server"),
        CallHeaderKey.METHOD_NAME: change.get("method", "work"),
        CallHeaderKey.TARGET_NAME: change.get("target", "site"),
    }
    payload = {ObjectCallKey.ARGS: change.get("args", []), ObjectCallKey.KWARGS: change.get("kwargs", {"value": 3})}
    if "payload_method" in change:
        payload[ObjectCallKey.METHOD_NAME] = change["payload_method"]
    if "payload_target" in change:
        payload[ObjectCallKey.TARGET_NAME] = change["payload_target"]
    reply = dispatch._call_app_method(new_cell_message(headers, payload), app, Mock())
    assert reply.get_header(MessageHeaderKey.RETURN_CODE) != ReturnCode.OK
    assert expected in reply.payload[CallReplyKey.ERROR]
