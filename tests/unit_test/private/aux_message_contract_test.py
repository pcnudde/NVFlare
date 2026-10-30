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

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from nvflare.apis.fl_constant import ProcessType, ReservedKey, ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import ReservedHeaderKey, Shareable, make_reply
from nvflare.fuel.f3.cellnet.defs import MessageHeaderKey
from nvflare.fuel.f3.cellnet.defs import ReturnCode as CellReturnCode
from nvflare.fuel.f3.message import Message
from nvflare.private import aux_runner
from nvflare.private.aux_runner import AuxMsgTarget, AuxRunner


@pytest.fixture
def runner():
    cell = Mock()
    engine = Mock()
    engine.get_cell.return_value = cell
    runner = AuxRunner(engine)
    runner.logger = Mock()
    return runner


def _context():
    context = FLContext()
    context.set_prop(ReservedKey.PROCESS_TYPE, ProcessType.CLIENT_JOB)
    context.set_prop(ReservedKey.RUN_NUM, "job-1")
    context.set_prop("public", "retained", private=False)
    return context


@pytest.mark.parametrize(
    "topic,handler,error",
    [
        (1, lambda: None, TypeError),
        ("", lambda: None, ValueError),
        ("topic", None, ValueError),
        ("topic", "not callable", TypeError),
    ],
)
def test_handler_registration_rejects_invalid_contract_without_reserving_topic(runner, topic, handler, error):
    with pytest.raises(error):
        runner.register_aux_message_handler(topic, handler)
    assert runner.topic_table == {}


def test_handler_registration_is_unique_and_dispatch_preserves_peer_and_cookies(runner):
    context = FLContext()
    request = Shareable()
    request.set_peer_props({"sender": "site-2"})
    request.add_cookie("correlation", "id-1")
    observations = []

    def handler(topic, request, fl_ctx):
        observations.append((topic, fl_ctx.get_peer_context().get_prop("sender")))
        reply = Shareable()
        reply["answer"] = 17
        return reply

    runner.register_aux_message_handler("topic", handler)
    with pytest.raises(ValueError, match="already registered"):
        runner.register_aux_message_handler("topic", lambda: None)
    reply = runner.dispatch("topic", request, context)
    assert reply["answer"] == 17 and reply.get_cookie("correlation") == "id-1"
    assert observations == [("topic", "site-2")]


@pytest.mark.parametrize(
    "case,code",
    [
        ("unknown", ReturnCode.TOPIC_UNKNOWN),
        ("missing", ReturnCode.MISSING_PEER_CONTEXT),
        ("bad_peer", ReturnCode.BAD_PEER_CONTEXT),
        ("bad_request", ReturnCode.BAD_REQUEST_DATA),
        ("handler_error", ReturnCode.HANDLER_EXCEPTION),
    ],
)
def test_dispatch_returns_distinct_protocol_errors_for_invalid_requests(runner, case, code):
    runner.register_aux_message_handler("topic", Mock(side_effect=RuntimeError("handler failed")))
    request = Shareable()
    if case == "bad_request":
        request = object()
    elif case == "bad_peer":
        request.set_peer_props(["not a dict"])
    elif case == "handler_error":
        request.set_peer_props({"sender": "site-1"})
    process = runner._process_request if case == "bad_peer" else runner.dispatch
    reply = process("unknown" if case == "unknown" else "topic", request, FLContext())
    assert reply.get_return_code() == code


def test_wait_for_cell_returns_ready_cell_and_bounds_missing_cell(monkeypatch, runner):
    runner.cell_wait_timeout = 1.0
    assert runner._wait_for_cell() is runner.engine.get_cell.return_value
    runner.engine.get_cell.return_value = None
    stamps = iter([0.0, 2.0])
    monkeypatch.setattr(aux_runner, "time", SimpleNamespace(time=lambda: next(stamps), sleep=Mock()))
    assert runner._wait_for_cell() is None
    runner.logger.error.assert_called_once()


@pytest.mark.parametrize(
    "rc,expected",
    [
        (CellReturnCode.TIMEOUT, ReturnCode.TIMEOUT),
        (CellReturnCode.COMM_ERROR, ReturnCode.COMMUNICATION_ERROR),
        (CellReturnCode.PROCESS_EXCEPTION, ReturnCode.EXECUTION_EXCEPTION),
        ("unexpected", ReturnCode.ERROR),
    ],
)
def test_cell_error_codes_are_translated_to_application_protocol_errors(runner, rc, expected):
    message = Message(headers={MessageHeaderKey.RETURN_CODE: rc}, payload=None)
    result = runner._process_cell_replies(
        {"relay.site-2.job-1": message}, "topic", "channel", {"relay.site-2.job-1": "site-2"}
    )
    assert list(result) == ["site-2"]
    assert result["site-2"].get_return_code() == expected


def test_successful_cell_payload_must_be_a_shareable(runner):
    result = runner._process_cell_replies(
        {"site-2": Message(headers={}, payload="wrong type")}, "topic", "channel", {"site-2": "site-2"}
    )
    assert result["site-2"].get_return_code() == ReturnCode.ERROR


@pytest.mark.parametrize("timeout", [0.0, 3.0])
def test_multicast_requests_carry_job_scoped_destinations_public_props_and_security(runner, timeout):
    context, request = _context(), Shareable()
    target = AuxMsgTarget("site-2", "relay.site-2")
    cell = runner.engine.get_cell.return_value
    cell.broadcast_multi_requests.return_value = {
        "relay.site-2.job-1": Message(headers={}, payload=make_reply(ReturnCode.OK))
    }
    result = runner.multicast_aux_requests("topic", [(target, request)], timeout, context, optional=True, secure=True)
    assert request.get_peer_props()["public"] == "retained"
    assert request.get_header(ReservedHeaderKey.TOPIC) == "topic"
    if timeout:
        args, kwargs = cell.broadcast_multi_requests.call_args
        assert args[1] == timeout and kwargs["secure"] is True and kwargs["optional"] is True
        assert result["site-2"].get_return_code() == ReturnCode.OK
    else:
        args, kwargs = cell.fire_multi_requests_and_forget.call_args
        assert kwargs == {"optional": True} and result == {}
    message = args[0]["relay.site-2.job-1"]
    assert message.target == "relay.site-2.job-1" and message.topic == "topic"
    assert message.message.payload is request


@pytest.mark.parametrize("optional", [False, True])
def test_optional_multicast_failure_is_contained_and_uses_expected_log_level(runner, optional):
    runner.engine.get_cell.return_value.broadcast_multi_requests.side_effect = RuntimeError("offline")
    assert (
        runner.multicast_aux_requests(
            "topic", [(AuxMsgTarget.server_target(), Shareable())], 1, _context(), optional=optional
        )
        == {}
    )
    assert runner.logger.error.called is (not optional)


def test_no_targets_no_cell_and_bad_payload_do_not_send_multicast(runner):
    assert runner.multicast_aux_requests("topic", [], 1, _context()) == {}
    assert runner.multicast_aux_requests("topic", [(AuxMsgTarget.server_target(), object())], 1, _context()) == {}
    cell = runner.engine.get_cell.return_value
    cell.broadcast_multi_requests.assert_not_called()
    runner._wait_for_cell = lambda: None
    assert runner.multicast_aux_requests("topic", [(AuxMsgTarget.server_target(), Shareable())], 1, _context()) == {}


@pytest.mark.parametrize("bulk", [False, True])
def test_fire_and_forget_chooses_queued_or_immediate_cell_delivery(runner, bulk):
    request = Shareable()
    assert (
        runner.send_aux_request(
            [AuxMsgTarget.server_target()], "topic", request, 0, _context(), bulk_send=bulk, optional=True
        )
        == {}
    )
    cell = runner.engine.get_cell.return_value
    selected = cell.queue_message if bulk else cell.fire_and_forget
    assert selected.call_args.kwargs["targets"] == ["server.job-1"]
    assert selected.call_args.kwargs["message"].payload is request
