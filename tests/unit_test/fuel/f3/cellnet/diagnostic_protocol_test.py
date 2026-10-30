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

from nvflare.fuel.f3.cellnet import net_agent as subject
from nvflare.fuel.f3.cellnet.defs import MessageHeaderKey, ReturnCode
from nvflare.fuel.f3.cellnet.net_agent import NetAgent
from nvflare.fuel.f3.message import Message
from tests.timing_utils import isolate_time


def reply(payload=None, code=ReturnCode.OK):
    return Message(headers={MessageHeaderKey.RETURN_CODE: code}, payload=payload)


@pytest.fixture
def agent(monkeypatch):
    clock = isolate_time(monkeypatch, subject)
    clock.perf_counter = Mock(side_effect=range(100, 200))
    cell = Mock()
    cell.get_fqcn.return_value = "site-1"
    cell.agents = {"site-2": object()}
    return NetAgent(cell)


def test_speed_probe_counts_each_error_class_and_verifies_payload_digest(agent):
    payload = subject._ONE_K
    agent.cell.send_request.side_effect = [
        reply(payload),
        reply(b""),
        reply(bytes([2]) * 1024),
        reply(code=ReturnCode.TIMEOUT),
        reply(code=ReturnCode.COMM_ERROR),
        reply(code=ReturnCode.MSG_TOO_BIG),
        reply(code=ReturnCode.PROCESS_EXCEPTION),
    ]
    result = agent._request_speed_test("site-2", 7, 1).payload
    assert {k: result[k] for k in ("timeouts", "comm_errors", "size_errors", "proc_errors", "other_errors")} == {
        "timeouts": 1,
        "comm_errors": 1,
        "size_errors": 1,
        "proc_errors": 2,
        "other_errors": 1,
    }
    assert result["average"] == result["total"] / 7
    assert result["prep"] == 1 and "7 rounds" in result["test"]
    assert agent.cell.send_request.call_count == 7
    for call in agent.cell.send_request.call_args_list:
        assert call.kwargs["request"].payload == payload
        assert call.kwargs["target"] == "site-2" and call.kwargs["timeout"] == 10


@pytest.mark.parametrize("params", [None, [], {}, {"to": "bad name"}])
def test_speed_request_rejects_invalid_shape_or_target_before_probe(agent, params, monkeypatch):
    probe = Mock()
    monkeypatch.setattr(agent, "_request_speed_test", probe)
    assert (
        agent._do_speed(Message(payload=params)).get_header(MessageHeaderKey.RETURN_CODE) == ReturnCode.INVALID_REQUEST
    )
    probe.assert_not_called()


def test_speed_request_normalizes_nonpositive_dimensions_before_probe(agent, monkeypatch):
    probe = Mock(return_value=reply({"result": "ok"}))
    monkeypatch.setattr(agent, "_request_speed_test", probe)
    assert agent._do_speed(Message(payload={"to": "site-2", "num": 0, "size": -1})).payload == {"result": "ok"}
    probe.assert_called_once_with("site-2", 100, 1000)


def test_stress_probe_counts_transmissions_and_detects_remote_error_and_digest_mismatch(agent, monkeypatch):
    monkeypatch.setattr(subject, "random", SimpleNamespace(randrange=lambda length: 0))
    monkeypatch.setattr(subject, "os", SimpleNamespace(urandom=lambda size: b"x" * size))
    agent.cell.send_request.side_effect = [reply(b"x" * 1024), reply(b"y" * 1024), reply(code=ReturnCode.TIMEOUT)]
    result = agent._do_stress_test({"targets": ["site-1", "site-2"], "num": 3})
    assert result["counts"] == {"site-2": 3} and result["errors"] == {"site-2": 2}
    assert result["time"] == 1
    assert all(
        c.kwargs["target"] == "site-2" and c.kwargs["timeout"] == 1 for c in agent.cell.send_request.call_args_list
    )


@pytest.mark.parametrize(
    "params,error",
    [
        (None, "expect dict"),
        ({}, "no targets"),
        ({"targets": ["site-2"]}, "missing num"),
        ({"targets": ["site-1"], "num": 1}, "no targets to try"),
    ],
)
def test_stress_parameter_errors_do_not_send_requests(agent, params, error):
    assert error in agent._do_stress_test(params)["error"]
    agent.cell.send_request.assert_not_called()


def test_distributed_stress_omits_self_and_preserves_errors_per_peer(agent):
    targets = ["site-1", "site-2", "site-3"]
    agent.cell.broadcast_request.return_value = {
        "site-2": reply({"count": 3}),
        "site-3": reply(code=ReturnCode.TIMEOUT),
    }
    result = agent.start_stress_test(targets, num_rounds=3, timeout=4)
    assert result == {"site-2": {"count": 3}, "site-3": f"RC={ReturnCode.TIMEOUT}"}
    sent = agent.cell.broadcast_request.call_args.kwargs
    assert sent["targets"] == ["site-2", "site-3"]
    assert sent["request"].payload == {"targets": targets, "num": 3} and sent["timeout"] == 4
    assert targets == ["site-1", "site-2", "site-3"]
    assert "no targets" in agent.start_stress_test(["site-1"])["error"]


@pytest.mark.parametrize("method,expected_type", [("get_peers", list), ("get_connectors", dict)])
@pytest.mark.parametrize("outcome", ["valid", "invalid", "error"])
def test_remote_inventory_validates_payload_type_and_reports_transport_error(agent, method, expected_type, outcome):
    payload = ["site-2"] if expected_type is list else {"listener": "tcp://host"}
    if outcome == "invalid":
        payload = "unexpected"
    agent.cell.send_request.return_value = reply(payload, ReturnCode.TIMEOUT if outcome == "error" else ReturnCode.OK)
    error, result = getattr(agent, method)("site-2")
    if outcome == "valid":
        assert not error and result == payload
    else:
        assert "error" in error and not result
    assert agent.cell.send_request.call_args.kwargs["target"] == "site-2"


@pytest.mark.parametrize("outcome", ["valid", "wrong_type", "error"])
def test_route_diagnostics_preserve_request_reply_headers_and_remote_errors(agent, outcome):
    payload = {"request": {"from": "site-1"}, "reply": {"from": "site-2"}}
    if outcome == "wrong_type":
        payload = []
    agent.cell.send_request.return_value = reply(payload, ReturnCode.TIMEOUT if outcome == "error" else ReturnCode.OK)
    error, response_headers, request_headers = agent.start_route("site-1", "site-2")
    if outcome == "valid":
        assert error == "" and response_headers == {"from": "site-2"} and request_headers == {"from": "site-1"}
    else:
        assert error
    assert agent.cell.send_request.call_args.kwargs["request"].payload == "site-2"


@pytest.mark.parametrize("outcome", ["valid", "wrong_type", "error"])
def test_public_speed_probe_returns_timing_and_validates_remote_result(agent, outcome):
    agent.cell.send_request.return_value = reply(
        [1] if outcome == "wrong_type" else {"count": 3}, ReturnCode.TIMEOUT if outcome == "error" else ReturnCode.OK
    )
    result = agent.speed_test("site-1", "site-2", 3, 1)
    assert result["test_time"] == 1
    assert ("error" in result) is (outcome != "valid")
    if outcome == "valid":
        assert result["count"] == 3
    assert "error" in agent.speed_test("invalid name", "site-2", 3, 1)
    assert "error" in agent.speed_test("site-1", "invalid name", 3, 1)


def test_url_inventory_merges_peer_usage_and_surfaces_bad_reply_types(agent, monkeypatch):
    monkeypatch.setattr(agent, "_get_url_use_of_cell", lambda url: "local-use")
    broadcasts = Mock(
        return_value={
            "site-2": reply({"site-2": "remote-use"}),
            "site-3": reply("bad"),
            "site-4": reply(code=ReturnCode.TIMEOUT),
        }
    )
    monkeypatch.setattr(agent, "_broadcast_to_subs", broadcasts)
    result = agent.get_url_use("tcp://host")
    assert result["site-1"] == "local-use" and result["site-2"] == "remote-use"
    assert "bad reply type" in result["site-3"] and "error" in result["site-4"]
    assert broadcasts.call_args.kwargs["message"].payload == "tcp://host"
