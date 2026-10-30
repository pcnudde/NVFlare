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

from nvflare.fuel.f3.cellnet import net_agent as module
from nvflare.fuel.f3.cellnet.defs import MessageHeaderKey, ReturnCode
from nvflare.fuel.f3.message import Message


@pytest.fixture
def agent():
    subject = module.NetAgent.__new__(module.NetAgent)
    subject.cell, subject.logger = Mock(), Mock()
    subject.cell.get_fqcn.return_value = "local"
    subject.change_root_cb = Mock()
    return subject


@pytest.mark.parametrize(
    "method,topic,args,payload",
    [
        ("get_msg_stats_table", module._TOPIC_MSG_STATS, ("peer", "count"), {"mode": "count"}),
        ("get_pool_list", module._TOPIC_LIST_POOLS, ("peer",), None),
        ("show_pool", module._TOPIC_SHOW_POOL, ("peer", "messages", "count"), {"mode": "count", "pool": "messages"}),
        ("get_comm_config", module._TOPIC_COMM_CONFIG, ("peer",), None),
        ("get_config_vars", module._TOPIC_CONFIG_VARS, ("peer",), None),
        ("get_process_info", module._TOPIC_PROCESS_INFO, ("peer",), None),
    ],
)
def test_successful_diagnostic_requests_preserve_payload_and_address_correct_topic(agent, method, topic, args, payload):
    response = {"rows": [["site", "value"]]}
    agent.cell.send_request.return_value = Message(
        headers={MessageHeaderKey.RETURN_CODE: ReturnCode.OK}, payload=response
    )
    assert getattr(agent, method)(*args) == response
    options = agent.cell.send_request.call_args.kwargs
    assert options["topic"] == topic and options["target"] == "peer" and options["timeout"] == 1.0
    assert options["request"].payload == payload


def test_route_request_retains_both_header_directions_and_root_change_propagates_before_callback(agent):
    agent.get_route_info = Mock(return_value=({"hop": "reply"}, {"hop": "request"}))
    reply = agent._do_start_route(Message(payload="peer.child"))
    assert reply.payload == {"request": {"hop": "request"}, "reply": {"hop": "reply"}}
    agent.get_route_info.assert_called_once_with("peer.child")
    order = Mock()
    agent.change_root = Mock()
    order.attach_mock(agent.change_root, "propagate")
    order.attach_mock(agent.change_root_cb, "callback")
    assert agent._do_change_root(Message(payload="tcp://root")) is None
    assert [call[0] for call in order.mock_calls] == ["propagate", "callback"]
    agent.change_root_cb.assert_called_once_with("tcp://root")


def test_url_usage_reply_merges_nested_sites_and_surfaces_bad_or_failed_children(agent):
    agent._get_url_use_of_cell = Mock(return_value="listener")
    agent._broadcast_to_subs = Mock(
        return_value={
            "valid": Message(headers={MessageHeaderKey.RETURN_CODE: ReturnCode.OK}, payload={"child": "connector"}),
            "bad": Message(headers={MessageHeaderKey.RETURN_CODE: ReturnCode.OK}, payload=[]),
            "failed": Message(headers={MessageHeaderKey.RETURN_CODE: ReturnCode.TIMEOUT}),
        }
    )
    response = agent._do_url_use(Message(payload="tcp://root"))
    assert response.payload["local"] == "listener" and response.payload["child"] == "connector"
    assert "bad reply type" in response.payload["bad"] and "error" in response.payload["failed"]
    assert agent._broadcast_to_subs.call_args.kwargs["message"].payload == "tcp://root"


def test_stats_pool_diagnostic_reports_exact_headers_rows_and_mode(monkeypatch, agent):
    pool = Mock()
    pool.get_table.return_value = (["name"], [["count"]])
    monkeypatch.setattr(module, "StatsPoolManager", SimpleNamespace(get_pool=Mock(return_value=pool)))
    reply = agent._do_show_pool(Message(payload={"pool": "messages", "mode": "count"}))
    assert reply.payload == {"headers": ["name"], "rows": [["count"]]}
    module.StatsPoolManager.get_pool.assert_called_once_with("messages")
    pool.get_table.assert_called_once_with("count")


def test_local_stop_handlers_use_same_shutdown_policy_but_cell_request_receives_ack(agent):
    agent.stop = Mock()
    assert agent._do_stop(Message()) is None
    assert isinstance(agent._do_stop_cell(Message()), Message)
    assert agent.stop.call_count == 2


def test_process_diagnostic_handles_platform_without_resource_module(monkeypatch, agent):
    monkeypatch.setattr(module, "resource", None)
    monkeypatch.setattr(
        module,
        "threading",
        SimpleNamespace(active_count=lambda: 1, enumerate=lambda: [SimpleNamespace(ident=7, name="worker")]),
    )
    reply = agent._do_process_info(Message())
    values = dict(reply.payload["rows"])
    assert values["Memory Usage"] == "N/A (not supported on this platform)"
    assert values["Thread Count"] == "1" and values["Thread:7"] == "worker"
