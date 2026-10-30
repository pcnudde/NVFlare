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
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from nvflare.fuel.f3.cellnet import net_agent as module
from nvflare.fuel.f3.cellnet.connector_manager import ConnectorData
from nvflare.fuel.f3.cellnet.defs import MessageHeaderKey, ReturnCode
from nvflare.fuel.f3.message import Message


@pytest.fixture
def agent(monkeypatch):
    worker = Mock()
    worker.is_alive.return_value = True
    factory = Mock(return_value=worker)
    monkeypatch.setattr(module, "threading", SimpleNamespace(Lock=threading.Lock, Thread=factory))
    cell = Mock()
    cell.get_fqcn.return_value = "site"
    subject = module.NetAgent(cell)
    subject.logger = Mock()
    subject.worker_factory = factory
    return subject


def test_subnet_registration_validates_unique_monitor_and_owns_fake_workers(agent):
    with pytest.raises(ValueError, match="must not be empty"):
        module.SubnetMonitor("subnet", [], 2)
    monitor = module.SubnetMonitor("subnet", ["site"], 2)
    with pytest.raises(RuntimeError, match="No NetAgent"):
        monitor.stop_subnet()
    with pytest.raises(ValueError, match="must be SubnetMonitor"):
        agent.add_subnet_monitor(object())
    agent.add_subnet_monitor(monitor)
    assert monitor.agent is agent and agent.monitors == {"subnet": monitor}
    with pytest.raises(ValueError, match="already exists"):
        agent.add_subnet_monitor(monitor)
    agent.add_to_subnet("subnet", "monitor")
    agent.add_to_subnet("other", "monitor")
    assert agent.subnets == {"subnet": "monitor", "other": "monitor"}
    assert agent.worker_factory.call_count == 2
    agent.agent_closed_cb = Mock()
    agent.close()
    agent.close()
    assert agent.worker_factory.return_value.join.call_count == 2
    agent.agent_closed_cb.assert_called_once_with()
    agent.delete_subnet_monitor("subnet")
    assert agent.monitors == {}


def test_subnet_heartbeat_transitions_are_debounced_and_timeout_uses_last_seen(monkeypatch, agent):
    now = [10.0]
    monkeypatch.setattr(module, "time", SimpleNamespace(time=lambda: now[0]))
    monitor = module.SubnetMonitor("subnet", ["site"], 2)
    monitor.member_online = Mock()
    monitor.member_offline = Mock()
    member = monitor.members["site"]
    monitor.put_member_online(member)
    now[0] = 11
    monitor.put_member_online(member)
    assert member.last_heartbeat_time == 11
    monitor.put_member_offline(member)
    monitor.member_online.assert_called_once_with("site")
    monitor.member_offline.assert_not_called()
    now[0] = 14
    monitor.put_member_offline(member)
    monitor.put_member_offline(member)
    monitor.member_offline.assert_called_once_with("site")
    assert member.state == member.STATE_OFFLINE
    monitor.put_member_online(member)
    assert monitor.member_online.call_count == 2
    monitor.agent = agent
    assert monitor.stop_subnet() is agent.cell.broadcast_request.return_value
    assert agent.cell.broadcast_request.call_args.kwargs["targets"] == ["site"]
    member.state = member.STATE_OFFLINE
    assert monitor.stop_subnet() is None


@pytest.mark.parametrize("state", ["no_monitors", "missing_subnet", "unknown_member", "valid"])
def test_heartbeat_dispatch_only_updates_declared_subnet_members(agent, state):
    monitor = module.SubnetMonitor("subnet", ["site"], 2)
    monitor.member_online = Mock()
    if state != "no_monitors":
        agent.monitors["subnet"] = monitor
    request = Message(
        headers={MessageHeaderKey.ORIGIN: "other" if state == "unknown_member" else "site"},
        payload={"subnet_id": "missing" if state == "missing_subnet" else "subnet"},
    )
    agent._do_heartbeat(request)
    assert monitor.member_online.call_count == int(state == "valid")
    assert agent.logger.warning.call_count == int(state != "valid")


@pytest.mark.parametrize("interval", [-1, 0.2])
def test_heartbeat_loop_observes_close_after_interval_and_never_leaves_worker(monkeypatch, agent, interval):
    agent.subnets = {"subnet": "monitor"}
    agent.cell.comm_configurator.get_subnet_heartbeat_interval.return_value = interval
    now, sleeps = [0.0], []

    def pause(seconds):
        sleeps.append(seconds)
        assert len(sleeps) <= 2, "heartbeat ignored shutdown or deadline"
        now[0] += 6
        if len(sleeps) == 2:
            agent.asked_to_close = True

    monkeypatch.setattr(module, "time", SimpleNamespace(time=lambda: now[0], sleep=pause))
    agent._subnet_heartbeat()
    assert sleeps == [0.1, 0.1]
    assert agent.cell.fire_and_forget.call_count == 2
    assert agent.cell.fire_and_forget.call_args.kwargs["message"].payload == {"subnet_id": "subnet"}


def test_monitor_loop_checks_snapshot_then_honors_close_on_next_poll(monkeypatch, agent):
    monitor = module.SubnetMonitor("subnet", ["site"], 1)
    monitor.put_member_offline = Mock()
    agent.monitors["subnet"] = monitor
    pauses = []

    def pause(seconds):
        pauses.append(seconds)
        assert len(pauses) == 1, "monitor ignored shutdown"
        agent.asked_to_close = True

    monkeypatch.setattr(module, "time", SimpleNamespace(sleep=pause))
    agent._monitor_subnet()
    monitor.put_member_offline.assert_called_once_with(monitor.members["site"])
    assert pauses == [0.5]


def test_cell_inventory_propagates_failed_children_and_connector_use_precedence(agent):
    agent._broadcast_to_subs = Mock(
        return_value={
            "child": Message(headers={MessageHeaderKey.RETURN_CODE: ReturnCode.OK}, payload=["child", "child.job"]),
            "lost": Message(headers={MessageHeaderKey.RETURN_CODE: ReturnCode.TIMEOUT}),
        }
    )
    error, names = agent.request_cells_info()
    assert names[:3] == ["site", "child", "child.job"]
    assert "lost" in error and names[-1] == error

    def connector(url, active=True):
        return ConnectorData("handle", url, active, {"secure": True})

    cell = agent.cell
    cell.int_listener = connector("internal", False)
    cell.ext_listeners = {"external": connector("external", False)}
    cell.bb_ext_connector = connector("root")
    cell.bb_int_connector = connector("parent")
    cell.adhoc_connectors = {"peer": connector("adhoc")}
    expected = dict(
        internal="int_listen",
        external="ext_listen",
        root="bb_ext_connect",
        parent="int_connect",
        adhoc="adhoc_connect",
        unknown="none",
    )
    for url, use in expected.items():
        assert agent._get_url_use_of_cell(url) == use
    result = agent._get_connectors()
    assert result["int_listener"]["type"] == "listener"
    assert result["bb_ext_connector"]["type"] == "connector"
    assert result["adhoc_connectors"]["peer"]["params"] == {"secure": True}


def test_bulk_probe_omits_self_and_preserves_remote_errors_and_queued_sequence(monkeypatch, agent):
    agent.cell.broadcast_request.return_value = {
        "peer": Message(payload="queued", headers={MessageHeaderKey.RETURN_CODE: ReturnCode.OK}),
        "lost": Message(headers={MessageHeaderKey.RETURN_CODE: ReturnCode.TIMEOUT}),
    }
    assert agent.start_bulk_test(["site"], 3) == {"error": "no targets for bulk test"}
    assert agent.start_bulk_test(["site", "peer", "lost"], 3) == {"peer": "queued", "lost": "RC=timeout"}
    assert agent.cell.broadcast_request.call_args.kwargs["targets"] == ["peer", "lost"]
    numbers = iter([4, 5, 6])
    monkeypatch.setattr(module, "random", SimpleNamespace(randint=lambda low, high: next(numbers)))
    assert agent._do_bulk_test(Message(payload=3)).payload == "queued: [4, 5, 6]"
    assert [call.kwargs["message"].payload for call in agent.cell.queue_message.call_args_list] == [4, 5, 6]
    assert all(call.kwargs["targets"] == module.FQCN.ROOT_SERVER for call in agent.cell.queue_message.call_args_list)
    agent._do_bulk_item(Message(headers={MessageHeaderKey.ORIGIN: "peer"}, payload=4))
    assert "got 4 from peer" in agent.cell.logger.info.call_args.args[0]


@pytest.mark.parametrize(
    "method",
    ["get_msg_stats_table", "get_pool_list", "show_pool", "get_comm_config", "get_config_vars", "get_process_info"],
)
def test_diagnostic_requests_surface_remote_failure_with_detail(agent, method):
    agent.cell.send_request.return_value = Message(
        headers={MessageHeaderKey.RETURN_CODE: ReturnCode.TIMEOUT, MessageHeaderKey.ERROR: "offline"}
    )
    args = (
        ["peer", "pool", "count"]
        if method == "show_pool"
        else ["peer", "count"] if method == "get_msg_stats_table" else ["peer"]
    )
    result = getattr(agent, method)(*args)
    assert ReturnCode.TIMEOUT in result
    if method != "get_msg_stats_table":
        assert "offline" in result
    assert agent.cell.send_request.call_args.kwargs["target"] == "peer"


def test_statistics_handlers_preserve_headers_and_unknown_pool_error(monkeypatch, agent):
    pool = Mock()
    pool.get_table.return_value = (["name"], [["pool"]])
    monkeypatch.setattr(
        module,
        "StatsPoolManager",
        SimpleNamespace(get_table=pool.get_table, get_pool=lambda name: pool if name == "present" else None),
    )
    assert agent._do_list_pools(Message()).payload == {"headers": ["name"], "rows": [["pool"]]}
    assert agent._do_show_pool(Message(payload={"pool": "present", "mode": "count"})).payload["rows"] == [["pool"]]
    pool.get_table.assert_called_with("count")
    missing = agent._do_show_pool(Message(payload={"pool": "missing"}))
    assert missing.get_header(MessageHeaderKey.RETURN_CODE) == ReturnCode.INVALID_REQUEST
    assert "missing" in missing.get_header(MessageHeaderKey.ERROR)
    agent.cell.msg_stats_pool = pool
    assert agent._do_msg_stats(Message(payload={"mode": "percent"})).payload["headers"] == ["name"]
    pool.get_table.assert_called_with("percent")


def test_process_info_remains_available_on_platform_without_resource_module(monkeypatch, agent):
    monkeypatch.setattr(module, "resource", None)
    monkeypatch.setattr(module, "os", SimpleNamespace(getpid=lambda: 77))
    monkeypatch.setattr(
        module,
        "threading",
        SimpleNamespace(active_count=lambda: 1, enumerate=lambda: [SimpleNamespace(ident=7, name="main")]),
    )
    rows = agent._do_process_info(Message()).payload["rows"]
    assert ["Process ID", "77"] in rows
    assert ["Memory Usage", "N/A (not supported on this platform)"] in rows
    assert ["Thread:7", "main"] in rows
