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
from unittest.mock import MagicMock

import pytest

from nvflare.client.ipc import defs
from nvflare.client.ipc import ipc_agent as module
from nvflare.fuel.f3.cellnet.defs import MessageHeaderKey, ReturnCode
from nvflare.fuel.f3.cellnet.utils import make_reply
from nvflare.fuel.f3.message import Message

pytestmark = pytest.mark.usefixtures("isolated_fobs")


class Clock:
    def __init__(self):
        self.now = 100.0
        self.sleeps = []
        self.after_sleep = None

    def time(self):
        return self.now

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        assert len(self.sleeps) <= 50, "unexpected unbounded IPC poll loop"
        self.now += seconds
        if self.after_sleep:
            self.after_sleep()


@pytest.fixture
def agent(monkeypatch, tmp_path):
    cell = MagicMock()
    cell_factory = MagicMock(return_value=cell)
    network = MagicMock()
    monkeypatch.setattr(module, "Cell", cell_factory)
    monkeypatch.setattr(module, "NetAgent", MagicMock(return_value=network))
    configuration = SimpleNamespace(
        initialize=MagicMock(), find_file=MagicMock(return_value=str(tmp_path / "rootCA.pem"))
    )
    monkeypatch.setattr(module, "ConfigService", configuration)
    thread_factory = MagicMock()
    monkeypatch.setattr(module, "threading", SimpleNamespace(Lock=threading.Lock, Thread=thread_factory))
    clock = Clock()
    monkeypatch.setattr(module, "time", clock)
    extensions = MagicMock()
    monkeypatch.setattr(module, "register_ext_decomposers", extensions)
    subject = module.IPCAgent(
        "tcp://parent:1",
        "site-1",
        "app",
        str(tmp_path),
        secure_mode=True,
        decomposer_module="custom.decomposers",
        resend_result_interval=0.3,
    )
    subject.test_clock = clock
    subject.test_factories = SimpleNamespace(
        cell=cell_factory, network=module.NetAgent, config=configuration, thread=thread_factory, extensions=extensions
    )
    return subject


def _task(task_id="one", **kwargs):
    task = defs.Task("train", task_id, {"round": 1}, {"w": 2})
    for key, value in kwargs.items():
        setattr(task, key, value)
    return task


def _request(task_id="one", payload=None):
    return Message(
        headers={
            MessageHeaderKey.ORIGIN: "site-1.job",
            defs.MsgHeader.TASK_ID: task_id,
            defs.MsgHeader.TASK_NAME: "train",
        },
        payload={defs.PayloadKey.DATA: {"w": 2}, defs.PayloadKey.META: {"round": 1}} if payload is None else payload,
    )


def test_agent_construction_registers_protocol_callbacks_and_credentials_without_runtime_start(agent):
    factories = agent.test_factories
    kwargs = factories.cell.call_args.kwargs
    assert kwargs["fqcn"] == "site-1.-app"
    assert kwargs["parent_url"] == "tcp://parent:1"
    assert kwargs["secure"] and kwargs["create_internal_listener"] is False
    assert kwargs["credentials"] == agent.credentials
    assert len(kwargs["credentials"]) == 1
    assert agent.cell.register_request_cb.call_count == 4
    assert {call.kwargs["topic"] for call in agent.cell.register_request_cb.call_args_list} == {
        defs.TOPIC_GET_TASK,
        defs.TOPIC_HEARTBEAT,
        defs.TOPIC_BYE,
        defs.TOPIC_ABORT,
    }
    agent.cell.start.assert_not_called()
    agent.cell.core_cell.add_incoming_request_filter.assert_called_once_with(
        channel="*", topic="*", cb=agent._msg_received
    )
    agent.cell.core_cell.add_incoming_reply_filter.assert_called_once_with(
        channel="*", topic="*", cb=agent._msg_received
    )
    factories.extensions.assert_called_once_with("custom.decomposers")


def test_secure_agent_requires_root_certificate_before_allocating_cell(agent, tmp_path):
    agent.test_factories.config.find_file.return_value = None
    agent.test_factories.cell.reset_mock()
    with pytest.raises(ValueError, match="rootCA.pem"):
        module.IPCAgent("tcp://parent:1", "site-1", "app", str(tmp_path), secure_mode=True)
    agent.test_factories.cell.assert_not_called()


def test_start_stop_own_cell_monitor_and_network_cleanup_and_reject_restart(agent):
    agent.stop()
    agent.cell.stop.assert_not_called()
    agent.start()
    agent.start()
    agent.cell.start.assert_called_once_with()
    factory = agent.test_factories.thread
    assert factory.call_args.kwargs == {"target": agent._monitor, "daemon": True}
    factory.return_value.start.assert_called_once_with()
    agent.stop()
    agent.stop()
    agent.cell.stop.assert_called_once_with()
    agent.net_agent.close.assert_called_once_with()
    stopped = module.IPCAgent.__new__(module.IPCAgent)
    stopped.logger = MagicMock()
    stopped.is_started = False
    stopped.is_stopped = True
    with pytest.raises(defs.CallStateError, match="already stopped"):
        stopped.start()


def test_monitor_disconnects_then_expires_using_bounded_local_clock(agent, caplog):
    agent.is_connected = True
    agent.flare_site_connection_timeout = 0.1
    agent.flare_site_heartbeat_timeout = 0.5
    agent._monitor()
    assert not agent.is_connected and agent.is_done
    assert "disconnected" in caplog.text and "is dead" in caplog.text
    assert 1 <= len(agent.test_clock.sleeps) <= 4


def test_peer_activity_reconnects_updates_liveness_and_bye_closes_admission(agent):
    agent.peer_fqcn = "old.job"
    request = _request()
    agent.test_clock.now = 120
    agent._msg_received(request)
    assert agent.peer_fqcn == "site-1.job"
    assert agent.last_msg_time == 120 and agent.is_connected
    assert agent._handle_heartbeat(request).get_header(MessageHeaderKey.RETURN_CODE) == ReturnCode.OK
    assert agent._handle_bye(request).get_header(MessageHeaderKey.RETURN_CODE) == ReturnCode.OK
    with pytest.raises(defs.AgentClosed):
        agent.get_task(timeout=1)


@pytest.mark.parametrize(
    "payload", ["not a dict", {defs.PayloadKey.META: {"round": 1}}, {defs.PayloadKey.DATA: {"w": 1}}]
)
def test_task_admission_rejects_malformed_payload_without_changing_state(agent, payload):
    result = agent._receive_task(_request(payload=payload))
    assert result.get_header(MessageHeaderKey.RETURN_CODE) == ReturnCode.INVALID_REQUEST
    assert agent.current_task is agent.pending_task is None


def test_task_admission_deduplicates_and_buffers_replacement_until_previous_result(agent):
    assert agent._receive_task(_request()).get_header(MessageHeaderKey.RETURN_CODE) == ReturnCode.OK
    original = agent.current_task
    assert original.task_id == "one" and original.status == defs.Task.NEW
    agent._receive_task(_request())
    assert agent.current_task is original
    agent._receive_task(_request("two"))
    pending = agent.pending_task
    assert original.aborted and pending.task_id == "two"
    agent._receive_task(_request("two"))
    assert agent.pending_task is pending
    agent._receive_task(_request("three"))
    assert agent.pending_task.task_id == "three"
    assert agent.submit_result(defs.TaskResult({}, {})) is False
    assert agent.current_task.task_id == "three" and agent.pending_task is None


def test_new_task_marks_submitted_previous_result_as_received(agent):
    current = _task(status=defs.Task.FETCHED, last_send_result_time=100)
    agent.current_task = current
    agent._receive_task(_request("new"))
    assert current.already_received and not current.aborted
    assert agent.pending_task.task_id == "new"


@pytest.mark.parametrize("target", ["current", "pending", "unrelated"])
def test_abort_flags_active_task_or_discards_pending_without_touching_other_task(agent, target):
    current, pending = _task("current"), _task("pending")
    agent.current_task = current
    agent.pending_task = pending
    result = agent._handle_abort_task(_request(target))
    assert result.get_header(MessageHeaderKey.RETURN_CODE) == ReturnCode.OK
    assert current.aborted is (target == "current")
    assert agent.pending_task is (None if target == "pending" else pending)


@pytest.mark.parametrize("timeout,exception", [("bad", TypeError), (0, ValueError), (-1, ValueError)])
def test_get_task_rejects_invalid_timeout(agent, timeout, exception):
    with pytest.raises(exception, match="timeout"):
        agent.get_task(timeout)


def test_get_task_transitions_once_and_times_out_for_no_or_aborted_task(agent):
    assert agent.get_task(timeout=0.3) is None
    task = _task()
    agent.current_task = task
    assert agent.get_task(timeout=1) is task
    assert task.status == defs.Task.FETCHED
    with pytest.raises(defs.CallStateError, match="current task is in status"):
        agent.get_task(timeout=1)
    task.aborted = True
    assert agent.get_task(timeout=0.3) is None


@pytest.mark.parametrize("terminal", [ReturnCode.OK, ReturnCode.INVALID_REQUEST])
def test_result_retry_preserves_task_and_payload_until_terminal_reply(agent, terminal):
    task = _task(status=defs.Task.FETCHED)
    agent.current_task = task
    agent.is_connected = True
    agent.peer_fqcn = "site-1.job"
    agent.cell.send_request.side_effect = [None, make_reply(ReturnCode.COMM_ERROR), make_reply(terminal)]
    result = defs.TaskResult({"round": 1}, {"w": 3})
    assert agent.submit_result(result) is (terminal == ReturnCode.OK)
    assert agent.num_results_submitted == 1 and agent.current_task is None
    assert task.status == defs.Task.PROCESSED and task.last_send_result_time >= 100
    calls = agent.cell.send_request.call_args_list
    assert len(calls) == 3
    for call in calls:
        kwargs = call.kwargs
        assert kwargs["channel"] == defs.CHANNEL and kwargs["topic"] == defs.TOPIC_SUBMIT_RESULT
        assert kwargs["target"] == "site-1.job" and kwargs["timeout"] == 30
        message = kwargs["request"]
        assert message.get_header(defs.MsgHeader.TASK_ID) == "one"
        assert message.get_header(defs.MsgHeader.TASK_NAME) == "train"
        assert message.payload == {defs.PayloadKey.META: {"round": 1}, defs.PayloadKey.DATA: {"w": 3}}
    assert agent.test_clock.sleeps


def test_result_submission_waits_for_connection_and_allows_restart_checkpoint_once(agent):
    agent.cell.send_request.return_value = make_reply(ReturnCode.OK)
    agent.test_clock.after_sleep = lambda: setattr(agent, "is_connected", True)
    agent.peer_fqcn = "site-1.job"
    assert agent._do_submit_result(defs.TaskResult({}, {"w": 2})) is True
    message = agent.cell.send_request.call_args.kwargs["request"]
    assert message.get_header(defs.MsgHeader.TASK_ID) == ""
    assert message.get_header(defs.MsgHeader.TASK_NAME) == ""
    assert agent._do_submit_result(defs.TaskResult({}, {})) is False


def test_result_submission_rejects_invalid_type_unfetched_and_aborted_task(agent):
    with pytest.raises(TypeError, match="TaskResult"):
        agent._do_submit_result({})
    agent.current_task = _task()
    with pytest.raises(defs.CallStateError, match="current task is in status"):
        agent._do_submit_result(defs.TaskResult({}, {}))
    agent.current_task.aborted = True
    assert agent._do_submit_result(defs.TaskResult({}, {})) is False
    assert agent.num_results_submitted == 0


@pytest.mark.parametrize("state", ["stopped", "done", "aborted", "received"])
def test_result_send_observes_terminal_conditions_before_transport(agent, state):
    task = _task(last_send_result_time=100)
    if state == "stopped":
        agent.is_stopped = True
    elif state == "done":
        agent.is_done = True
    elif state == "aborted":
        task.aborted = True
    else:
        task.already_received = True
    if state in ("stopped", "done"):
        with pytest.raises(defs.AgentClosed):
            agent._send_result(task, defs.TaskResult({}, {}))
    else:
        assert agent._send_result(task, defs.TaskResult({}, {})) is (state == "received")
    agent.cell.send_request.assert_not_called()


def test_task_result_validates_return_code_and_metadata_and_normalizes_empty_values():
    with pytest.raises(TypeError, match="meta must be dict"):
        defs.TaskResult("invalid", {})
    with pytest.raises(TypeError, match="return_code must be str"):
        defs.TaskResult({}, {}, return_code=3)
    result = defs.TaskResult(None, None)
    assert result.meta == {} and result.data == {}
