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
from concurrent.futures import Future
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from nvflare.fuel.f3.comm_error import CommError
from nvflare.fuel.f3.drivers.connector_info import Mode
from nvflare.fuel.f3.endpoint import Endpoint, EndpointState
from nvflare.fuel.f3.sfm import conn_manager as subject
from tests.timing_utils import isolate_time


@pytest.fixture
def manager():
    value = object.__new__(subject.ConnManager)
    value.local_endpoint = Endpoint("local")
    value.sfm_endpoints = {}
    value.sfm_conns = {}
    value.connectors = {}
    value.receivers = {}
    value.monitors = []
    value.started = False
    value.stopped = False
    value.lock = threading.Lock()
    value.conn_mgr_executor = Mock()
    value.frame_mgr_executor = Mock()
    value.heartbeat_monitor = Mock()
    value.send_frame_stats = Mock()
    value.null_conn = Mock()
    return value


@pytest.mark.parametrize("failure", ["driver", "future", "executor", None])
def test_stop_requests_every_driver_before_waiting_and_cleans_both_executors(manager, failure):
    order = []
    future = Future()
    if failure == "future":
        future.set_exception(OSError("shutdown failed"))
    else:
        future.set_result(None)
    event = threading.Event()
    event.set()
    first, second = Mock(), Mock()

    def stop_first():
        order.append("first")
        if failure == "driver":
            raise OSError("shutdown failed")
        return future

    def stop_second():
        order.append("second")
        return event

    first.shutdown.side_effect = stop_first
    second.shutdown.side_effect = stop_second
    manager.connectors = {
        "first": SimpleNamespace(driver=first, stopped=threading.Event()),
        "second": SimpleNamespace(driver=second, stopped=threading.Event()),
    }
    if failure == "executor":
        manager.conn_mgr_executor.shutdown.side_effect = OSError("shutdown failed")
    if failure:
        with pytest.raises(OSError, match="shutdown failed"):
            manager.stop()
    else:
        manager.stop()
    assert order == ["first", "second"]
    assert manager.stopped
    assert all(connector.stopped.is_set() for connector in manager.connectors.values())
    manager.conn_mgr_executor.shutdown.assert_called_once_with(True)
    manager.frame_mgr_executor.shutdown.assert_called_once_with(True)
    manager.heartbeat_monitor.stop.assert_called_once_with()


@pytest.mark.parametrize("mode", [Mode.ACTIVE, Mode.PASSIVE])
def test_connector_retry_backoff_is_capped_and_stops_without_real_waiting(monkeypatch, mode):
    stopped = Mock()
    stopped.is_set.side_effect = [False, False] * 5 + [False, True]
    driver = Mock()
    starter = driver.connect if mode == Mode.ACTIVE else driver.listen
    starter.side_effect = OSError("unavailable")
    connector = SimpleNamespace(started=False, mode=mode, driver=driver, params={}, stopped=stopped)
    clock = isolate_time(monkeypatch, subject)
    monkeypatch.setattr(clock, "time", Mock(side_effect=[0, 1] * 5 + [0]))
    subject.ConnManager.start_connector_task(connector)
    assert connector.started
    assert starter.call_count == 6
    assert [c.args[0] for c in stopped.wait.call_args_list] == [1, 2, 4, 8, 10]


@pytest.mark.parametrize("state", ["absent", "unready", "no-connections"])
def test_message_send_rejects_unavailable_or_not_ready_remote_endpoints(manager, state):
    endpoint = Endpoint("remote")
    endpoint.state = EndpointState.IDLE if state == "unready" else EndpointState.READY
    if state != "absent":
        sfm = Mock(endpoint=endpoint)
        sfm.get_connection.return_value = None
        manager.sfm_endpoints[endpoint.name] = sfm
    with pytest.raises(CommError):
        manager.send_message(endpoint, 5, {}, b"data")


def test_message_send_flattens_payload_and_uses_selected_stream_connection(manager):
    endpoint = Endpoint("remote")
    endpoint.state = EndpointState.READY
    sfm = Mock(endpoint=endpoint)
    sfm.next_stream_id.return_value = 7
    manager.sfm_endpoints[endpoint.name] = sfm
    manager.send_message(endpoint, 5, {"key": "value"}, [b"a", b"b"])
    sfm.get_connection.assert_called_once_with(7)
    sfm.get_connection.return_value.send_data.assert_called_once_with(5, 7, {"key": "value"}, bytearray(b"ab"))
    manager.send_frame_stats.record_value.assert_called_once()


def test_endpoint_removal_closes_every_transport_connection(manager):
    connections = [Mock(), Mock()]
    endpoint = Endpoint("remote")
    manager.sfm_endpoints["remote"] = SimpleNamespace(endpoint=endpoint, connections=connections)
    assert manager.find_endpoint("remote") is endpoint
    assert manager.get_connections("remote") == connections
    manager.remove_endpoint("remote")
    for connection in connections:
        connection.conn.close.assert_called_once_with()
    assert manager.find_endpoint("remote") is None
    assert manager.get_connections("remote") is None
    manager.remove_endpoint("remote")


def test_duplicate_receiver_and_shutdown_submissions_are_rejected_safely(manager):
    receiver = Mock()
    manager.register_message_receiver(1, receiver)
    with pytest.raises(CommError, match="already registered"):
        manager.register_message_receiver(1, receiver)
    manager.frame_mgr_executor.submit.side_effect = RuntimeError("executor stopped")
    manager.send_loopback_message(manager.local_endpoint, 1, {}, b"data")
    manager.process_frame(Mock(), b"frame")
    assert manager.frame_mgr_executor.submit.call_count == 2
    manager.stopped = True
    manager.send_loopback_message(manager.local_endpoint, 1, {}, b"data")
    manager.process_frame(Mock(), b"frame")
    assert manager.frame_mgr_executor.submit.call_count == 2


def test_loopback_receiver_error_is_contained_and_missing_receiver_is_ignored(manager):
    receiver = Mock()
    receiver.process_message.side_effect = ValueError("handler failed")
    manager.receivers = {1: receiver}
    manager.loopback_message_task(manager.local_endpoint, 1, {"key": "value"}, b"data")
    call = receiver.process_message.call_args.args
    assert call[:3] == (manager.local_endpoint, manager.null_conn, 1)
    assert call[3].payload == b"data"
    manager.loopback_message_task(manager.local_endpoint, 2, {}, b"data")
    assert receiver.process_message.call_count == 1
