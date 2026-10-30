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

from nvflare.fuel.f3.drivers import grpc_driver as subject


@pytest.mark.parametrize("failure", ["context", "channel", None])
def test_sync_grpc_close_contains_rpc_errors_and_closes_queue_before_external_handles(failure):
    connection = object.__new__(subject.StreamConnection)
    connection.closing = False
    connection.lock = threading.Lock()
    connection.logger = Mock()
    connection.oq = Mock()
    connection.context = Mock()
    connection.channel = Mock()
    if failure == "context":
        connection.context.abort.side_effect = OSError("already aborted")
    if failure == "channel":
        connection.channel.close.side_effect = OSError("already closed")
    context, channel = connection.context, connection.channel
    connection.close()
    assert connection.closing and connection.context is connection.channel is None
    connection.oq.close.assert_called_once_with()
    context.abort.assert_called_once_with(subject.grpc.StatusCode.CANCELLED, "service closed")
    channel.close.assert_called_once_with()


@pytest.mark.parametrize("outcome", ["admission-rejected", "cancelled", "transport-error", "success"])
def test_sync_grpc_connector_reaps_session_after_every_outcome(monkeypatch, outcome):
    driver = object.__new__(subject.GrpcDriver)
    driver.logger = Mock()
    driver.options = []
    driver.is_stopping = Mock(return_value=False)
    driver.add_connection = Mock(return_value=outcome != "admission-rejected")
    driver.close_connection = Mock()
    channel = Mock()
    grpc_proxy = SimpleNamespace(
        insecure_channel=Mock(return_value=channel), FutureCancelledError=subject.grpc.FutureCancelledError
    )
    monkeypatch.setattr(subject, "grpc", grpc_proxy)
    monkeypatch.setattr(subject, "get_address", Mock(return_value="host:1234"))
    monkeypatch.setattr(subject, "ssl_required", Mock(return_value=False))
    monkeypatch.setattr(subject, "QQ", Mock())
    stub = Mock()
    monkeypatch.setattr(subject, "StreamerStub", Mock(return_value=stub))
    connection = Mock()
    if outcome == "cancelled":
        connection.read_loop.side_effect = grpc_proxy.FutureCancelledError()
    elif outcome == "transport-error":
        connection.read_loop.side_effect = OSError("broken stream")
    monkeypatch.setattr(subject, "StreamConnection", Mock(return_value=connection))
    driver.connect(SimpleNamespace(params={}, stopped=threading.Event()))
    connection.close.assert_called_once_with()
    driver.close_connection.assert_called_once_with(connection)
    assert connection.read_loop.call_count == int(outcome != "admission-rejected")


def test_sync_grpc_shutdown_stops_admission_once_and_returns_server_completion():
    driver = object.__new__(subject.GrpcDriver)
    driver.stop_connection_admission = Mock()
    driver.close_all = Mock()
    driver.server = Mock()
    driver.closing = False
    assert driver.shutdown() is driver.server.shutdown.return_value
    assert driver.closing
    assert driver.shutdown() is None
    driver.close_all.assert_called_once_with()
    driver.server.shutdown.assert_called_once_with()
