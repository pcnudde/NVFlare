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

import asyncio
import socket
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from nvflare.fuel.f3.comm_error import CommError
from nvflare.fuel.f3.connection import ConnState
from nvflare.fuel.f3.drivers import aio_http_driver as http
from nvflare.fuel.f3.drivers import socket_conn as tcp
from nvflare.fuel.f3.drivers.driver_params import DriverParams


@pytest.fixture
def websocket_connection():
    subject = http.WsConnection.__new__(http.WsConnection)
    subject.name, subject.state, subject.closing = "unit-ws", ConnState.IDLE, False
    subject.websocket = SimpleNamespace(send_bytes=AsyncMock(), close=AsyncMock())
    subject.aio_context = SimpleNamespace(run_coro=Mock(side_effect=lambda coro: coro.close()))
    return subject


def test_async_send_failure_closes_websocket_through_context_boundary(websocket_connection):
    subject = websocket_connection
    subject.websocket.send_bytes.side_effect = ValueError("send failed")
    asyncio.run(subject._async_send_frame(b"frame"))
    subject.websocket.send_bytes.assert_awaited_once_with(b"frame")
    assert subject.closing
    subject.websocket.close.assert_called_once_with()
    subject.aio_context.run_coro.assert_called_once()


@pytest.mark.parametrize("kind", ["close", "error", "unknown", "local_close"])
def test_websocket_read_loop_handles_control_messages_and_stops_after_local_close(kind):
    kinds = http.aiohttp.WSMsgType
    process = Mock()

    async def messages():
        yield SimpleNamespace(type=kinds.BINARY, data=b"frame")
        if kind == "local_close":
            return
        yield SimpleNamespace(type=kinds.CLOSE if kind == "close" else kinds.ERROR if kind == "error" else kinds.TEXT)

    class Websocket:
        def __aiter__(self):
            return messages()

        def exception(self):
            return "remote error"

    connection = SimpleNamespace(websocket=Websocket(), process_frame=process, closing=False)
    if kind == "local_close":
        process.side_effect = lambda frame: setattr(connection, "closing", True)
    asyncio.run(http.AioHttpDriver._read_loop(connection))
    process.assert_called_once_with(b"frame")


@pytest.mark.parametrize("closing", [True, False])
def test_socket_receive_loop_contains_nonprotocol_error_and_reports_current_shutdown_state(closing, caplog):
    caplog.set_level("DEBUG", logger=tcp.log.name)
    subject = tcp.SocketConnection.__new__(tcp.SocketConnection)
    subject.name, subject.state, subject.closing = "unit-tcp", ConnState.IDLE, closing
    subject.read_frame_loop = Mock(side_effect=ValueError("socket failed"))
    assert subject.read_loop() is None
    subject.read_frame_loop.assert_called_once_with()
    assert "is closed" in caplog.text
    assert ("socket failed" in caplog.text) is (not closing)


def test_socket_properties_survive_missing_peer_and_nodelay_failure_without_real_socket():
    subject = tcp.SocketConnection.__new__(tcp.SocketConnection)
    subject.name, subject.state, subject.secure = "unit-tcp", ConnState.IDLE, True
    subject.sock = Mock(family=socket.AF_INET)
    subject.sock.setsockopt.side_effect = OSError("unsupported")
    subject.sock.getpeername.side_effect = OSError("peer gone")
    subject.sock.getsockname.return_value = ("localhost", 8002)
    subject.sock.getpeercert.return_value = b"certificate"
    subject.record_peer = Mock()
    subject._set_tcp_no_delay()
    result = subject._get_socket_properties()
    assert result[DriverParams.PEER_ADDR.value] == "N/A:0"
    assert result[DriverParams.LOCAL_ADDR.value] == "localhost:8002"
    subject.record_peer.assert_called_once_with(result, b"certificate", secure=True)


@pytest.mark.parametrize("error", [CommError(CommError.CLOSED, "closed"), CommError(CommError.ERROR, "bad data")])
def test_socket_protocol_error_does_not_escape_read_loop(error, caplog):
    caplog.set_level("DEBUG", logger=tcp.log.name)
    subject = tcp.SocketConnection.__new__(tcp.SocketConnection)
    subject.name, subject.state, subject.closing = "unit-tcp", ConnState.IDLE, False
    subject.read_frame_loop = Mock(side_effect=error)
    assert subject.read_loop() is None
    assert ("closed by peer" if error.code == CommError.CLOSED else "due to error") in caplog.text


@pytest.mark.parametrize("failure", ["construct", "read", "rejected"])
def test_http_connection_handler_contains_failure_and_releases_only_created_connection(monkeypatch, failure, caplog):
    caplog.set_level("ERROR", logger=http.log.name)
    driver = http.AioHttpDriver.__new__(http.AioHttpDriver)
    driver.is_stopping = Mock(return_value=False)
    driver.connector = SimpleNamespace(stopped=SimpleNamespace(is_set=lambda: False))
    driver.aio_context = object()
    driver.ssl_context = None
    connection = Mock()
    factory = Mock(
        return_value=connection, side_effect=ValueError("construct failed") if failure == "construct" else None
    )
    monkeypatch.setattr(http, "WsConnection", factory)
    driver.add_connection = Mock(return_value=failure != "rejected")
    driver.close_connection = Mock()
    driver._read_loop = AsyncMock(side_effect=ValueError("read failed"))
    websocket = SimpleNamespace(close=AsyncMock())
    asyncio.run(driver._connection_handler(websocket))
    assert driver.close_connection.call_count == int(failure != "construct")
    if failure == "rejected":
        websocket.close.assert_awaited_once_with()
        driver._read_loop.assert_not_awaited()
    else:
        assert ("construct failed" if failure == "construct" else "read failed") in caplog.text


def test_http_websocket_upgrade_delegates_prepared_request_to_connection_handler(monkeypatch):
    driver = http.AioHttpDriver.__new__(http.AioHttpDriver)
    driver._connection_handler = AsyncMock()
    websocket = SimpleNamespace(prepare=AsyncMock())
    factory = Mock(return_value=websocket)
    monkeypatch.setattr(http, "web", SimpleNamespace(WebSocketResponse=factory))
    request = object()
    assert asyncio.run(driver._websocket_handler(request)) is websocket
    factory.assert_called_once_with(max_msg_size=http.MAX_FRAME_SIZE)
    websocket.prepare.assert_awaited_once_with(request)
    driver._connection_handler.assert_awaited_once_with(websocket)
