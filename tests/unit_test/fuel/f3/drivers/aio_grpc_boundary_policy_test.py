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
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from nvflare.fuel.f3.drivers import aio_grpc_driver as subject
from nvflare.fuel.f3.drivers.driver_params import DriverParams


@pytest.fixture
def driver():
    value = object.__new__(subject.AioGrpcDriver)
    value.logger = Mock()
    value.options = [("option", 1)]
    value.closing = False
    value.add_connection = Mock(return_value=True)
    value.close_connection = Mock()
    return value


@pytest.mark.parametrize("secure", [False, True])
@pytest.mark.parametrize("outcome", ["success", "rejected", "read-error", "cancelled"])
def test_grpc_connect_always_closes_channel_and_owned_session(driver, monkeypatch, secure, outcome):
    channel = Mock()
    channel.close = AsyncMock()
    grpc_aio = SimpleNamespace(secure_channel=Mock(return_value=channel), insecure_channel=Mock(return_value=channel))
    monkeypatch.setattr(subject, "grpc", SimpleNamespace(aio=grpc_aio))
    monkeypatch.setattr(subject, "get_address", Mock(return_value="host:1234"))
    monkeypatch.setattr(subject, "ssl_required", Mock(return_value=secure))
    credentials = Mock()
    monkeypatch.setattr(subject, "get_grpc_client_credentials", Mock(return_value=credentials))
    connection = Mock()
    connection.read_loop = AsyncMock(
        side_effect=(
            OSError("read failed")
            if outcome == "read-error"
            else asyncio.CancelledError() if outcome == "cancelled" else None
        )
    )
    factory = Mock(return_value=connection)
    monkeypatch.setattr(subject, "AioStreamSession", factory)
    stub = Mock()
    monkeypatch.setattr(subject, "StreamerStub", Mock(return_value=stub))
    connector = SimpleNamespace(params={"parameter": "value"})
    driver.add_connection.return_value = outcome != "rejected"
    if outcome == "read-error":
        with pytest.raises(OSError, match="read failed"):
            asyncio.run(driver._start_connect(connector, Mock()))
    else:
        asyncio.run(driver._start_connect(connector, Mock()))
    channel.close.assert_awaited_once_with()
    connection.close.assert_called_once_with()
    assert connection.channel is None
    driver.close_connection.assert_called_once_with(connection)
    assert connection.read_loop.await_count == int(outcome != "rejected")
    props = factory.call_args.kwargs["conn_props"]
    assert props[DriverParams.PEER_ADDR.value] == "host:1234"
    if secure:
        assert props[DriverParams.PEER_CN.value] == "N/A"
        grpc_aio.secure_channel.assert_called_once_with("host:1234", options=driver.options, credentials=credentials)
    else:
        grpc_aio.insecure_channel.assert_called_once_with("host:1234", options=driver.options)


@pytest.mark.parametrize("outcome", ["success", "constructor-error", "start-error"])
def test_server_start_context_is_signaled_on_each_outcome(driver, monkeypatch, outcome):
    server = Mock()
    server.start = AsyncMock(side_effect=OSError("start failed") if outcome == "start-error" else None)

    def create(*args, **kwargs):
        if outcome == "constructor-error":
            kwargs["conn_ctx"].error = "invalid configuration"
        return server

    monkeypatch.setattr(subject, "Server", Mock(side_effect=create))
    ctx = subject._ConnCtx()
    connector = Mock()
    asyncio.run(driver._start_server(connector, Mock(), ctx))
    assert ctx.waiter.is_set()
    assert server.start.await_count == int(outcome != "constructor-error")
    if outcome == "success":
        assert ctx.conn is True and ctx.error is None
    else:
        assert ctx.error


@pytest.mark.parametrize("secure,port_error", [(False, False), (True, False), (True, True)])
def test_server_listener_creation_uses_correct_security_and_records_binding_failure(monkeypatch, secure, port_error):
    grpc_server = Mock()
    grpc_server.add_secure_port.side_effect = OSError("bind failed") if port_error else None
    monkeypatch.setattr(subject, "grpc", SimpleNamespace(aio=SimpleNamespace(server=Mock(return_value=grpc_server))))
    monkeypatch.setattr(subject, "add_StreamerServicer_to_server", Mock())
    monkeypatch.setattr(subject, "get_address", Mock(return_value="host:1234"))
    monkeypatch.setattr(subject, "ssl_required", Mock(return_value=secure))
    credentials = Mock()
    monkeypatch.setattr(subject, "get_grpc_server_credentials", Mock(return_value=credentials))
    ctx = subject._ConnCtx()
    server = subject.Server(Mock(), SimpleNamespace(params={}), Mock(), [], ctx)
    assert server.grpc_server is grpc_server
    if secure:
        grpc_server.add_secure_port.assert_called_once_with("host:1234", server_credentials=credentials)
    else:
        grpc_server.add_insecure_port.assert_called_once_with("host:1234")
    assert bool(ctx.error) == port_error


@pytest.mark.parametrize("failure", [None, "start", "shutdown"])
def test_server_runtime_reports_start_errors_and_contains_shutdown_errors(monkeypatch, failure):
    server = object.__new__(subject.Server)
    server.logger = Mock()
    server.grpc_server_stop_grace = 0.5
    server.driver = SimpleNamespace(closing=False)
    server.connector = SimpleNamespace(stopped=threading.Event())
    server.grpc_server = Mock()
    server.grpc_server.start = AsyncMock(side_effect=OSError("start failed") if failure == "start" else None)
    server.grpc_server.wait_for_termination = AsyncMock()
    server.grpc_server.stop = AsyncMock(side_effect=OSError("shutdown failed") if failure == "shutdown" else None)
    proxy = SimpleNamespace(**vars(asyncio))
    proxy.sleep = AsyncMock()
    monkeypatch.setattr(subject, "asyncio", proxy)
    ctx = subject._ConnCtx()
    if failure == "start":
        with pytest.raises(OSError, match="start failed"):
            asyncio.run(server.start(ctx))
        assert "cannot start server" in ctx.error
    else:
        asyncio.run(server.start(ctx))
        server.grpc_server.wait_for_termination.assert_awaited_once_with()
    asyncio.run(server.shutdown())
    server.grpc_server.stop.assert_awaited_once_with(grace=0.5)
    assert proxy.sleep.await_count == int(failure != "shutdown")


@pytest.mark.parametrize("closing", [False, True])
def test_stream_session_send_failure_becomes_comm_error_unless_already_closed(monkeypatch, closing):
    session = object.__new__(subject.AioStreamSession)
    session.name = "session"
    session.closing = closing
    session.state = None
    session.logger = Mock()
    session.oq = SimpleNamespace(put=lambda frame: frame)
    session.aio_ctx = Mock()
    session.aio_ctx.run_coro.side_effect = OSError("event loop stopped")
    monkeypatch.setattr(subject.AioStreamSession, "seq_num", 0)
    if closing:
        session.send_frame(memoryview(b"data"))
    else:
        with pytest.raises(subject.CommError, match="Error sending frame"):
            session.send_frame(memoryview(b"data"))
    frame = session.aio_ctx.run_coro.call_args.args[0]
    assert frame.data == b"data" and frame.seq == 1
