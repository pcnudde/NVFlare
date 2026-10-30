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
from concurrent.futures import CancelledError
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from nvflare.fuel.f3.drivers import aio_grpc_driver as aio
from nvflare.fuel.f3.drivers import grpc_driver as sync


@pytest.mark.parametrize("module", [sync, aio])
@pytest.mark.parametrize("use_aio", [False, True, None])
def test_transport_selection_honors_explicit_async_preference(monkeypatch, module, use_aio):
    monkeypatch.setattr(module, "use_aio_grpc", lambda: use_aio)
    driver = module.GrpcDriver if module is sync else module.AioGrpcDriver
    expected = (
        (["grpc", "grpcs"] if module is sync else ["agrpc", "agrpcs"])
        if use_aio is None
        else (["grpc", "grpcs", "agrpc", "agrpcs"] if use_aio == (module is aio) else [])
    )
    assert driver.supported_transports() == expected
    monkeypatch.setattr(module, "requires_secure_connection", lambda resources: True)
    urls = Mock(return_value=("listen", "connect"))
    monkeypatch.setattr(module, "get_tcp_urls", urls)
    resources = {"host": "site"}
    assert driver.get_urls("grpc", resources) == ("listen", "connect")
    urls.assert_called_once_with("grpcs", resources)


@pytest.mark.parametrize("mode", ["closed", "missing-receiver", "receiver-error", "send-error"])
def test_sync_stream_rejects_queue_failure_and_finishes_read_queue_on_dispatch_errors(monkeypatch, mode):
    connection = sync.StreamConnection.__new__(sync.StreamConnection)
    connection.logger, connection.oq = Mock(), Mock()
    connection.side, connection.name = "CLIENT", "connection"
    connection.closing = mode == "closed"
    connection.frame_receiver = None if mode == "missing-receiver" else Mock()
    if mode == "receiver-error":
        connection.frame_receiver.process_frame.side_effect = ValueError("bad frame")
    if mode == "send-error":
        monkeypatch.setattr(sync.StreamConnection, "seq_num", 0)
        connection.oq.append.side_effect = OSError("queue closed")
        with pytest.raises(sync.CommError, match="queue closed"):
            connection.send_frame(memoryview(b"data"))
        queued = connection.oq.append.call_args.args[0]
        assert queued.seq == 1 and queued.data == b"data"
    else:
        connection.read_loop(iter([sync.Frame(seq=1, data=b"data")]))
        connection.oq.close.assert_called_once_with()
        if mode == "closed":
            connection.frame_receiver.process_frame.assert_not_called()
        elif mode == "missing-receiver":
            assert "not registered" in connection.logger.error.call_args.args[0]
        else:
            connection.frame_receiver.process_frame.assert_called_once_with(b"data")
            assert any("exception" in call.args[0] for call in connection.logger.debug.call_args_list)


@pytest.mark.parametrize("outcome", ["rejected", "output-error", "success"])
def test_sync_server_callback_cleans_fake_worker_and_connection_on_each_outcome(monkeypatch, outcome):
    worker = Mock()
    thread_factory = Mock(return_value=worker)
    monkeypatch.setattr(
        sync, "threading", SimpleNamespace(Thread=thread_factory, current_thread=threading.current_thread)
    )
    monkeypatch.setattr(sync, "QQ", Mock())
    monkeypatch.setattr(sync, "get_address", lambda params: "local:123")
    connection = Mock()
    connection.generate_output.side_effect = ValueError("output failed") if outcome == "output-error" else None
    connection.generate_output.return_value = iter([b"reply"])
    monkeypatch.setattr(sync, "StreamConnection", Mock(return_value=connection))
    driver = Mock()
    driver.add_connection.return_value = outcome != "rejected"
    service = sync.Servicer(SimpleNamespace(driver=driver, connector=SimpleNamespace(params={})))
    service.logger = Mock()
    context = Mock()
    context.auth_context.return_value = {}
    output = list(service.Stream(iter([]), context))
    assert output == ([b"reply"] if outcome == "success" else [])
    connection.close.assert_called_once_with()
    driver.close_connection.assert_called_once_with(connection)
    assert worker.start.call_count == worker.join.call_count == int(outcome != "rejected")
    if outcome == "output-error":
        assert "output failed" in service.logger.error.call_args.args[0]


def test_sync_secure_client_uses_credentials_and_shutdown_prevents_channel_creation(monkeypatch):
    driver = sync.GrpcDriver.__new__(sync.GrpcDriver)
    driver.logger, driver.options = Mock(), []
    driver.is_stopping = Mock(return_value=False)
    driver.add_connection, driver.close_connection = Mock(return_value=True), Mock()
    channel, credentials, connection = Mock(), Mock(), Mock()
    proxy = SimpleNamespace(
        secure_channel=Mock(return_value=channel), FutureCancelledError=sync.grpc.FutureCancelledError
    )
    monkeypatch.setattr(sync, "grpc", proxy)
    monkeypatch.setattr(sync, "get_address", lambda params: "site:443")
    monkeypatch.setattr(sync, "ssl_required", lambda params: True)
    monkeypatch.setattr(sync, "get_grpc_client_credentials", Mock(return_value=credentials))
    monkeypatch.setattr(sync, "StreamerStub", Mock())
    monkeypatch.setattr(sync, "StreamConnection", Mock(return_value=connection))
    monkeypatch.setattr(sync, "QQ", Mock())
    connector = SimpleNamespace(params={}, stopped=threading.Event())
    driver.connect(connector)
    proxy.secure_channel.assert_called_once_with("site:443", options=[], credentials=credentials)
    connection.read_loop.assert_called_once()
    connection.close.assert_called_once_with()
    driver.close_connection.assert_called_once_with(connection)
    connector.stopped.set()
    driver.connect(connector)
    assert proxy.secure_channel.call_count == 1


def test_secure_server_construction_passes_credentials_without_starting_native_runtime(monkeypatch):
    native, credentials, executor = Mock(), Mock(), Mock()
    monkeypatch.setattr(sync, "grpc", SimpleNamespace(server=Mock(return_value=native)))
    monkeypatch.setattr(sync, "futures", SimpleNamespace(ThreadPoolExecutor=Mock(return_value=executor)))
    monkeypatch.setattr(sync, "add_StreamerServicer_to_server", Mock())
    monkeypatch.setattr(sync, "get_address", lambda params: "site:443")
    monkeypatch.setattr(sync, "ssl_required", lambda params: True)
    monkeypatch.setattr(sync, "get_grpc_server_credentials", Mock(return_value=credentials))
    server = sync.Server(Mock(), SimpleNamespace(params={}), 2, [])
    native.add_secure_port.assert_called_once_with("site:443", server_credentials=credentials)
    assert server.grpc_server is native
    native.start.assert_not_called()


def test_simulated_async_disconnect_owns_fake_worker_and_waits_configured_delay(monkeypatch):
    worker = Mock()
    monkeypatch.setattr(aio, "threading", SimpleNamespace(Lock=threading.Lock, Thread=Mock(return_value=worker)))
    monkeypatch.setattr(aio, "CommConfigurator", lambda: SimpleNamespace(get_bool_var=lambda *args, **kwargs: True))
    session = aio.AioStreamSession(Mock(), Mock(), {}, context=Mock())
    session.logger, session.close = Mock(), Mock()
    monkeypatch.setattr(aio, "random", SimpleNamespace(randint=Mock(return_value=13)))
    sleep = Mock(side_effect=[None])
    monkeypatch.setattr(aio, "time", SimpleNamespace(sleep=sleep))
    worker.start.assert_called_once_with()
    session._disconnect()
    sleep.assert_called_once_with(13)
    session.close.assert_called_once_with()
    context = SimpleNamespace(abort=AsyncMock(side_effect=ValueError("already closed")))
    asyncio.run(session._abort(context))
    context.abort.assert_awaited_once_with(aio.grpc.StatusCode.CANCELLED, "service closed")


def test_async_connect_cancellation_is_contained_and_submitted_coroutine_is_released(monkeypatch):
    driver = aio.AioGrpcDriver.__new__(aio.AioGrpcDriver)
    driver.logger = Mock()
    future = Mock()
    future.result.side_effect = CancelledError()

    def submit(coro):
        coro.close()
        return future

    context = SimpleNamespace(run_coro=Mock(side_effect=submit))
    monkeypatch.setattr(aio, "AioContext", SimpleNamespace(get_global_context=lambda: context))
    driver.connect(Mock())
    context.run_coro.assert_called_once()
    future.result.assert_called_once_with()
    assert "cancelled" in driver.logger.debug.call_args.args[0]
