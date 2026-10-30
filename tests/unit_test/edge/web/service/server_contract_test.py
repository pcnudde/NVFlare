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
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from nvflare.edge.constants import EdgeApiStatus
from nvflare.edge.web.service import server
from nvflare.edge.web.service.edge_api_pb2 import Request
from nvflare.edge.web.service.utils import make_reply


@pytest.mark.parametrize("outcome", ["reply", "empty", "error"])
def test_servicer_dispatches_on_configured_executor_and_converts_failures(monkeypatch, outcome):
    pool, handler = Mock(), Mock()
    monkeypatch.setattr(
        server,
        "concurrent",
        SimpleNamespace(futures=SimpleNamespace(thread=SimpleNamespace(ThreadPoolExecutor=Mock(return_value=pool)))),
    )
    reply = make_reply(EdgeApiStatus.OK, {"accepted": True})
    run = AsyncMock(return_value=reply if outcome == "reply" else None)
    if outcome == "error":
        run.side_effect = ValueError("failed handler")
    loop = SimpleNamespace(run_in_executor=run)
    context = SimpleNamespace(get_event_loop=lambda: loop)
    servicer = server.Servicer(handler, context, max_workers=2)
    req = Request(type="request")
    result = asyncio.run(servicer.Query(req, None))
    run.assert_awaited_once_with(pool, handler.handle_query, req)
    assert result.status == (EdgeApiStatus.OK if outcome == "reply" else EdgeApiStatus.ERROR)
    if outcome == "reply":
        assert result is reply


@pytest.fixture
def api(monkeypatch):
    monkeypatch.setattr(server.AioContext, "get_global_context", Mock(return_value=Mock()))
    return server.EdgeApiServer(Mock(), "host:9000", grpc_options=[("option", 1)], max_workers=3)


@pytest.mark.parametrize("secure", [False, True])
def test_server_start_configures_secure_or_insecure_port_without_binding_network(api, monkeypatch, secure):
    grpc_server = Mock(start=AsyncMock(), wait_for_termination=AsyncMock())
    factory = Mock(return_value=grpc_server)
    monkeypatch.setattr(server.grpc.aio, "server", factory)
    servicer = object()
    servicer_factory = Mock(return_value=servicer)
    register = Mock()
    monkeypatch.setattr(server, "Servicer", servicer_factory)
    monkeypatch.setattr(server, "add_EdgeApiServicer_to_server", register)
    api.ssl_credentials = object() if secure else None
    asyncio.run(api._start())
    assert api.grpc_server is grpc_server
    factory.assert_called_once_with(options=[("option", 1)])
    servicer_factory.assert_called_once_with(api.handler, api.aio_ctx, 3)
    register.assert_called_once_with(servicer, grpc_server)
    if secure:
        grpc_server.add_secure_port.assert_called_once_with("host:9000", server_credentials=api.ssl_credentials)
        grpc_server.add_insecure_port.assert_not_called()
    else:
        grpc_server.add_insecure_port.assert_called_once_with("host:9000")
        grpc_server.add_secure_port.assert_not_called()
    grpc_server.start.assert_awaited_once()
    grpc_server.wait_for_termination.assert_awaited_once()


@pytest.mark.parametrize("fails", [False, True])
def test_shutdown_waits_for_grace_and_releases_reference_only_after_stop(api, monkeypatch, fails):
    grpc_server = Mock(stop=AsyncMock(side_effect=ValueError("stop failed") if fails else None))
    api.grpc_server = grpc_server
    pause = AsyncMock()
    monkeypatch.setattr(server, "asyncio", SimpleNamespace(sleep=pause))
    asyncio.run(api._shutdown())
    grpc_server.stop.assert_awaited_once_with(grace=0.5)
    if fails:
        assert api.grpc_server is grpc_server
        pause.assert_not_awaited()
    else:
        assert api.grpc_server is None
        pause.assert_awaited_once_with(0.5)


def test_synchronous_start_and_shutdown_submit_lifecycle_coroutines_and_release_waiter(api):
    submitted = []

    def submit(coroutine):
        submitted.append(coroutine.cr_code.co_name)
        coroutine.close()

    api.aio_ctx.run_coro.side_effect = submit
    api.waiter = Mock()
    api.start()
    api.waiter.wait.assert_called_once_with()
    api.shutdown()
    api.waiter.set.assert_called_once_with()
    assert submitted == ["_start", "_shutdown"]
