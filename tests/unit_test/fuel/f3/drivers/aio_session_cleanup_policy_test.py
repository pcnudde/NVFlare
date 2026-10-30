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

import grpc
import pytest

from nvflare.fuel.f3.drivers import aio_grpc_driver as subject
from nvflare.fuel.f3.drivers.connector_info import Mode
from nvflare.fuel.f3.drivers.grpc.streamer_pb2 import Frame


@pytest.fixture
def session(monkeypatch):
    monkeypatch.setattr(
        subject, "CommConfigurator", lambda: SimpleNamespace(get_bool_var=lambda *args, **kwargs: False)
    )
    context = Mock()
    value = subject.AioStreamSession(context, SimpleNamespace(mode=Mode.ACTIVE), {})
    value.logger = Mock()
    return value


@pytest.mark.parametrize("failure", [False, True])
def test_session_close_cancels_reader_and_closes_both_rpc_boundaries(session, failure):
    task = Mock()
    session.read_task = task
    session.context = SimpleNamespace(abort=AsyncMock(side_effect=OSError("already aborted") if failure else None))
    session.channel = SimpleNamespace(close=AsyncMock())
    channel = session.channel
    context = session.context
    loop = Mock()
    if failure:
        loop.call_soon_threadsafe.side_effect = RuntimeError("loop stopped")
    else:
        loop.call_soon_threadsafe.side_effect = lambda callback: callback()
    session.aio_ctx.get_event_loop.return_value = loop
    session.aio_ctx.run_coro.side_effect = lambda coro: asyncio.run(coro)
    session.close()
    assert session.closing and session.read_task is session.context is session.channel is None
    assert task.cancel.call_count == int(not failure)
    context.abort.assert_awaited_once_with(grpc.StatusCode.CANCELLED, "service closed")
    channel.close.assert_awaited_once_with()
    session.close()
    assert channel.close.await_count == 1


@pytest.mark.parametrize("failure", ["cancelled", "unavailable", "generic"])
def test_read_loop_delivers_prior_frames_and_contains_remote_or_application_error(session, failure):
    error = (
        ValueError("bad stream")
        if failure == "generic"
        else grpc.aio.AioRpcError(
            grpc.StatusCode.CANCELLED if failure == "cancelled" else grpc.StatusCode.UNAVAILABLE, None, None, "closed"
        )
    )

    async def incoming():
        yield Frame(seq=1, data=b"payload")
        raise error

    session.process_frame = Mock()
    asyncio.run(session.read_loop(incoming()))
    session.process_frame.assert_called_once_with(b"payload")
    logs = " ".join(str(c.args[0]) for c in session.logger.debug.call_args_list)
    assert (
        "closed by peer" if failure == "cancelled" else "Error: closed" if failure == "unavailable" else "bad stream"
    ) in logs


def test_read_loop_drops_frames_after_local_close(session):
    async def incoming():
        yield Frame(seq=1, data=b"payload")

    session.closing = True
    session.process_frame = Mock()
    asyncio.run(session.read_loop(incoming()))
    session.process_frame.assert_not_called()
    with pytest.raises(asyncio.CancelledError, match="connection closed"):
        asyncio.run(session.read_oq())


def test_output_queue_returns_buffer_and_keeps_read_task_owned_for_cancellation(session):
    async def exercise():
        frame = Frame(seq=7, data=b"payload")
        session.oq.put_nowait(frame)
        result = await session.read_oq()
        assert result is frame
        assert session.read_task.done()
        session.read_task = None

    asyncio.run(exercise())
