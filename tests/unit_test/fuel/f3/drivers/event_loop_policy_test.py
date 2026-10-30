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

from nvflare.fuel.f3.drivers import aio_context as subject
from tests.timing_utils import isolate_time


@pytest.mark.parametrize("failure", [False, True])
def test_event_loop_runner_closes_loop_even_when_runtime_fails(monkeypatch, failure):
    loop = Mock()
    if failure:
        loop.run_forever.side_effect = OSError("loop failure")
    pending = Mock()
    pending.done.return_value = False
    pending.cancelled.return_value = False
    finished = Mock()
    finished.done.return_value = True
    proxy = SimpleNamespace(
        new_event_loop=Mock(return_value=loop), set_event_loop=Mock(), all_tasks=Mock(return_value=[pending, finished])
    )
    monkeypatch.setattr(subject, "asyncio", proxy)
    context = subject.AioContext("test")
    if failure:
        with pytest.raises(OSError, match="loop failure"):
            context.run_aio_loop()
    else:
        context.run_aio_loop()
    assert context.ready.is_set() and context.loop is loop
    loop.set_exception_handler.assert_called_once_with(context._handle_exception)
    loop.close.assert_called_once_with()
    assert loop.run_until_complete.call_count == int(not failure)
    if not failure:
        loop.run_until_complete.assert_called_once_with(pending)


@pytest.mark.parametrize("failure", [False, True])
def test_event_loop_shutdown_cancels_every_task_then_stops_without_real_wait(monkeypatch, failure):
    context = subject.AioContext("test")
    loop = Mock()
    context.loop = loop
    tasks = [Mock(), Mock()]
    proxy = SimpleNamespace(all_tasks=Mock(side_effect=[tasks, []]))
    monkeypatch.setattr(subject, "asyncio", proxy)
    if failure:
        loop.call_soon_threadsafe.side_effect = RuntimeError("closed loop")
    loop.is_running.return_value = False
    clock = isolate_time(monkeypatch, subject)
    monkeypatch.setattr(clock, "time", Mock(side_effect=[0, 0]))
    context.stop_aio_loop()
    assert [c.args[0] for c in loop.call_soon_threadsafe.call_args_list] == [
        tasks[0].cancel,
        tasks[1].cancel,
        loop.stop,
    ]
    assert proxy.all_tasks.call_count == 2


def test_global_context_is_created_once_and_close_retires_owned_context(monkeypatch):
    monkeypatch.setattr(subject.AioContext, "_global_ctx", None)
    thread = Mock()
    monkeypatch.setattr(subject, "threading", SimpleNamespace(Thread=Mock(return_value=thread), Event=threading.Event))
    first = subject.AioContext.get_global_context()
    assert subject.AioContext.get_global_context() is first
    assert thread.daemon is True
    thread.start.assert_called_once_with()
    first.stop_aio_loop = Mock()
    subject.AioContext.close_global_context()
    assert subject.AioContext._global_ctx is None
    first.stop_aio_loop.assert_called_once_with()


def test_exception_handler_contains_malformed_callback_context():
    context = subject.AioContext("test")
    context.logger = Mock()
    context._handle_exception(None, {})
    assert "exception in aio exception handler" in context.logger.debug.call_args.args[0]
