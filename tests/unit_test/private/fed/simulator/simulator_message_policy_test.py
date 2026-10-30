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
from contextlib import nullcontext
from unittest.mock import Mock

import pytest

from nvflare.apis.fl_constant import FLContextKey, ReservedKey, ReservedTopic, ServerCommandKey
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import ReturnCode, Shareable
from nvflare.private.fed.server.server_engine import ServerEngine
from nvflare.private.fed.simulator import simulator_server as subject


@pytest.fixture
def server():
    value = object.__new__(subject.SimulatorServer)
    value.logger = Mock()
    value.engine = Mock()
    value.engine.lock = threading.Lock()
    value.engine.run_processes = {"job": {}}
    value.engine.new_context.side_effect = lambda: nullcontext(FLContext())
    return value


def test_simulator_task_request_and_submission_preserve_peer_context_and_assignment(server):
    ctx = FLContext()
    peer = FLContext()
    runner = Mock()
    runner.process_task_request.return_value = ("train", "task", Shareable())
    ctx.set_prop(FLContextKey.RUNNER, runner)
    client = object()
    data, task_id, name = server._process_task_request(client, ctx, peer)
    assert (name, task_id) == ("train", "task")
    assert ctx.get_peer_context() is peer
    result = Shareable()
    result.set_header(ServerCommandKey.FL_CLIENT, client)
    result.set_header(ServerCommandKey.TASK_NAME, "train")
    result.add_cookie(FLContextKey.TASK_ID, "task")
    server.engine.new_context.side_effect = lambda: nullcontext(ctx)
    server._submit_update({ReservedKey.SHAREABLE: result, ReservedKey.SHARED_FL_CONTEXT: peer}, None)
    runner.process_submission.assert_called_once_with(client, "train", "task", result, ctx)


@pytest.mark.parametrize("failure", [False, True])
def test_simulator_aux_dispatch_translates_engine_failure(server, failure):
    reply = Shareable()
    reply["ok"] = True
    server.engine.dispatch.return_value = reply
    server.engine.dispatch.side_effect = OSError("runner offline") if failure else None
    ctx = FLContext()
    request = Shareable()
    result = server._aux_communicate(ctx, request, None, "topic")
    if failure:
        assert result.get_return_code() == ReturnCode.COMMUNICATION_ERROR
    else:
        assert result is reply
    server.engine.dispatch.assert_called_once_with(topic="topic", request=request, fl_ctx=ctx)


@pytest.mark.parametrize("topic", ["topic", ReservedTopic.END_RUN])
def test_simulator_aux_failure_is_logged_except_expected_end_run(monkeypatch, topic):
    engine = object.__new__(subject.SimulatorServerEngine)
    engine.logger = Mock()
    send = Mock(side_effect=OSError("closed"))
    monkeypatch.setattr(ServerEngine, "send_aux_to_targets", send)
    assert engine.send_aux_request([], topic, Shareable(), 1, FLContext()) is None
    assert engine.logger.error.call_count == int(topic != ReservedTopic.END_RUN)


def test_simulator_multicast_excludes_end_run_but_forwards_other_topics(monkeypatch):
    engine = object.__new__(subject.SimulatorServerEngine)
    multicast = Mock(return_value={"site-a": "ok"})
    monkeypatch.setattr(ServerEngine, "multicast_aux_requests", multicast)
    ctx = FLContext()
    request = Shareable()
    assert engine.multicast_aux_requests(ReservedTopic.END_RUN, {"site-a": request}, 1, ctx) == {}
    multicast.assert_not_called()
    assert engine.multicast_aux_requests("topic", {"site-a": request}, 1, ctx, True, True) == {"site-a": "ok"}
    multicast.assert_called_once_with("topic", {"site-a": request}, 1, ctx, True, secure=True)


def test_simulator_engine_events_and_stop_clean_runtime_tables(server):
    engine = object.__new__(subject.SimulatorServerEngine)
    engine.run_manager = Mock()
    ctx = FLContext()
    engine.fire_event("event", ctx)
    engine.run_manager.fire_event.assert_called_once_with("event", ctx)
    engine.run_manager = None
    engine.fire_event("event", ctx)
    server.stop_run_engine_cell()
    server.engine.ask_to_stop.assert_called_once_with()
    server.stop_training()
    assert server.engine.run_processes == {}
