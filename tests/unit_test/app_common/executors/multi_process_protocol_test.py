# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

import os
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from nvflare.apis.executor import Executor
from nvflare.apis.fl_constant import FLContextKey, ReservedKey, ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.apis.signal import Signal
from nvflare.app_common.executors import multi_process_executor as subject
from tests.timing_utils import isolate_time


class ProcessExecutor(subject.MultiProcessExecutor):
    def get_multi_process_command(self):
        return "python"


@pytest.fixture
def runtime(monkeypatch):
    executor = ProcessExecutor("learner", num_of_processes=2, components=[])
    executor.engine = Mock()
    executor.executor = Mock(spec=Executor)
    executor.targets = ["site.job.0", "site.job.1"]
    ctx = FLContext()
    ctx.set_prop(ReservedKey.ENGINE, executor.engine, private=True, sticky=True)
    ctx.set_prop(ReservedKey.IDENTITY_NAME, "site", private=False, sticky=True)
    ctx.set_prop(ReservedKey.RUN_NUM, "job", private=False, sticky=True)
    ctx.set_prop(FLContextKey.ARGS, SimpleNamespace(workspace="/workspace"), private=True, sticky=True)
    cell = executor.engine.client.cell
    cell.get_fqcn.return_value = "site.job"
    cell.get_root_url_for_child.return_value = "tcp://root"
    cell.get_internal_listener_url.return_value = "tcp://parent"
    cell.is_cell_reachable.return_value = True
    cell.broadcast_request.return_value = {"rank": subject.F3make_reply(subject.F3ReturnCode.OK, "", None)}
    monkeypatch.setattr(subject.ConfigService, "get_str_var", Mock(return_value="decomposers"))
    monkeypatch.setattr(subject, "popen_in_new_session", Mock(return_value=Mock(pid=1234)))
    monkeypatch.setattr(
        subject,
        "os",
        SimpleNamespace(environ=os.environ, getpid=lambda: 100, getpgid=Mock(return_value=1234), killpg=Mock()),
    )
    return executor, ctx, cell


@pytest.mark.parametrize("value,error", [(0, ValueError), (-1, ValueError), (1.5, TypeError), ("2", TypeError)])
def test_process_count_is_validated_before_any_launch(value, error):
    with pytest.raises(error):
        ProcessExecutor(num_of_processes=value)


def test_initialize_builds_components_once_and_requires_executor(runtime, monkeypatch):
    executor, ctx, _ = runtime
    learner = executor.executor
    executor.components = {"learner": learner}
    build = Mock()
    monkeypatch.setattr(executor, "_build_components", build)
    launch = Mock()
    monkeypatch.setattr(executor, "_initialize_multi_process", launch)
    executor.initialize(ctx)
    executor.initialize(ctx)
    assert build.call_count == 1 and launch.call_count == 2
    assert executor.executor is learner
    executor.components.clear()
    with pytest.raises(ValueError, match="invalid executor"):
        executor.initialize(ctx)


@pytest.mark.parametrize("mode", ["ok", "bad-reply", "unreachable"])
def test_launch_encodes_rank_configuration_and_handles_remote_initialization_failure(runtime, monkeypatch, mode):
    executor, ctx, cell = runtime
    executor.log_exception = Mock()
    if mode == "bad-reply":
        cell.broadcast_request.return_value = {
            "rank": subject.F3make_reply(subject.F3ReturnCode.INVALID_REQUEST, "rejected", None)
        }
    elif mode == "unreachable":
        cell.is_cell_reachable.return_value = False
        clock = isolate_time(monkeypatch, subject)
        clock.time = Mock(side_effect=[0, 61])
        clock.sleep = Mock()
    executor._initialize_multi_process(ctx)
    argv = subject.popen_in_new_session.call_args.args[0]
    assert argv[:3] == ["python", "-m", "nvflare.private.fed.app.client.sub_worker_process"]
    assert argv[argv.index("--num_processes") + 1] == "2"
    assert argv[argv.index("--parent_pid") + 1] == "100"
    assert argv[argv.index("--root_url") + 1] == "tcp://root"
    assert subject.popen_in_new_session.call_args.kwargs["shell"] is False
    assert executor.log_exception.called is (mode != "ok")
    if mode == "unreachable":
        cell.broadcast_request.assert_not_called()
    else:
        assert executor.targets == ["site.job.0", "site.job.1"]
        init = cell.broadcast_request.call_args.kwargs
        assert init["topic"] == subject.MultiProcessCommandNames.INITIALIZE
        assert init["request"].payload[subject.CommunicationMetaData.LOCAL_EXECUTOR] == "learner"
        assert cell.register_request_cb.call_count == 2


def test_event_relay_preserves_event_payload_and_does_not_echo_worker_origin(runtime):
    executor, ctx, cell = runtime
    executor._pass_event_to_rank_processes("event", ctx)
    sent = cell.fire_and_forget.call_args.kwargs
    assert sent["targets"] == executor.targets
    assert sent["topic"] == subject.MultiProcessCommandNames.FIRE_EVENT
    assert sent["message"].payload[subject.CommunicationMetaData.EVENT_TYPE] == "event"
    ctx.set_prop(FLContextKey.EVENT_ORIGIN_SITE, subject.CommunicateData.SUB_WORKER_PROCESS, private=True, sticky=False)
    executor._pass_event_to_rank_processes("echo", ctx)
    assert cell.fire_and_forget.call_count == 1


def test_event_relay_transport_failure_logs_without_recursive_event(runtime):
    executor, ctx, cell = runtime
    cell.fire_and_forget.side_effect = ValueError("transport stopped")
    executor.log_warning = Mock()
    executor._pass_event_to_rank_processes("event", ctx)
    assert executor.log_warning.call_args.kwargs["fire_event"] is False


def test_received_result_and_worker_event_merge_context_and_rank(runtime):
    executor, _, _ = runtime
    inbound, local = FLContext(), FLContext()
    inbound.set_prop("metric", 0.9, private=False, sticky=False)
    executor.engine.new_context.return_value = local
    result = Shareable({"weight": [3]})
    reply = executor.receive_execute_result(
        subject.CellMessage(
            payload={subject.CommunicationMetaData.FL_CTX: inbound, subject.CommunicationMetaData.SHAREABLE: result}
        )
    )
    assert executor.execute_complete and executor.execute_result is result
    assert local.get_prop("metric") == 0.9
    assert reply.get_header(subject.MessageHeaderKey.RETURN_CODE) == ReturnCode.OK
    event = subject.CellMessage(
        payload={
            subject.CommunicationMetaData.FL_CTX: inbound,
            subject.CommunicationMetaData.EVENT_TYPE: "updated",
            subject.CommunicationMetaData.RANK_NUMBER: 2,
        }
    )
    forwarded = executor._relay_fire_event(event)
    executor.engine.fire_event.assert_called_once_with("updated", local)
    assert local.get_prop(FLContextKey.FROM_RANK_NUMBER) == 2
    assert local.get_prop(FLContextKey.EVENT_ORIGIN_SITE) == subject.CommunicateData.SUB_WORKER_PROCESS
    assert forwarded.payload[subject.CommunicationMetaData.FL_CTX].get_prop("metric") == 0.9


def test_execute_wait_is_controlled_by_result_completion_and_missing_executor_is_rejected(runtime, monkeypatch):
    executor, ctx, _ = runtime
    expected = Shareable({"weight": 2})

    def complete(**kwargs):
        assert kwargs["task_name"] == "train"
        executor.execute_result = expected
        executor.execute_complete = True

    monkeypatch.setattr(executor, "_execute_multi_process", Mock(side_effect=complete))
    assert executor.execute("train", Shareable(), ctx, Signal()) is expected
    executor.executor = None
    with pytest.raises(RuntimeError, match="no executor"):
        executor.execute("train", Shareable(), ctx, Signal())


@pytest.mark.parametrize("mode", ["ok", "abort", "transport-error"])
def test_task_dispatch_encodes_model_or_finalizes_on_abort(runtime, mode):
    executor, ctx, cell = runtime
    executor.finalize = Mock()
    signal = Signal()
    if mode == "abort":
        signal.trigger(True)
    elif mode == "transport-error":
        cell.fire_and_forget.side_effect = ValueError("closed")
    task = Shareable({"weight": 3})
    reply = executor._execute_multi_process("train", task, ctx, signal)
    if mode == "abort":
        assert reply.get_return_code() == ReturnCode.OK
        executor.finalize.assert_called_once_with(ctx)
        cell.fire_and_forget.assert_not_called()
    elif mode == "transport-error":
        assert reply.get_return_code() == ReturnCode.EXECUTION_RESULT_ERROR
    else:
        data = cell.fire_and_forget.call_args.kwargs["message"].payload
        assert data[subject.CommunicationMetaData.TASK_NAME] == "train"
        assert data[subject.CommunicationMetaData.SHAREABLE] is task


def test_finalize_is_idempotent_signals_owned_process_and_joins_only_live_relays(runtime):
    executor, ctx, cell = runtime
    executor.exe_process = Mock(pid=1234)
    alive, stopped = Mock(), Mock()
    alive.is_alive.return_value = True
    stopped.is_alive.return_value = False
    executor.relay_threads = [alive, stopped]
    executor.finalize(ctx)
    executor.finalize(ctx)
    assert executor.finalized and executor.stop_execute
    subject.os.killpg.assert_called_once_with(1234, 9)
    executor.exe_process.terminate.assert_called_once()
    alive.join.assert_called_once()
    stopped.join.assert_not_called()
    assert cell.fire_and_forget.call_count == 1
    assert cell.fire_and_forget.call_args.kwargs["topic"] == subject.MultiProcessCommandNames.CLOSE
