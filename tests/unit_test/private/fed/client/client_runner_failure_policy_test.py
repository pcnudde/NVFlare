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
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from nvflare.apis.event_type import EventType
from nvflare.apis.fl_constant import FLContextKey, ReservedTopic
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import ReservedHeaderKey, Shareable
from nvflare.apis.signal import Signal
from nvflare.private.fed.client import client_runner as subject
from nvflare.widgets.info_collector import GroupInfoCollector, InfoCollector
from tests.timing_utils import isolate_time


@pytest.fixture
def runner():
    value = object.__new__(subject.ClientRunner)
    value.engine = Mock()
    value.engine.new_context.side_effect = lambda: nullcontext(FLContext())
    value.task_lock = threading.Lock()
    value.running_tasks = {}
    value.job_id = "job"
    value.run_abort_signal = Signal()
    value.parent_target = "server"
    value.job_heartbeat_interval = 10
    for name in ("log_info", "log_error", "log_debug", "log_exception", "fire_event"):
        setattr(value, name, Mock())
    return value


def test_fetch_loop_creates_daemon_heartbeat_and_checks_abort_after_one_task(runner, monkeypatch):
    thread = Mock()
    factory = Mock(return_value=thread)
    monkeypatch.setattr(subject, "threading", SimpleNamespace(Thread=factory))
    clock = isolate_time(monkeypatch, subject)
    monkeypatch.setattr(clock, "sleep", Mock(side_effect=[None]))

    def fetch(ctx):
        runner.run_abort_signal.trigger(True)
        return 0.25, True

    runner.fetch_and_run_one_task = Mock(side_effect=fetch)
    runner._try_run()
    factory.assert_called_once_with(target=runner._send_job_heartbeat, args=[], daemon=True)
    thread.start.assert_called_once_with()
    runner.fetch_and_run_one_task.assert_called_once()
    clock.sleep.assert_called_once_with(0.25)


def test_job_heartbeat_uses_optional_fire_and_forget_and_short_shutdown_check(runner, monkeypatch):
    clock = isolate_time(monkeypatch, subject)
    monkeypatch.setattr(clock, "time", Mock(side_effect=[11, 11]))

    def pause(delay):
        assert not runner.run_abort_signal.triggered, "heartbeat ignored abort"
        runner.run_abort_signal.trigger(True)

    monkeypatch.setattr(clock, "sleep", pause)
    runner._send_job_heartbeat()
    call = runner.engine.send_aux_request.call_args.kwargs
    assert call["targets"] == ["server"]
    assert call["topic"] == ReservedTopic.JOB_HEART_BEAT
    assert call["timeout"] == 0 and call["optional"] is True
    assert isinstance(call["request"], Shareable)


@pytest.mark.parametrize("failure", ["ready", "run"])
def test_run_contains_readiness_or_execution_errors_and_releases_all_services(runner, monkeypatch, failure):
    runner.init_run = Mock(return_value=True)
    runner._try_run = Mock(side_effect=RuntimeError("execution failed") if failure == "run" else None)
    runner.end_run_events_sequence = Mock()
    ready = Mock(side_effect=RuntimeError("ready failed") if failure == "ready" else None)
    downloads, reliable = Mock(), Mock()
    monkeypatch.setattr(subject, "DownloadService", downloads)
    monkeypatch.setattr(subject, "ReliableMessage", reliable)
    runner.running_tasks = {"pending": object()}
    runner.run("/app", SimpleNamespace(), on_ready=ready)
    ready.assert_called_once_with()
    runner._try_run.assert_called_once_with()
    runner.log_exception.assert_called_once()
    runner.end_run_events_sequence.assert_called_once_with()
    reliable.shutdown.assert_called_once_with()
    downloads.shutdown.assert_called_once_with()
    runner.engine.shutdown_streamer.assert_called_once_with()
    assert runner.running_tasks == {}


def test_aux_task_dispatch_builds_assignment_with_requested_identity(runner):
    data = Shareable()
    data.set_header(ReservedHeaderKey.TASK_NAME, "train")
    data.set_header(ReservedHeaderKey.TASK_ID, "task-a")
    runner._process_task = Mock(return_value=Shareable())
    ctx = FLContext()
    reply = runner._handle_do_task("execute", data, ctx)
    task, context = runner._process_task.call_args.args
    assert (task.name, task.task_id, task.data) == ("train", "task-a", data)
    assert context is ctx and reply is runner._process_task.return_value


def test_stats_report_running_task_names_and_fatal_event_sets_abort(runner):
    ctx = FLContext()
    collector = Mock(spec=GroupInfoCollector)
    ctx.set_prop(InfoCollector.CTX_KEY_STATS_COLLECTOR, collector)
    runner.running_tasks = {"task": SimpleNamespace(name="train")}
    runner.handle_event(InfoCollector.EVENT_TYPE_GET_STATS, ctx)
    collector.set_info.assert_called_once_with(
        group_name="ClientRunner", info={"job_id": "job", "current_tasks": ["train"]}
    )
    ctx.set_prop(FLContextKey.EVENT_DATA, "fatal")
    runner.handle_event(EventType.FATAL_SYSTEM_ERROR, ctx)
    assert runner._run_abort_requested and runner.run_abort_signal.triggered


@pytest.mark.parametrize(
    "event,key",
    [
        (EventType.TASK_ASSIGNMENT_SENT, FLContextKey.TASK_DATA),
        (EventType.TASK_RESULT_RECEIVED, FLContextKey.TASK_RESULT),
    ],
)
def test_child_task_events_without_executor_fire_general_post_event(runner, event, key):
    ctx = FLContext()
    data = Shareable()
    ctx.set_prop(key, data)
    runner.find_executor = Mock(return_value=None)
    callback = (
        runner._handle_task_sent_event
        if event == EventType.TASK_ASSIGNMENT_SENT
        else runner._handle_task_result_received_event
    )
    callback(event, ctx)
    runner.log_error.assert_called_once()
    expected = (
        EventType.POST_TASK_ASSIGNMENT_SENT
        if event == EventType.TASK_ASSIGNMENT_SENT
        else EventType.POST_TASK_RESULT_RECEIVED
    )
    runner.fire_event.assert_called_once_with(expected, ctx)
