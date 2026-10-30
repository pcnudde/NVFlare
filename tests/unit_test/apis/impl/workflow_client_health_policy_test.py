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

from contextlib import nullcontext
from unittest.mock import Mock

import pytest

from nvflare.apis.client import Client
from nvflare.apis.controller_spec import ClientTask, Task
from nvflare.apis.event_type import EventType
from nvflare.apis.fl_constant import FLContextKey
from nvflare.apis.fl_context import FLContext
from nvflare.apis.impl import wf_comm_server as subject
from nvflare.apis.shareable import Shareable
from tests.timing_utils import isolate_time


@pytest.fixture
def runtime(monkeypatch):
    server = subject.WFCommServer()
    ctx = FLContext()
    engine = Mock()
    engine.new_context.side_effect = lambda: nullcontext(ctx)
    engine.get_clients.return_value = [Client("a", "a"), Client("b", "b")]
    server._engine = engine
    for name in ("log_warning", "log_error", "log_info", "log_debug", "fire_event", "system_panic"):
        monkeypatch.setattr(server, name, Mock())
    clock = isolate_time(monkeypatch, subject)
    monkeypatch.setattr(clock, "time", lambda: 100.0)
    monkeypatch.setattr(subject.ConfigService, "get_float_var", lambda **kwargs: kwargs["default"])
    return server, ctx, engine, clock


def test_dead_report_grace_disconnect_and_reconnection_emit_one_event_each(runtime, monkeypatch):
    server, ctx, _, clock = runtime
    server.process_dead_client_report("a", ctx)
    initial = server._dead_clients["a"]
    server.process_dead_client_report("a", ctx)
    assert server._dead_clients["a"] is initial
    server._check_dead_clients()
    assert server.get_client_disconnect_time("a") is None
    monkeypatch.setattr(clock, "time", lambda: 161.0)
    server._check_dead_clients()
    server._check_dead_clients()
    assert server.get_client_disconnect_time("a") == 161.0
    server.fire_event.assert_called_once_with(EventType.CLIENT_DISCONNECTED, ctx)
    assert ctx.get_prop(FLContextKey.DISCONNECTED_CLIENT_NAME) == "a"
    server.client_is_active("a", "task request", ctx)
    assert server.get_client_disconnect_time("a") is None
    assert ctx.get_prop(FLContextKey.RECONNECTED_CLIENT_NAME) == "a"
    assert [call.args[0] for call in server.fire_event.call_args_list] == [
        EventType.CLIENT_DISCONNECTED,
        EventType.CLIENT_RECONNECTED,
    ]


@pytest.mark.parametrize(
    "dead, minimum, required, violated",
    [
        ([], 2, [], False),
        (["a"], 2, [], True),
        (["a"], 1, ["a"], True),
        (["a"], 1, ["b"], False),
        (["a", "b"], 1, [], True),
    ],
)
def test_deployment_policy_checks_minimum_and_mandatory_live_sites(runtime, dead, minimum, required, violated):
    server, ctx, _, _ = runtime
    ctx.set_prop(FLContextKey.JOB_META, {"min_clients": minimum, "mandatory_clients": required})
    for name in dead:
        status = subject._DeadClientStatus()
        status.disconnect_time = 99.0
        server._dead_clients[name] = status
    assert server._job_policy_violated() is violated


def test_task_health_waits_for_lead_time_and_ignores_clients_with_received_results(runtime):
    server, _, _, _ = runtime
    task = Task("train", Shareable())
    task.targets = ["a", "b"]
    task.schedule_time = 90.0
    assert server._get_task_dead_clients(task) is None
    task.schedule_time = 0.0
    client_task = ClientTask(Client("b", "b"), task)
    client_task.result_received_time = 50.0
    task.client_tasks = [client_task]
    status = subject._DeadClientStatus()
    status.disconnect_time = 99.0
    server._dead_clients["a"] = status
    assert server._get_task_dead_clients(task) == ["a"]
    client_task.result_received_time = None
    assert server._get_task_dead_clients(task) is None


def test_monitor_panics_instead_of_scheduling_when_deployment_policy_is_violated(runtime, monkeypatch):
    server, ctx, _, _ = runtime
    monkeypatch.setattr(server, "_check_dead_clients", Mock())
    policy = Mock(side_effect=[True])
    check = Mock()
    monkeypatch.setattr(server, "_job_policy_violated", policy)
    monkeypatch.setattr(server, "check_tasks", check)
    server._monitor_tasks()
    server.system_panic.assert_called_once_with("Aborting job due to deployment policy violation", ctx)
    check.assert_not_called()


def test_monitor_stops_after_shutdown_flag_without_leaving_worker_running(runtime, monkeypatch):
    server, _, _, clock = runtime
    monkeypatch.setattr(server, "_check_dead_clients", Mock())
    monkeypatch.setattr(server, "_job_policy_violated", Mock(side_effect=[False]))
    check = Mock()
    monkeypatch.setattr(server, "check_tasks", check)
    pauses = []

    def sleep(interval):
        pauses.append(interval)
        assert len(pauses) == 1, "monitor ignored shutdown flag"
        server._all_done = True

    monkeypatch.setattr(clock, "sleep", sleep)
    server._monitor_tasks()
    check.assert_called_once_with()
    assert pauses == [0.2]
