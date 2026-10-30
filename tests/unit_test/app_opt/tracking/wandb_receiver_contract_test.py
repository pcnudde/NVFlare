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

from queue import Queue
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from nvflare.apis.analytix import AnalyticsData, AnalyticsDataType, LogWriterName
from nvflare.apis.fl_constant import ProcessType
from nvflare.apis.fl_context import FLContext
from nvflare.app_opt.tracking.wandb import wandb_receiver as subject


@pytest.fixture
def receiver(monkeypatch):
    monkeypatch.setenv("WANDB_MODE", "offline")
    return subject.WandBReceiver({"project": "project", "group": "group", "job_type": "train", "name": "run"})


@pytest.mark.parametrize("missing", ["project", "group", "job_type"])
def test_wandb_rejects_incomplete_run_metadata(missing):
    args = {"project": "p", "group": "g", "job_type": "train"}
    del args[missing]
    with pytest.raises(ValueError, match=missing):
        subject._check_wandb_args(args)


def test_queue_merges_same_step_flushes_on_advance_and_discards_old_steps(receiver, monkeypatch):
    run = MagicMock()
    log = MagicMock()
    monkeypatch.setattr(subject.wandb, "init", MagicMock(return_value=run))
    monkeypatch.setattr(subject.wandb, "log", log)
    queue = Queue()
    for kind, data, step in [
        ("init", {"project": "p"}, 0),
        ("log", {"loss": 2.0}, 0),
        ("log", {"accuracy": 0.5}, None),
        ("log", {"loss": 1.0}, 1),
        ("log", {"stale": 42}, 0),
        ("stop", {}, 0),
    ]:
        queue.put(subject.WandBTask("site-1", kind, data, step))
    receiver._process_queue_tasks(queue)
    assert log.call_args_list[0].args == ({"loss": 2.0, "accuracy": 0.5}, 0)
    assert log.call_args_list[1].args == ({"loss": 1.0}, 1)
    assert log.call_count == 2
    run.finish.assert_called_once_with()
    assert queue.empty()


def test_queue_finishes_run_when_logging_boundary_raises(receiver, monkeypatch):
    run = MagicMock()
    monkeypatch.setattr(subject.wandb, "init", MagicMock(return_value=run))
    monkeypatch.setattr(subject.wandb, "log", MagicMock(side_effect=RuntimeError("backend")))
    queue = Queue()
    for kind, data, step in [("init", {}, 0), ("log", {"m": 1}, 0), ("stop", {}, 0)]:
        queue.put(subject.WandBTask("site", kind, data, step))
    with pytest.raises(RuntimeError, match="backend"):
        receiver._process_queue_tasks(queue)
    run.finish.assert_called_once_with()


@pytest.mark.parametrize("server", [False, True])
def test_initialize_builds_per_site_queues_without_launching_real_processes(receiver, monkeypatch, server):
    ctx = MagicMock(spec=FLContext)
    ctx.get_peer_context.return_value = None
    ctx.get_process_type.return_value = ProcessType.SERVER_JOB if server else "client"
    ctx.get_identity_name.return_value = "site-1"
    ctx.get_job_id.return_value = "simulate_job"
    ctx.get_engine.return_value.get_clients.return_value = [SimpleNamespace(name="site-1")]
    process = MagicMock()
    monkeypatch.setattr(subject, "Process", MagicMock(return_value=process))
    monkeypatch.setattr(subject, "Queue", Queue)
    monkeypatch.setattr(subject, "time", SimpleNamespace(time=lambda: 123, sleep=lambda _: None))
    receiver.mode = "online"
    monkeypatch.setattr(subject.wandb, "login", MagicMock(side_effect=ValueError("offline")))
    receiver.initialize(ctx)
    task = receiver.queues["site-1"].get_nowait()
    assert task.task_type == "init"
    assert task.task_data["name"] == "site-1-123-run"
    assert task.task_data["mode"] == "offline"
    process.start.assert_called_once_with()
    receiver.finalize(ctx)
    assert receiver.queues["site-1"].get_nowait().task_type == "stop"
    process.join.assert_called_once_with(10.0)
    process.terminate.assert_called_once_with()


@pytest.mark.parametrize("server", [False, True])
def test_initialize_rejects_missing_participants(receiver, server):
    ctx = MagicMock(spec=FLContext)
    ctx.get_peer_context.return_value = None
    ctx.get_process_type.return_value = ProcessType.SERVER_JOB if server else "client"
    ctx.get_engine.return_value.get_clients.return_value = []
    ctx.get_identity_name.return_value = None
    with pytest.raises(RuntimeError):
        receiver.initialize(ctx)
    assert receiver.processes == {}


@pytest.mark.parametrize(
    "kind,value,expected",
    [
        (AnalyticsDataType.METRIC, 0.3, {"value": 0.3}),
        (AnalyticsDataType.PARAMETER, "adam", {"value": "adam"}),
        (AnalyticsDataType.METRICS, {"loss": 0.3}, {"loss": 0.3}),
        (AnalyticsDataType.PARAMETERS, {"optimizer": "adam"}, {"optimizer": "adam"}),
    ],
)
def test_save_serializes_analytics_to_site_queue(receiver, kind, value, expected):
    queue = Queue()
    receiver.queues["site"] = queue
    data = AnalyticsData("value", value, kind, sender=LogWriterName.WANDB, global_step=3)
    receiver.save(MagicMock(), data.to_dxo().to_shareable(), "site")
    task = queue.get_nowait()
    assert task.task_data == expected
    assert task.step == 3 and task.task_owner == "site" and task.task_type == "log"
    receiver.save(MagicMock(), data.to_dxo().to_shareable(), "missing")
    assert queue.empty()
