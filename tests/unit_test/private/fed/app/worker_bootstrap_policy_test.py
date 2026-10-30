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
from argparse import Namespace
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from nvflare.private.fed.app import utils
from nvflare.private.fed.app.client import sub_worker_process as sub_worker
from nvflare.private.fed.app.simulator import simulator_worker
from tests.timing_utils import isolate_time


@pytest.mark.parametrize("already_stopping", [False, True])
def test_parent_monitor_stops_runner_on_requested_stop_or_missing_parent(monkeypatch, already_stopping):
    runner = Mock()
    event = threading.Event()
    if already_stopping:
        event.set()
    monkeypatch.setattr(utils, "psutil", SimpleNamespace(pid_exists=Mock(return_value=False)))
    utils.monitor_parent_process(runner, 123, event)
    runner.stop.assert_called_once_with()
    assert utils.psutil.pid_exists.call_count == int(not already_stopping)


def test_parent_liveness_guard_kills_owned_tree_and_group_through_controlled_boundaries(monkeypatch):
    event = threading.Event()
    monkeypatch.setattr(utils, "psutil", SimpleNamespace(pid_exists=lambda pid: False))
    kill = Mock()
    killpg = Mock()
    monkeypatch.setattr(utils, "kill_child_processes", kill)
    monkeypatch.setattr(utils, "os", SimpleNamespace(getpid=lambda: 77, getpgid=lambda pid: 88, killpg=killpg))
    utils.check_parent_alive(123, event)
    kill.assert_called_once_with(77)
    killpg.assert_called_once_with(88, 9)


def test_child_process_cleanup_contains_missing_parent_and_kills_each_descendant(monkeypatch):
    class MissingProcess(Exception):
        pass

    children = [Mock(), Mock()]
    parent = Mock()
    parent.children.return_value = children
    process = Mock(return_value=parent)
    monkeypatch.setattr(utils, "psutil", SimpleNamespace(Process=process, NoSuchProcess=MissingProcess))
    utils.kill_child_processes(77)
    parent.children.assert_called_once_with(recursive=True)
    for child in children:
        child.kill.assert_called_once_with()
    process.side_effect = MissingProcess()
    utils.kill_child_processes(88)
    assert all(child.kill.call_count == 1 for child in children)


def test_parent_monitor_waits_only_until_next_shutdown_check(monkeypatch):
    event = threading.Event()
    monkeypatch.setattr(utils, "psutil", SimpleNamespace(pid_exists=lambda pid: True))
    clock = isolate_time(monkeypatch, utils)

    def pause(delay):
        assert not event.is_set(), "parent monitor ignored stop request"
        event.set()

    monkeypatch.setattr(clock, "sleep", pause)
    runner = Mock()
    utils.monitor_parent_process(runner, 77, event)
    runner.stop.assert_called_once_with()


def test_subworker_bootstrap_passes_rank_scoped_statistics_and_closes_audit(monkeypatch, tmp_path):
    monkeypatch.setenv("LOCAL_RANK", "2")
    workspace = Mock()
    factory = Mock(return_value=workspace)
    monkeypatch.setattr(sub_worker, "Workspace", factory)
    for name in ("configure_logging", "fobs_initialize", "register_ext_decomposers", "set_stats_pool_config_for_job"):
        monkeypatch.setattr(sub_worker, name, Mock())
    for name in ("SecurityContentService", "AuditService", "PrivacyService"):
        monkeypatch.setattr(sub_worker, name, Mock())
    monkeypatch.setattr(sub_worker, "create_privacy_manager", Mock())
    worker = Mock()
    constructor = Mock(return_value=worker)
    monkeypatch.setattr(sub_worker, "SubWorkerExecutor", constructor)
    thread = Mock()
    monkeypatch.setattr(
        sub_worker, "threading", SimpleNamespace(Thread=Mock(return_value=thread), Event=threading.Event)
    )
    stats = Mock(return_value="stats failed")
    monkeypatch.setattr(sub_worker, "create_stats_pool_files_for_job", stats)
    logger = Mock()
    monkeypatch.setattr(sub_worker, "get_script_logger", Mock(return_value=logger))
    args = Namespace(
        workspace=str(tmp_path),
        client_name="site-a",
        job_id="job",
        decomposer_module="decomposers",
        num_processes="4",
        parent_pid=77,
    )
    sub_worker.main(args)
    constructor.assert_called_once_with(args, workspace, 4, 2)
    sub_worker.set_stats_pool_config_for_job.assert_called_once_with(workspace, "job", prefix="rank2")
    worker.run.assert_called_once_with()
    sub_worker.AuditService.close.assert_called_once_with()
    stats.assert_called_once_with(workspace, "job", prefix="rank2")
    logger.warning.assert_called_once_with("stats failed")
    thread.start.assert_called_once_with()


@pytest.mark.parametrize("failure", [False, True])
def test_simulator_worker_bootstrap_sets_gpu_scope_and_cleans_connection_after_execution(
    monkeypatch, tmp_path, failure
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("CUDA_DEVICE_ORDER", "original-order")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "original-devices")
    args = Namespace(
        parent_pid=77,
        workspace=str(tmp_path),
        client="site-a",
        logging_config="INFO",
        decomposer_module="decomposers",
        gpu="2,3",
        port=1234,
    )
    workspace = Mock()
    monkeypatch.setattr(simulator_worker, "Workspace", Mock(return_value=workspace))
    for name in ("dynamic_log_config", "fobs_initialize", "register_ext_decomposers", "init_security_content_service"):
        monkeypatch.setattr(simulator_worker, name, Mock())
    monkeypatch.setattr(simulator_worker, "AuthorizationService", Mock())
    monkeypatch.setattr(simulator_worker, "AuditService", Mock())
    stop = threading.Event()
    thread = Mock()
    monkeypatch.setattr(
        simulator_worker, "threading", SimpleNamespace(Thread=Mock(return_value=thread), Event=lambda: stop)
    )
    conn = Mock()
    monkeypatch.setattr(simulator_worker, "_create_connection", Mock(return_value=conn))
    worker = Mock()
    worker.run.side_effect = RuntimeError("task failed") if failure else None
    monkeypatch.setattr(simulator_worker, "ClientTaskWorker", Mock(return_value=worker))
    if failure:
        with pytest.raises(RuntimeError, match="task failed"):
            simulator_worker.main(args)
    else:
        simulator_worker.main(args)
    worker.run.assert_called_once_with(args, conn)
    assert simulator_worker.os.environ["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID"
    assert simulator_worker.os.environ["CUDA_VISIBLE_DEVICES"] == "2,3"
    assert stop.is_set()
    conn.close.assert_called_once_with()
    simulator_worker.AuditService.close.assert_called_once_with()
    thread.start.assert_called_once_with()
