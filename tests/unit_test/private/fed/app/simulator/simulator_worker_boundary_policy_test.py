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

import subprocess
import threading
from argparse import Namespace
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from nvflare.private.fed.app.simulator import simulator_runner as module


@pytest.mark.parametrize("failure", [True, False])
def test_simulator_client_runner_submits_configured_work_and_always_requests_client_cleanup(monkeypatch, failure):
    subject = module.SimulatorClientRunner.__new__(module.SimulatorClientRunner)
    subject.logger = Mock()
    subject.args = Namespace(threads=2, end_run_for_all=True)
    subject.kv_list = {"simulator_worker_timeout": 4.5}
    subject.federated_clients = [Mock(), Mock()]
    subject.run_client_thread = Mock()
    executor = Mock()
    executor.submit.side_effect = lambda callback, params: callback(params)
    factory = Mock(return_value=executor, side_effect=ValueError("cannot allocate executor") if failure else None)
    monkeypatch.setattr(module, "ThreadPoolExecutor", factory)
    workers = []

    def worker(**kwargs):
        value = Mock()
        value.kwargs = kwargs
        workers.append(value)
        return value

    monkeypatch.setattr(module, "threading", SimpleNamespace(Lock=threading.Lock, Thread=worker))
    subject.run("gpu-1")
    assert subject.run_client_thread.call_count == (0 if failure else 2)
    if not failure:
        call = subject.run_client_thread.call_args.args
        assert call[:2] == (2, "gpu-1") and call[3:] == (True, 4.5)
        executor.shutdown.assert_called_once_with()
    assert subject.logger.error.call_count == int(failure)
    assert [value.kwargs["args"][0] for value in workers] == subject.federated_clients
    for value in workers:
        value.start.assert_called_once_with()
        assert value.kwargs["target"] == subject._shutdown_client


@pytest.mark.parametrize("mode", ["none", "dead", "timeout", "gone"])
def test_simulator_process_cleanup_handles_finished_workers_and_escalates_timeout(mode):
    subject = module.SimulatorClientRunner.__new__(module.SimulatorClientRunner)
    process = Mock()
    process.poll.return_value = 0 if mode == "dead" else None
    if mode == "timeout":
        process.wait.side_effect = subprocess.TimeoutExpired("worker", 5)
    elif mode == "gone":
        process.terminate.side_effect = OSError("already gone")
    subject._cleanup_process(None if mode == "none" else process)
    assert process.terminate.call_count == int(mode in ["timeout", "gone"])
    assert process.kill.call_count == int(mode == "timeout")


def test_simulator_client_launch_passes_configuration_and_gpu_to_fake_runner(monkeypatch):
    subject = module.SimulatorRunner.__new__(module.SimulatorRunner)
    subject.args, subject.client_config, subject.deploy_args, subject.build_ctx = [object() for _ in range(4)]
    runtime = Mock()
    factory = Mock(return_value=runtime)
    monkeypatch.setattr(module, "SimulatorClientRunner", factory)
    clients = [object()]
    subject.client_run("custom", clients, "gpu-1")
    factory.assert_called_once_with(
        "custom", subject.args, clients, subject.client_config, subject.deploy_args, subject.build_ctx
    )
    runtime.run.assert_called_once_with("gpu-1")


def test_simulator_job_validation_stops_before_deployment_on_bad_archive(monkeypatch):
    subject = module.SimulatorRunner.__new__(module.SimulatorRunner)
    subject.args = Namespace(job_folder="/jobs/example")
    monkeypatch.setattr(module, "zip_directory_to_bytes", Mock(return_value=b"archive"))
    monkeypatch.setattr(module, "convert_legacy_zipped_app_to_job", Mock(return_value=b"converted"))
    validator = Mock()
    validator.validate.return_value = (False, "invalid job", {})
    monkeypatch.setattr(module, "JobMetaValidator", Mock(return_value=validator))
    with pytest.raises(RuntimeError, match="invalid job"):
        subject.validate_job_data()
    validator.validate.assert_called_once_with("example", b"converted")
