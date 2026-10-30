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

import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from nvflare.apis.fl_constant import FLContextKey
from nvflare.apis.fl_context import FLContext
from nvflare.app_common.widgets import job_stats_reporter as subject


def test_sampler_start_contains_optional_probe_failures_without_starting_real_thread(monkeypatch):
    fake_psutil = SimpleNamespace(Process=Mock(side_effect=RuntimeError("permission")))
    fake_nvml = SimpleNamespace(
        nvmlInit=Mock(),
        nvmlDeviceGetCount=Mock(side_effect=RuntimeError("driver")),
        nvmlShutdown=Mock(side_effect=RuntimeError("shutdown")),
    )
    monkeypatch.setitem(sys.modules, "psutil", fake_psutil)
    monkeypatch.setitem(sys.modules, "pynvml", fake_nvml)
    fake_thread = Mock()
    monkeypatch.setattr(
        subject, "threading", SimpleNamespace(Event=Mock(return_value=Mock()), Thread=Mock(return_value=fake_thread))
    )
    sampler = subject._ResourceSampler(0.1)
    sampler.start()
    fake_thread.start.assert_called_once()
    fake_nvml.nvmlShutdown.assert_called_once()
    assert sampler.psutil is None
    assert sampler.pynvml is None
    assert sampler.gpu_handles == []
    assert set(sampler.samples[0]) == {"time"}
    sampler.stop()
    fake_thread.join.assert_called_once_with(timeout=1.0)


def test_sampling_tolerates_disappearing_processes_battery_and_gpu_probes():
    sampler = subject._ResourceSampler(1)
    process = Mock()
    process.children.side_effect = RuntimeError("exited")
    process.memory_info.side_effect = RuntimeError("exited")
    process.cpu_percent.side_effect = RuntimeError("exited")
    sampler.process = process
    sampler.psutil = SimpleNamespace(sensors_battery=Mock(side_effect=RuntimeError("unsupported")))
    sampler._sample(True)
    assert set(sampler.samples[0]) == {"time"}
    assert sampler._process_family() == [process]
    nvml = SimpleNamespace(
        nvmlDeviceGetProcessUtilization=Mock(side_effect=RuntimeError("unsupported")),
        nvmlDeviceGetAccountingStats=Mock(side_effect=RuntimeError("unsupported")),
        nvmlDeviceGetComputeRunningProcesses=Mock(side_effect=RuntimeError("unsupported")),
    )
    sampler.pynvml = nvml
    assert sampler._sample_process_gpu("gpu", {5}) == (None, None)


def test_gpu_sampling_excludes_unrelated_invalid_and_unavailable_observations():
    sampler = subject._ResourceSampler(1)
    sampler.pynvml = SimpleNamespace(
        NVML_VALUE_NOT_AVAILABLE=-1,
        nvmlDeviceGetProcessUtilization=lambda *_: [
            SimpleNamespace(pid=5, smUtil=-1, timeStamp=1),
            SimpleNamespace(pid=9, smUtil=90, timeStamp=1),
        ],
        nvmlDeviceGetAccountingStats=lambda *args: SimpleNamespace(gpuUtilization=25),
        nvmlDeviceGetComputeRunningProcesses=lambda *_: [
            SimpleNamespace(pid=9, usedGpuMemory=100),
            SimpleNamespace(pid=5, usedGpuMemory=-1),
        ],
    )
    assert sampler._sample_process_gpu("gpu", {5}) == (25, None)


def test_client_after_send_and_stop_cleanup_sampler_state_exactly_once():
    reporter = subject.JobStatsReporter()
    sampler = Mock()
    reporter._client_task_info = {"task": {"sampler": sampler}}
    reporter._client_task_telemetry = {"task": {"execution_time": 1}}
    ctx = FLContext()
    ctx.set_prop(FLContextKey.TASK_ID, "task", private=True, sticky=False)
    reporter._handle_client_after_send(ctx)
    reporter._handle_client_after_send(ctx)
    sampler.stop.assert_called_once()
    assert reporter._client_task_info == {}
    assert reporter._client_task_telemetry == {}
    second = Mock()
    reporter._client_task_info = {"other": {"sampler": second}}
    reporter._client_task_telemetry = {"other": {}}
    reporter._stop_client_samplers(clear=True)
    reporter._stop_client_samplers(clear=True)
    second.stop.assert_called_once()
    assert reporter._client_task_info == {}
    assert reporter._client_task_telemetry == {}


def test_lazy_payload_sizes_are_derived_from_file_and_unknown_references_stay_unknown(tmp_path):
    class TensorDownloadRef:
        pass

    ref = TensorDownloadRef()
    path = tmp_path / "tensor"
    path.write_bytes(b"abcdef")
    ref.file_path = str(path)
    assert subject._estimate_size_bytes(ref) == 6
    path.unlink()
    assert subject._estimate_payload_mb(ref) is None
    del ref.file_path
    assert subject._estimate_payload_mb(ref) is None
    ref.size = 5
    assert subject._estimate_size_bytes(ref) == 5

    class BrokenTensor:
        def numel(self):
            raise RuntimeError("lazy")

        def element_size(self):
            return 8

    assert subject._estimate_size_bytes(BrokenTensor()) > 0


def test_report_path_and_format_boundaries(tmp_path):
    assert subject._format_mb(None) == "N/A"
    assert subject._format_mb(2048) == "2.00 GB"
    with pytest.raises(ValueError, match="non-empty string"):
        subject._validate_report_filename(None, "report")
    with pytest.raises(ValueError, match="under"):
        subject._prepare_output_path(str(tmp_path), "../escape")
