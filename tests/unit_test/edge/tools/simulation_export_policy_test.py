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

from unittest.mock import Mock

import pytest

from nvflare.edge.simulation.device_task_processor import DeviceTaskProcessor
from nvflare.edge.tools.edge_job import EdgeJob
from nvflare.edge.widgets.tp_runner import TPRunner
from nvflare.edge.widgets.tpo_runner import TPORunner
from nvflare.job_config.file_source import FileSource


def test_file_simulation_export_adds_configuration_then_runtime_runner(tmp_path, monkeypatch):
    job = EdgeJob("edge", "method")
    add = Mock()
    monkeypatch.setattr(job, "to_clients", add)
    config = tmp_path / "devices.json"
    config.write_text('{"devices": 2}')
    job.configure_simulation_with_file(str(config))
    source = add.call_args_list[0].args[0]
    runner = add.call_args_list[1].args[0]
    assert isinstance(source, FileSource) and source.src_path == str(config) and source.app_folder_type == "config"
    assert isinstance(runner, TPRunner) and add.call_args_list[1].kwargs["id"] == "tp_runner"
    assert job.simulation_set
    with pytest.raises(RuntimeError, match="already configured"):
        job.configure_simulation_with_file(str(config))


def test_simulation_file_validation_happens_before_component_export(tmp_path, monkeypatch):
    job, add = EdgeJob("edge", "method"), Mock()
    monkeypatch.setattr(job, "to_clients", add)
    with pytest.raises(ValueError, match="does not exist"):
        job.configure_simulation_with_file(str(tmp_path / "missing"))
    config = tmp_path / "bad.json"
    config.write_text("invalid")
    with pytest.raises(ValueError, match="not a valid JSON"):
        job.configure_simulation_with_file(str(config))
    add.assert_not_called()
    assert not job.simulation_set


def test_object_simulation_export_wires_task_processor_id_and_resource_parameters(monkeypatch):
    job = EdgeJob("edge", "method")
    add = Mock(side_effect=["processor", None])
    monkeypatch.setattr(job, "to_clients", add)
    processor = Mock(spec=DeviceTaskProcessor)
    job.configure_simulation(processor, job_timeout=5, num_devices=3, num_workers=2)
    add.assert_any_call(processor, "task_processor")
    assert isinstance(add.call_args_list[1].args[0], TPORunner)
    assert add.call_args_list[1].args[1] == "tpo_runner" and job.simulation_set
    with pytest.raises(RuntimeError, match="already configured"):
        job.configure_simulation(processor)
