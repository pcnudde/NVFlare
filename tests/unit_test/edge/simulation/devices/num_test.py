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

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from nvflare.apis.dxo import DXO, from_dict
from nvflare.edge.simulation.devices import num
from nvflare.edge.simulation.devices.num import NumDevice, NumDeviceFactory, NumProcessor
from nvflare.edge.web.models.job_response import JobResponse
from nvflare.edge.web.models.task_response import TaskResponse
from tests.timing_utils import isolate_time


@pytest.mark.parametrize("kind", ["device", "processor"])
@pytest.mark.parametrize("value,expected", [(4, 5), (-2, -1), (None, 1)])
def test_numeric_task_increments_value_and_preserves_dxo_format(monkeypatch, kind, value, expected):
    clock = isolate_time(monkeypatch, num)
    clock.sleep = Mock()
    uniform = Mock(return_value=0.25)
    monkeypatch.setattr(num, "random", SimpleNamespace(uniform=uniform))
    worker = NumDevice("dev", 0.1, 0.5) if kind == "device" else NumProcessor(0.1, 0.5)
    task = TaskResponse("OK", task_data=DXO("number", {} if value is None else {"value": value}).to_dict())

    result = worker.do_task(task) if kind == "device" else worker.process_task(task)

    assert from_dict(result).data_kind == "number"
    assert from_dict(result).data == {"value": expected}
    uniform.assert_called_once_with(0.1, 0.5)
    clock.sleep.assert_called_once_with(0.25)


@pytest.mark.parametrize("kind", ["device", "processor"])
def test_non_numeric_payload_rejected_without_waiting(monkeypatch, kind):
    clock = isolate_time(monkeypatch, num)
    clock.sleep = Mock()
    worker = NumDevice("dev") if kind == "device" else NumProcessor()
    task = TaskResponse("OK", task_data=DXO("weights", {"w": 1}).to_dict())
    with pytest.raises(ValueError, match="bad model data kind"):
        (worker.do_task if kind == "device" else worker.process_task)(task)
    clock.sleep.assert_not_called()


def test_factory_and_processor_expose_assigned_device_job_information():
    device = NumDeviceFactory(0.01, 0.1).make_device("dev-1")
    device.set_job("job", "edge", "number", {"rounds": 3})
    processor = NumProcessor()
    assert processor.device_info is None and processor.user_info is None
    processor.device = device
    assert processor.device_info.device_id == "dev-1"
    assert processor.user_info.user_id == "dev-1"
    assert (processor.job_id, processor.job_name, processor.job_method, processor.job_data) == (
        "job",
        "edge",
        "number",
        {"rounds": 3},
    )
    assert (device.min_train_time, device.max_train_time) == (0.01, 0.1)
    processor.setup(JobResponse("OK"))
    processor.shutdown()
    processor.cancel()
