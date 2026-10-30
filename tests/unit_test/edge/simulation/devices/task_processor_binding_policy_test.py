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

from nvflare.edge.constants import EdgeApiStatus
from nvflare.edge.simulation.device_task_processor import DeviceTaskProcessor
from nvflare.edge.simulation.devices.tp import TPDeviceFactory, TPODeviceFactory
from nvflare.edge.web.models.task_response import TaskResponse


class RecordingProcessor(DeviceTaskProcessor):
    def __init__(self):
        super().__init__()
        self.jobs = []
        self.closed = False

    def setup(self, job):
        self.jobs.append((self.device.job_id, job))

    def process_task(self, task):
        return {"device": self.device.device_id, "task": task.task_name, "job": self.job_id}

    def shutdown(self):
        self.closed = True


def test_parser_factory_binds_identity_before_job_and_task_dispatch():
    processor = RecordingProcessor()
    parser = Mock()
    parser.get_endpoint.return_value = "https://edge.test"
    parser.get_processor.return_value = processor
    device = TPDeviceFactory(parser).make_device("device-one")
    parser.get_processor.assert_called_once_with({"device_id": "device-one", "user_id": "demo_id"})
    assert device.endpoint_url == "https://edge.test"
    assert processor.device_info is device.get_device_info()
    assert processor.user_info is device.get_user_info()
    assert processor.device_info.device_id == "device-one"
    device.set_job("job", "train", "edge", {"epochs": 2})
    bound_id, request = processor.jobs[0]
    assert bound_id == request.job_id == "job"
    assert request.status == EdgeApiStatus.OK
    assert (processor.job_name, processor.job_method, processor.job_data) == ("train", "edge", {"epochs": 2})
    assert device.do_task(TaskResponse(status=EdgeApiStatus.OK, task_name="fit")) == {
        "device": "device-one",
        "task": "fit",
        "job": "job",
    }
    device.shutdown()
    assert processor.closed


def test_object_factory_deep_copies_mutable_processor_state_per_device():
    prototype = RecordingProcessor()
    factory = TPODeviceFactory(prototype)
    one, two = factory.make_device("one"), factory.make_device("two")
    one.set_job("job-one", "train", "edge", {})
    assert one.processor is not two.processor
    assert one.processor is not prototype
    assert two.processor.device is two
    assert prototype.device is None
    assert len(one.processor.jobs) == 1
    assert two.processor.jobs == prototype.jobs == []
    assert two.endpoint_url == ""
