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
from unittest.mock import Mock

import pytest

from nvflare.app_common.resource_consumers import gpu_resource_consumer as gpu
from nvflare.app_common.resource_consumers.list_resource_consumer import ListResourceConsumer


@pytest.fixture(autouse=True)
def device_environment(monkeypatch):
    monkeypatch.setenv("CUDA_DEVICE_ORDER", "original-order")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "original-visible")


def test_list_consumer_maps_assigned_devices_and_leaves_unrequested_types_untouched():
    consumer = ListResourceConsumer()
    assert os.environ["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID"
    consumer.consume({"cpu": [1]})
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "original-visible"
    consumer.consume({"gpu": [2, 4]})
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "2,4"
    consumer.consume({"gpu": []})
    assert os.environ["CUDA_VISIBLE_DEVICES"] == ""


@pytest.mark.parametrize("resources,error", [({1: 2}, None), ({5: 2}, "does not exist"), ({1: 4}, "free mem")])
def test_gpu_consumer_validates_identity_and_free_memory_before_exposing_devices(monkeypatch, resources, error):
    monkeypatch.setattr(gpu, "get_host_gpu_ids", Mock(return_value=[1, 3]))
    memory = Mock(return_value={1: 3072, 3: 8192})
    monkeypatch.setattr(gpu, "get_host_gpu_memory_free", memory)
    consumer = gpu.GPUResourceConsumer()
    if error:
        with pytest.raises(RuntimeError, match=error):
            consumer.consume(resources)
        assert os.environ["CUDA_VISIBLE_DEVICES"] == "original-visible"
    else:
        consumer.consume(resources)
        assert os.environ["CUDA_VISIBLE_DEVICES"] == "1"
    memory.assert_called_once_with(unit="MiB")
