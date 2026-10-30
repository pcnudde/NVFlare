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

from unittest.mock import Mock

import pytest

from nvflare.app_common.resource_managers import gpu_resource_manager as subject


@pytest.mark.parametrize(
    "arguments,reason",
    [
        ({"num_of_gpus": -1}, "greater"),
        ({"num_of_gpus": "1"}, "type int"),
        ({"mem_per_gpu_in_GiB": "4"}, "type int or float"),
        ({"mem_per_gpu_in_GiB": -1}, "greater"),
        ({"expiration_period": "1"}, "type int or float"),
        ({"expiration_period": -1}, "greater"),
        ({"ignore_host": "yes"}, "type bool"),
        ({"exclusive": 1}, "must be a bool"),
    ],
)
def test_invalid_gpu_pool_options_fail_before_host_probing(monkeypatch, arguments, reason):
    probe = Mock()
    monkeypatch.setattr(subject, "get_host_gpu_ids", probe)
    settings = {"num_of_gpus": 1, "mem_per_gpu_in_GiB": 4}
    settings.update(arguments)
    with pytest.raises(ValueError, match=reason):
        subject.GPUResourceManager(**settings)
    probe.assert_not_called()


@pytest.mark.parametrize("memory,reason", [([], "determine GPU memory"), ([1024], "exceeds available GPU memory")])
def test_host_gpu_pool_rejects_missing_or_insufficient_device_memory(monkeypatch, memory, reason):
    monkeypatch.delenv(subject.CUDA_VISIBLE_DEVICES, raising=False)
    monkeypatch.setattr(subject, "get_host_gpu_ids", Mock(return_value=[5]))
    monkeypatch.setattr(subject, "get_host_gpu_memory_total", Mock(return_value=memory))
    with pytest.raises((RuntimeError, ValueError), match=reason):
        subject.GPUResourceManager(1, 4)
