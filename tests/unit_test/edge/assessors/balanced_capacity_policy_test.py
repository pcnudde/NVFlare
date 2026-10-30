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

from collections import Counter
from types import SimpleNamespace

from nvflare.edge.assessors import buff_device_manager as subject


def test_balanced_sampling_redistributes_capacity_after_small_client_is_exhausted(monkeypatch):
    manager = subject.BuffDeviceManager(6)
    mapping = {"a0": "a", **{f"b{i}": "b" for i in range(4)}, **{f"c{i}": "c" for i in range(4)}}
    manager.device_client_map = mapping.copy()
    choices = []

    def choice(values):
        choices.append(True)
        assert len(choices) <= 2, "capacity redistribution did not consume remaining holes"
        return sorted(values)[0]

    random = SimpleNamespace(
        sample=lambda values, k: sorted(values)[:k],
        shuffle=lambda values: values.sort(key=lambda pair: pair[0]),
        choice=choice,
    )
    monkeypatch.setattr(subject, "random", random)
    chosen = manager._balanced_device_sampling(set(mapping), 6)
    assert len(chosen) == 6 and chosen <= mapping.keys()
    assert Counter(mapping[key] for key in chosen) == {"a": 1, "b": 3, "c": 2}
    assert manager.device_client_map == mapping


def test_sampling_without_client_mapping_caps_sample_to_available_devices(monkeypatch):
    manager = subject.BuffDeviceManager(5)
    monkeypatch.setattr(subject, "random", SimpleNamespace(sample=lambda values, k: sorted(values)[:k]))
    assert manager._balanced_device_sampling({"one", "two"}, 5) == {"one", "two"}
    assert manager._balanced_device_sampling(set(), 2) == set()
    assert manager._balanced_device_sampling({"one"}, 0) == set()
