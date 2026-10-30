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

from nvflare.fuel.utils import gpu_utils as subject


def test_memory_inventory_parses_requested_units_and_keeps_total_free_queries_separate(monkeypatch):
    query = Mock(
        side_effect=[["memory.total [MiB]", "1024 MiB", "2048 MiB"], ["memory.free [MiB]", "512 MiB", "1536 MiB"]]
    )
    monkeypatch.setattr(subject, "use_nvidia_smi", query)
    assert subject.get_host_gpu_memory_total() == [1024, 2048]
    assert subject.get_host_gpu_memory_free() == [512, 1536]
    assert [call.args[0] for call in query.call_args_list] == ["memory.total", "memory.free"]
    assert subject._parse_gpu_mem(None) == []
    with pytest.raises(RuntimeError, match="unit does not match"):
        subject._parse_gpu_mem(["header", "2 GiB"])
