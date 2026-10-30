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

torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")

from nvflare.app_opt.pt import tensor_downloader as subject  # noqa: E402


def test_tensor_consumer_merges_real_safetensors_chunks_and_preserves_previous_results():
    first = subject.save_tensors({"w": torch.tensor([1.0, 2.0])})
    second = subject.save_tensors({"bias": torch.tensor([3.0])})
    existing = {"previous": 7}
    consumer = subject.TensorConsumer(None, {})
    result = consumer.consume_items([first, second], existing)
    assert result is existing and result["previous"] == 7
    torch.testing.assert_close(result["w"], torch.tensor([1.0, 2.0]))
    torch.testing.assert_close(result["bias"], torch.tensor([3.0]))
    callback = Mock(side_effect=lambda tensors, scale: {"scaled": tensors["w"] * scale})
    transformed = subject.TensorConsumer(callback, {"scale": 2}).consume_items([first], None)
    torch.testing.assert_close(transformed["scaled"], torch.tensor([2.0, 4.0]))
    assert "w" not in transformed
    callback.assert_called_once()
    with pytest.raises(TypeError, match="items must be list"):
        consumer.consume_items(first, {})
    with pytest.raises(ValueError, match="must be callable"):
        subject.TensorConsumer(4, {})


def test_add_tensors_registers_lazy_downloadable_without_starting_a_transfer():
    downloader = Mock()
    reference = subject.add_tensors(downloader, {"w": torch.arange(3)}, max_chunk_size=64)
    registered = downloader.add_object.call_args.args[0]
    assert reference is downloader.add_object.return_value
    assert registered.get_item_count() == 1
    assert registered.get_item_size(0) == 24
    torch.testing.assert_close(subject.load_tensors(registered.produce_item(0))["w"], torch.arange(3))
    registered.release()
    registered.prefetch_item(0)
    assert registered.get_item_size(0) is None
