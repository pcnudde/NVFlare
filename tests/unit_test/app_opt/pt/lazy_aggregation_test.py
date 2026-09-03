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

import gc
import os
import types
import weakref

import pytest
import torch
from safetensors.torch import load_file, save_file

from nvflare.app_common.abstract.fl_model import ParamsType
from nvflare.app_common.aggregators.weighted_aggregation_helper import AggregationStatsKey, WeightedAggregationHelper
from nvflare.app_opt.pt.lazy_aggregation import LazyWeightedAggregationHelper
from nvflare.app_opt.pt.lazy_tensor_dict import LazyTensorDict, _LazyRef, safetensors_refs


def _lazy_model(root, name, tensors) -> dict:
    """Write one chunk file per tensor, like DiskTensorConsumer does, and return lazy refs."""
    model_dir = root / name
    model_dir.mkdir()
    key_to_file = {}
    for index, (key, tensor) in enumerate(tensors.items()):
        file_path = str(model_dir / f"chunk_{index}.safetensors")
        save_file({key: tensor}, file_path)
        key_to_file[key] = (file_path, key)
    lazy_dict = LazyTensorDict(key_to_file=key_to_file, temp_dir=str(model_dir))
    return {key: lazy_dict.make_lazy_ref(key) for key in tensors}


def _add(helper, model, weight, contributor):
    helper.add(data=model, weight=weight, contributor_name=contributor, contribution_round=0)


def _load(result, key):
    ref = result[key]
    assert isinstance(ref, _LazyRef)
    return ref.materialize()


def _refs(path, tensors) -> dict:
    save_file(tensors, path)
    return safetensors_refs(str(path))


@pytest.fixture
def spill_dir(tmp_path):
    path = tmp_path / "spill"
    path.mkdir()
    return str(path)


def test_weighted_full_matches_in_memory_helper(tmp_path, spill_dir):
    first = {"z": torch.tensor([2.0, 4.0]), "a": torch.tensor([1.0, 3.0, 5.0])}
    second = {"z": torch.tensor([8.0, 10.0]), "a": torch.tensor([7.0, 9.0, 11.0])}
    helper = LazyWeightedAggregationHelper(spill_dir=spill_dir)
    _add(helper, _lazy_model(tmp_path, "first", first), 1.0, "site-1")
    _add(helper, _lazy_model(tmp_path, "second", second), 3.0, "site-2")
    stats = helper.get_aggregation_stats()

    result = helper.get_result(ParamsType.FULL)

    reference = WeightedAggregationHelper()
    reference.add(first, 1.0, "site-1", 0)
    reference.add(second, 3.0, "site-2", 0)
    expected = reference.get_result()
    assert list(result) == ["a", "z"]
    assert torch.equal(_load(result, "a"), expected["a"])
    assert torch.equal(_load(result, "z"), expected["z"])
    assert stats[AggregationStatsKey.ACCEPTED_CONTRIBUTIONS] == 2
    assert stats[AggregationStatsKey.CONTRIBUTORS] == ["site-1", "site-2"]
    assert helper.last_aggregation_stats[AggregationStatsKey.FULLY_MATCHED_KEYS] == 2
    (file_path,) = {ref.file_path for ref in result.values()}  # one sorted file below the spill dir
    assert os.path.dirname(os.path.dirname(file_path)) == spill_dir
    assert list(load_file(file_path)) == ["a", "z"]


def test_next_aggregation_deletes_the_previous_aggregate_and_the_consumed_contributions(tmp_path, spill_dir):
    first = _lazy_model(tmp_path, "first", {"w": torch.ones(2)})
    contribution_dir = os.path.dirname(first["w"].file_path)
    helper = LazyWeightedAggregationHelper(spill_dir=spill_dir)
    _add(helper, first, 1.0, "site-1")
    previous = helper.get_result()
    previous_dir = os.path.dirname(previous["w"].file_path)
    assert not os.path.exists(contribution_dir)
    assert os.path.isdir(previous_dir)

    helper = LazyWeightedAggregationHelper(spill_dir=spill_dir, base_model=previous)
    _add(helper, _lazy_model(tmp_path, "second", {"w": torch.full((2,), 3.0)}), 1.0, "site-1")
    result = helper.get_result(ParamsType.DIFF)

    assert torch.equal(_load(result, "w"), torch.tensor([4.0, 4.0]))  # the base stayed readable while aggregating
    assert not os.path.exists(previous_dir)
    assert os.listdir(spill_dir) == [os.path.basename(os.path.dirname(result["w"].file_path))]


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_reduced_precision_inputs_accumulate_in_float32_and_keep_their_dtype(tmp_path, spill_dir, dtype):
    helper = LazyWeightedAggregationHelper(spill_dir=spill_dir)
    _add(helper, _lazy_model(tmp_path, "first", {"w": torch.tensor([32768.0], dtype=dtype)}), 1.0, "site-1")
    _add(helper, _lazy_model(tmp_path, "second", {"w": torch.tensor([32768.0], dtype=dtype)}), 3.0, "site-2")

    result = _load(helper.get_result(), "w")

    assert result.dtype == dtype
    assert result.item() == 32768.0  # a float16 accumulator would have overflowed to inf


def test_integer_inputs_average_as_the_default_float_dtype(tmp_path, spill_dir):
    helper = LazyWeightedAggregationHelper(spill_dir=spill_dir)
    _add(helper, _lazy_model(tmp_path, "first", {"n": torch.tensor([10, 20])}), 2.0, "site-1")
    _add(helper, _lazy_model(tmp_path, "second", {"n": torch.tensor([5, 10])}), 3.0, "site-2")

    result = _load(helper.get_result(), "n")

    assert result.dtype == torch.get_default_dtype()
    assert torch.equal(result, torch.tensor([7.0, 14.0]))


def test_diff_adds_weighted_mean_to_base_and_copies_missing_keys(tmp_path, spill_dir):
    base = _refs(
        tmp_path / "base.safetensors", {"changed": torch.tensor([10.0, 20.0]), "unchanged": torch.tensor([7.0])}
    )
    helper = LazyWeightedAggregationHelper(spill_dir=spill_dir, base_model=base)
    _add(helper, _lazy_model(tmp_path, "first", {"changed": torch.tensor([1.0, 3.0])}), 1.0, "site-1")
    _add(helper, _lazy_model(tmp_path, "second", {"changed": torch.tensor([5.0, 7.0])}), 3.0, "site-2")

    result = helper.get_result(ParamsType.DIFF)

    assert list(result) == ["changed", "unchanged"]
    assert torch.equal(_load(result, "changed"), torch.tensor([14.0, 26.0]))
    assert torch.equal(_load(result, "unchanged"), torch.tensor([7.0]))


def test_diff_accepts_in_memory_base_model(tmp_path, spill_dir):
    base = {"w": torch.tensor([10.0])}
    helper = LazyWeightedAggregationHelper(spill_dir=spill_dir, base_model=base)
    _add(helper, _lazy_model(tmp_path, "delta", {"w": torch.tensor([2.0])}), 1.0, "site-1")

    assert torch.equal(_load(helper.get_result(ParamsType.DIFF), "w"), torch.tensor([12.0]))
    assert torch.equal(base["w"], torch.tensor([10.0]))


def test_diff_without_contributions_returns_the_base_refs(tmp_path, spill_dir):
    base = _refs(tmp_path / "base.safetensors", {"w": torch.ones(1)})
    helper = LazyWeightedAggregationHelper(spill_dir=spill_dir, base_model=base)

    result = helper.get_result(ParamsType.DIFF)

    assert result == base
    assert result["w"] is base["w"]
    assert os.listdir(spill_dir) == []


def test_diff_rejects_keys_absent_from_base_model(tmp_path, spill_dir):
    base = _refs(tmp_path / "base.safetensors", {"w": torch.ones(2)})
    helper = LazyWeightedAggregationHelper(spill_dir=spill_dir, base_model=base)
    _add(helper, _lazy_model(tmp_path, "delta", {"unexpected": torch.ones(2)}), 1.0, "site-1")

    with pytest.raises(ValueError, match="absent from the global model"):
        helper.get_result(ParamsType.DIFF)

    assert os.listdir(spill_dir) == []


def test_sparse_keys_and_exclude_vars(tmp_path, spill_dir):
    first = {"shared": torch.tensor([2.0]), "only_first": torch.tensor([3.0]), "bias": torch.ones(1)}
    second = {"shared": torch.tensor([4.0]), "only_second": torch.tensor([7.0]), "bias": torch.ones(1)}
    helper = LazyWeightedAggregationHelper(spill_dir=spill_dir, exclude_vars="bias")
    _add(helper, _lazy_model(tmp_path, "first", first), 1.0, "site-1")
    _add(helper, _lazy_model(tmp_path, "second", second), 1.0, "site-2")
    stats = helper.get_aggregation_stats()

    result = helper.get_result()

    assert torch.equal(_load(result, "shared"), torch.tensor([3.0]))
    assert torch.equal(_load(result, "only_first"), torch.tensor([3.0]))
    assert torch.equal(_load(result, "only_second"), torch.tensor([7.0]))
    assert "bias" not in result
    assert stats[AggregationStatsKey.FULLY_MATCHED_KEYS] == 1
    assert stats[AggregationStatsKey.PARTIALLY_MATCHED_KEYS] == 2
    assert stats[AggregationStatsKey.SKIPPED_KEYS] == 1


def test_in_memory_tensors_are_accepted_and_left_untouched(spill_dir):
    first = {"w": torch.tensor([2.0, 6.0])}
    second = {"w": torch.tensor([5.0, 9.0])}
    helper = LazyWeightedAggregationHelper(spill_dir=spill_dir)
    _add(helper, first, 1.0, "site-1")
    _add(helper, second, 2.0, "site-2")

    result = helper.get_result()

    assert torch.equal(_load(result, "w"), torch.tensor([4.0, 8.0]))
    assert torch.equal(first["w"], torch.tensor([2.0, 6.0]))


def test_non_tensor_values_are_rejected(spill_dir):
    helper = LazyWeightedAggregationHelper(spill_dir=spill_dir)

    with pytest.raises(TypeError, match="lazy aggregation requires"):
        _add(helper, {"w": [1.0, 2.0]}, 1.0, "site-1")


def test_shape_mismatch_across_contributions_is_rejected(tmp_path, spill_dir):
    helper = LazyWeightedAggregationHelper(spill_dir=spill_dir)
    _add(helper, _lazy_model(tmp_path, "first", {"w": torch.ones(2)}), 1.0, "site-1")
    _add(helper, _lazy_model(tmp_path, "second", {"w": torch.ones(3)}), 1.0, "site-2")

    with pytest.raises(ValueError, match="different shape or dtype"):
        helper.get_result()

    assert os.listdir(spill_dir) == []


@pytest.mark.parametrize("failure,expected_error", [("abort", RuntimeError), ("corrupt", ValueError)])
def test_failures_leave_no_partial_output(tmp_path, spill_dir, failure, expected_error):
    model = _lazy_model(tmp_path, "model", {"w": torch.ones(2)})
    signal = types.SimpleNamespace(triggered=failure == "abort")
    helper = LazyWeightedAggregationHelper(spill_dir=spill_dir, abort_signal=signal)
    _add(helper, model, 1.0, "site-1")
    if failure == "corrupt":
        with open(model["w"].file_path, "wb") as f:
            f.write(b"not safetensors")

    with pytest.raises(expected_error):
        helper.get_result()

    assert os.listdir(spill_dir) == []
    assert helper.get_aggregation_stats()[AggregationStatsKey.ACCEPTED_CONTRIBUTIONS] == 0


def test_materializes_one_input_tensor_at_a_time(tmp_path, spill_dir, monkeypatch):
    tensors = {"first": torch.ones(1024), "second": torch.ones(2048), "third": torch.ones(4096)}
    first = _lazy_model(tmp_path, "first_model", tensors)
    second = _lazy_model(tmp_path, "second_model", tensors)
    active = {"bytes": 0, "max": 0}
    calls = []
    original = _LazyRef.materialize

    def release(size):
        active["bytes"] -= size

    def tracked(ref):
        tensor = original(ref)
        size = tensor.numel() * tensor.element_size()
        calls.append(ref.key)
        active["bytes"] += size
        active["max"] = max(active["max"], active["bytes"])
        weakref.finalize(tensor, release, size)
        return tensor

    monkeypatch.setattr(_LazyRef, "materialize", tracked)
    helper = LazyWeightedAggregationHelper(spill_dir=spill_dir)
    _add(helper, first, 1.0, "site-1")
    _add(helper, second, 1.0, "site-2")

    result = helper.get_result()
    gc.collect()

    largest = max(t.numel() * t.element_size() for t in tensors.values())
    assert active["max"] <= 2 * largest  # the first input doubles as the accumulator, plus one more input
    assert active["bytes"] == 0
    assert calls == ["first", "first", "second", "second", "third", "third"]
    assert set(result) == set(tensors)
