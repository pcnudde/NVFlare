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

"""Tensor-at-a-time weighted aggregation for disk-backed PyTorch tensors."""

import math
import os
import tempfile
from functools import lru_cache
from typing import Any, Iterator, Mapping, Optional, Tuple

import torch

from nvflare.app_common.abstract.fl_model import ParamsType
from nvflare.app_common.aggregators.weighted_aggregation_helper import WeightedAggregationHelper
from nvflare.app_common.utils.lazy_value import is_lazy_value

from .lazy_tensor_dict import (
    TensorMetadata,
    _LazyRef,
    _TempDirRef,
    materialize,
    metadata_of,
    safetensors_dtype,
    safetensors_refs,
    write_safetensors,
)


def _output_dtype(dtype: torch.dtype) -> torch.dtype:
    """Floats and complex values keep their dtype; integer and bool inputs average as the default float."""
    return dtype if dtype.is_floating_point or dtype.is_complex else torch.get_default_dtype()


def _accumulation_dtype(dtype: torch.dtype) -> torch.dtype:
    """The output dtype, except that reduced-precision floats accumulate in float32."""
    output_dtype = _output_dtype(dtype)
    return torch.float32 if output_dtype.is_floating_point and output_dtype.itemsize < 4 else output_dtype


def _output_metadata(item: TensorMetadata) -> TensorMetadata:
    """Header entry of the aggregate for contributions with the given metadata."""
    if item.dtype.startswith(("F", "BF", "C")):
        return item
    dtype = torch.get_default_dtype()
    return TensorMetadata(
        shape=item.shape, dtype=safetensors_dtype(dtype), nbytes=math.prod(item.shape) * dtype.itemsize
    )


@lru_cache(maxsize=None)
def _promotes_in_place(source: torch.dtype, target: torch.dtype) -> bool:
    """Whether ``target.add_(source)`` accepts the source dtype without an explicit conversion."""
    try:
        return torch.promote_types(source, target) == target
    except RuntimeError:  # float8 dtypes take part in no promotion
        return False


def _accumulate(accumulator: torch.Tensor, tensor: torch.Tensor, weight: float) -> None:
    if not _promotes_in_place(tensor.dtype, accumulator.dtype):
        tensor = tensor.to(accumulator.dtype)
    accumulator.add_(tensor, alpha=weight)


def _check_contribution_value(key: str, value) -> None:
    if not (is_lazy_value(value) or isinstance(value, torch.Tensor)):
        raise TypeError(
            f"lazy aggregation requires torch.Tensor values or lazy tensor refs, got {type(value)} for '{key}'"
        )


class LazyWeightedAggregationHelper(WeightedAggregationHelper):
    """Weighted aggregation of disk-backed tensors, one model key at a time.

    ``add()`` only records contribution refs. ``get_result()`` materializes one
    contribution tensor at a time per key, accumulates it into a single accumulator and
    streams the aggregate into a new safetensors file below ``spill_dir``. The result
    is a dict of ``_LazyRef``; the file is removed once every ref to it is released.

    The result is always a full model: for ``ParamsType.DIFF`` contributions the mean
    difference is added to ``base_model`` and keys without contributions are copied
    from it.

    Args:
        spill_dir: directory receiving one sub-directory per aggregate, normally the
            job's tensor disk offload root.
        exclude_vars: regex of keys excluded from aggregation.
        base_model: current global model (tensors or lazy refs), required for DIFF.
        abort_signal: optional signal checked between tensors.
    """

    def __init__(
        self,
        spill_dir: str,
        exclude_vars: Optional[str] = None,
        base_model: Optional[Mapping[str, Any]] = None,
        abort_signal=None,
    ):
        super().__init__(exclude_vars=exclude_vars)
        self.spill_dir = spill_dir
        self.base_model = base_model or {}
        self.abort_signal = abort_signal

    def add(self, data, weight, contributor_name, contribution_round):
        """Record one contribution without loading tensor data."""
        with self.lock:
            for k, v in data.items():
                if self.exclude_vars is not None and self.exclude_vars.search(k):
                    self.skipped_keys.add(k)
                    continue
                _check_contribution_value(k, v)
                self.key_contribution_counts[k] = self.key_contribution_counts.get(k, 0) + 1
                self.total.setdefault(k, []).append((v, weight))
                self.counts[k] = self.counts.get(k, 0.0) + weight
            self.history.append({"contributor_name": contributor_name, "round": contribution_round, "weight": weight})

    def aggregate(self, params_type):
        """Aggregate into a full model on disk; DIFF contributions are applied to the base model."""
        if params_type is None:
            return self.get_result(), None
        return self.get_result(params_type), ParamsType.FULL

    def get_result(self, params_type: ParamsType = ParamsType.FULL) -> dict:
        """Write the weighted mean (applied to the base model for DIFF) to disk and return lazy refs."""
        with self.lock:
            try:
                if params_type == ParamsType.FULL:
                    base = {}
                    keys = sorted(self.total)
                elif params_type == ParamsType.DIFF:
                    base = self.base_model
                    unknown = sorted(set(self.total) - set(base))
                    if unknown:
                        raise ValueError(f"DIFF contains keys absent from the global model: {unknown}")
                    keys = sorted(base)
                else:
                    raise ValueError(f"lazy aggregation does not support params_type {params_type}")
                self.last_aggregation_stats = self._compute_aggregation_stats()
                if not self.total:
                    return dict(base)
                metadata = {key: self._result_metadata(key, base) for key in keys}
                return self._spill(metadata, self._result_tensors(keys, base))
            finally:
                self.reset_stats()
                self.base_model = {}  # let the previous aggregate's files go once the caller drops its refs

    def _result_metadata(self, key: str, base: Mapping[str, Any]) -> TensorMetadata:
        pending = self.total.get(key)
        if not pending:
            return metadata_of(base[key])
        first = metadata_of(pending[0][0])
        for value, _ in pending[1:]:
            if metadata_of(value) != first:
                raise ValueError(f"tensor '{key}' has different shape or dtype across contributions")
        if key in base and metadata_of(base[key]) != first:
            raise ValueError(f"DIFF tensor '{key}' does not match the shape or dtype of the global model")
        return _output_metadata(first)

    def _result_tensors(self, keys, base: Mapping[str, Any]) -> Iterator[Tuple[str, torch.Tensor]]:
        for key in keys:
            self._check_abort()
            pending = self.total.get(key)
            tensor = self._aggregate_key(pending, base.get(key)) if pending else materialize(base[key])
            yield key, tensor
            del tensor

    def _aggregate_key(self, pending, base_value) -> torch.Tensor:
        accumulator = None
        total_weight = 0.0
        for value, weight in pending:
            self._check_abort()
            tensor = materialize(value)
            if accumulator is None:
                output_dtype = _output_dtype(tensor.dtype)
                # A _LazyRef materializes a private copy that can become the accumulator; anything else may
                # still be referenced by its owner and must not be modified in place.
                accumulator = tensor.to(_accumulation_dtype(tensor.dtype), copy=not isinstance(value, _LazyRef))
                accumulator.mul_(weight)
            else:
                _accumulate(accumulator, tensor, weight)
            total_weight += weight
            del tensor
        accumulator.div_(total_weight)
        if base_value is not None:
            _accumulate(accumulator, materialize(base_value), 1.0)
        return accumulator.to(output_dtype)

    def _spill(self, metadata: dict, tensors: Iterator[Tuple[str, torch.Tensor]]) -> dict:
        temp_dir = tempfile.mkdtemp(prefix="nvflare_aggregate_", dir=self.spill_dir)
        temp_ref = _TempDirRef(temp_dir)
        file_path = os.path.join(temp_dir, "model.safetensors")
        try:
            write_safetensors(file_path, metadata, tensors)
        except BaseException:
            temp_ref.cleanup()
            raise
        return safetensors_refs(file_path, temp_ref)

    def _check_abort(self):
        if self.abort_signal is not None and self.abort_signal.triggered:
            raise RuntimeError("lazy aggregation aborted")
