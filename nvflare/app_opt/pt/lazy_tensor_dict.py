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

"""PT lazy tensor references used by tensor disk offload.

When `enable_tensor_disk_offload=True`, incoming streamed tensor payloads are written
to temporary safetensors files instead of being fully deserialized into memory.
`LazyTensorDict` maps item IDs to on-disk files, and `_LazyRef` defers loading until
`materialize()` is called by aggregation code.

Lazy refs are also the server's representation of a large global model: a persistor
can return refs into a checkpoint, and lazy aggregation streams its result into a new
safetensors file with `write_safetensors` and returns refs into it. FOBS serializes a
ref as an ordinary tensor, so clients and the wire format are unchanged.

This keeps peak memory lower for large models while still allowing deterministic
explicit cleanup via `cleanup()`, with GC as a fallback through `_TempDirRef`.
"""

import json
import logging
import math
import os
import re
import shutil
import struct
from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable, Mapping, Optional, Tuple

import torch
from safetensors import safe_open
from safetensors.torch import save as save_tensors

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TensorMetadata:
    """Shape, safetensors dtype name and byte size of a tensor, known without loading its data."""

    shape: Tuple[int, ...]
    dtype: str
    nbytes: int


def _dtype_size(dtype: str) -> int:
    if dtype == "BOOL":
        return 1
    match = re.match(r"^[A-Z]+(\d+)", dtype)
    if not match:
        raise ValueError(f"unsupported safetensors dtype: {dtype}")
    return max(1, int(match.group(1)) // 8)


@lru_cache(maxsize=4096)
def _read_file_metadata(file_path: str, mtime_ns: int, file_size: int) -> dict:
    del mtime_ns, file_size  # cache key only: a rewritten file is read again
    result = {}
    with safe_open(file_path, framework="pt", device="cpu") as tensor_file:
        for key in tensor_file.keys():
            tensor_slice = tensor_file.get_slice(key)
            shape = tuple(tensor_slice.get_shape())
            dtype = tensor_slice.get_dtype()
            result[key] = TensorMetadata(shape=shape, dtype=dtype, nbytes=math.prod(shape) * _dtype_size(dtype))
    return result


def read_safetensors_metadata(file_path: str) -> dict[str, TensorMetadata]:
    """Return the metadata of every tensor in a safetensors file, reading only its header."""
    stat = os.stat(file_path)
    return _read_file_metadata(os.path.realpath(file_path), stat.st_mtime_ns, stat.st_size)


@lru_cache(maxsize=None)
def safetensors_dtype(dtype: torch.dtype) -> str:
    probe = save_tensors({"t": torch.empty(0, dtype=dtype)})
    header_size = struct.unpack("<Q", probe[:8])[0]
    return json.loads(probe[8 : 8 + header_size])["t"]["dtype"]


def tensor_metadata(tensor: torch.Tensor) -> TensorMetadata:
    return TensorMetadata(
        shape=tuple(tensor.shape),
        dtype=safetensors_dtype(tensor.dtype),
        nbytes=tensor.numel() * tensor.element_size(),
    )


def is_lazy_tensor(value) -> bool:
    return callable(getattr(value, "materialize", None))


def materialize(value) -> torch.Tensor:
    """Return the tensor behind a lazy ref, or the value itself when it already is a tensor."""
    if is_lazy_tensor(value):
        value = value.materialize()
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"expected a torch.Tensor or lazy tensor ref but got {type(value)}")
    return value


def metadata_of(value) -> TensorMetadata:
    """Metadata of a tensor or lazy ref; refs without cheap metadata are materialized once."""
    get_metadata = getattr(value, "get_metadata", None)
    if callable(get_metadata):
        return get_metadata()
    return tensor_metadata(materialize(value))


def write_safetensors(
    file_path: str,
    metadata: Mapping[str, TensorMetadata],
    tensors: Iterable[Tuple[str, torch.Tensor]],
) -> None:
    """Stream tensors into a safetensors file, holding one tensor at a time.

    ``metadata`` fixes the header before any tensor exists; ``tensors`` must yield the
    same keys in the same order with matching shape and dtype.
    """
    if "__metadata__" in metadata:
        raise ValueError("'__metadata__' is reserved by safetensors")
    header = {}
    offset = 0
    for key, item in metadata.items():
        header[key] = {"dtype": item.dtype, "shape": list(item.shape), "data_offsets": [offset, offset + item.nbytes]}
        offset += item.nbytes
    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    header_bytes += b" " * (-len(header_bytes) % 8)

    expected_keys = iter(metadata)
    with open(file_path, "wb") as output:
        output.write(struct.pack("<Q", len(header_bytes)))
        output.write(header_bytes)
        for key, tensor in tensors:
            if key != next(expected_keys, None):
                raise ValueError(f"tensor '{key}' does not follow the declared header order")
            if tensor_metadata(tensor) != metadata[key]:
                raise ValueError(f"tensor '{key}' does not match its declared shape or dtype")
            output.write(memoryview(tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy()))
            del tensor
        if next(expected_keys, None) is not None:
            raise ValueError("fewer tensors than declared in the header")
        output.flush()
        os.fsync(output.fileno())


def _cleanup_temp_dir(path: str) -> None:
    try:
        shutil.rmtree(path)
    except FileNotFoundError:
        return
    except Exception as e:
        logger.warning("failed to cleanup tensor offload temp dir '%s': %s", path, e)


class _TempDirRef:
    """Reference-counted sentinel for a temp directory.

    Shared between LazyTensorDict and all _LazyRef instances created from it.
    The directory is deleted only when ALL holders are garbage collected.
    """

    def __init__(self, temp_dir: str):
        self.path = temp_dir
        self._deleted = False

    def cleanup(self):
        if not self._deleted:
            self._deleted = True
            _cleanup_temp_dir(self.path)

    def __del__(self):
        self.cleanup()

    def __deepcopy__(self, memo):
        # A copied payload must share the sentinel, otherwise either copy could
        # delete files the other still references.
        return self


class _LazyRef:
    """Lightweight placeholder for an on-disk tensor.

    Carries only file_path + key (~100 bytes). The tensor is loaded from disk
    only when materialize() is called, keeping memory near zero until then.

    Holds a reference to _TempDirRef to prevent premature cleanup. Refs into
    files that are not owned by the job (a user checkpoint) have no temp_ref.
    """

    def __init__(self, file_path: str, key: str, temp_ref: Optional[_TempDirRef] = None):
        self.file_path = file_path
        self.key = key
        self._temp_ref = temp_ref

    def materialize(self):
        """Load tensor from safetensors file. Opens mmap, copies data out, closes mmap."""
        with safe_open(self.file_path, framework="pt") as f:
            return f.get_tensor(self.key)

    def get_metadata(self) -> TensorMetadata:
        """Read tensor metadata without materializing its data."""
        try:
            return read_safetensors_metadata(self.file_path)[self.key]
        except KeyError as e:
            raise ValueError(f"safetensors file '{self.file_path}' has no tensor '{self.key}'") from e

    def __repr__(self):
        return f"_LazyRef({self.file_path!r}, key={self.key!r})"

    def __deepcopy__(self, memo):
        return _LazyRef(file_path=self.file_path, key=self.key, temp_ref=self._temp_ref)


def safetensors_refs(file_path: str, temp_ref: Optional[_TempDirRef] = None) -> dict[str, _LazyRef]:
    """Return a lazy ref for every tensor in a safetensors file."""
    return {key: _LazyRef(file_path, key, temp_ref) for key in read_safetensors_metadata(file_path)}


class LazyTensorDict:
    """Dict-like mapping of FOBS item_ids to on-disk safetensors files.

    Each entry maps an item_id to a (file_path, key) pair. Tensors are loaded
    via safetensors safe_open (mmap) on access.
    """

    def __init__(self, key_to_file: dict[str, tuple[str, str]], temp_dir: str):
        self._key_to_file = key_to_file
        self._temp_ref = _TempDirRef(temp_dir)

    def __getitem__(self, key):
        file_path, st_key = self._key_to_file[key]
        with safe_open(file_path, framework="pt") as f:
            return f.get_tensor(st_key)

    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default

    def keys(self):
        return self._key_to_file.keys()

    def __iter__(self):
        return iter(self._key_to_file)

    def items(self):
        for key in self._key_to_file:
            yield key, self[key]

    def values(self):
        for key in self._key_to_file:
            yield self[key]

    def __len__(self):
        return len(self._key_to_file)

    def __contains__(self, key):
        return key in self._key_to_file

    def make_lazy_ref(self, key) -> "_LazyRef":
        file_path, st_key = self._key_to_file[key]
        return _LazyRef(file_path=file_path, key=st_key, temp_ref=self._temp_ref)

    def cleanup(self):
        self._temp_ref.cleanup()
