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


import numpy as np
import pytest

torch = pytest.importorskip("torch")

from nvflare.app_opt.pt.quantization.ada_quant import AdaQuantizer


@pytest.mark.parametrize("compression", [False, True])
def test_adaptive_roundtrip_bounds_error_and_preserves_shape(compression):
    values = torch.linspace(-4, 7, 120, dtype=torch.float32).reshape(10, 12)
    quantizer = AdaQuantizer(weight=0.01, compression=compression)
    encoded, state = quantizer.quantize(values)
    decoded = quantizer.dequantized(torch.as_tensor(encoded), state)
    assert decoded.shape == values.shape
    assert decoded.dtype == torch.float64
    assert state["offset"] == 4
    assert torch.max(torch.abs(decoded - values)).item() <= state["norm"] / state["quantization_level"] / 2 + 1e-6


def test_constant_tensor_uses_bool_sentinel_and_restores_offset():
    values = torch.full((2, 3), -7.0)
    encoded, state = AdaQuantizer().quantize(values)
    assert encoded.dtype == torch.bool
    assert encoded.numel() == 1
    assert "norm" not in state
    assert torch.equal(AdaQuantizer().dequantized(encoded, state), values.double())


def test_compression_stores_bytes_only_when_smaller_and_restores_exact_codes():
    values = torch.zeros(4096)
    values[-1] = 1
    quantizer = AdaQuantizer(compression=True)
    encoded, state = quantizer.quantize(values)
    assert encoded.dtype == torch.bool
    assert state["compressed_tensor"].dtype == np.uint8
    assert len(state["compressed_tensor"]) < values.numel()
    assert torch.equal(quantizer.dequantized(encoded, state), values.double())


def test_tiny_weight_keeps_original_tensor_when_no_smaller_dtype_exists():
    values = torch.tensor([0, 1], dtype=torch.uint8)
    encoded, state = AdaQuantizer(weight=1e-12).quantize(values)
    assert encoded is values
    assert state == {}


@pytest.mark.parametrize("weight, expected_dtype", [(0.01, "u1"), (1e-5, "<u2")])
def test_level_selection_chooses_smallest_supported_unsigned_dtype(weight, expected_dtype):
    norm, levels, dtype = AdaQuantizer(weight=weight).get_number_of_quantization_levels(4, torch.tensor([0.0, 1.0]))
    assert norm == 1
    assert dtype == expected_dtype
    assert (levels + 1) & levels == 0


def test_level_selection_rejects_unsupported_intermediate_integer_width():
    with pytest.raises(RuntimeError, match="Invalid element_bits"):
        AdaQuantizer(weight=1e-9).get_number_of_quantization_levels(4, torch.tensor([0.0, 1.0]))
