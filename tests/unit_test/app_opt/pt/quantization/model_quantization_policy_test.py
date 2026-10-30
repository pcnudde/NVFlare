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

import threading
from unittest.mock import MagicMock

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("bitsandbytes")

from nvflare.apis.dxo import DXO, DataKind, MetaKey, from_shareable
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.app_opt.pt.quantization.dequantizer import ModelDequantizer
from nvflare.app_opt.pt.quantization.quantizer import ModelQuantizer
from tests.timing_utils import ObservedLock, ThreadGroup


@pytest.mark.parametrize("format", ["numpy", "torch"])
def test_float16_roundtrip_clamps_overflow_preserves_flags_and_clears_wire_metadata(format):
    values = np.array([-100000, -1.25, 2.5, 100000], dtype=np.float32)
    flags = np.array([True, False])
    half = np.array([1.0], dtype=np.float16)
    if format == "torch":
        values, flags, half = map(torch.as_tensor, (values, flags, half))
    dxo = DXO(DataKind.WEIGHT_DIFF, {"weights": values, "flags": flags, "half": half}, meta={"round": 3})
    quantizer = ModelQuantizer("FLOAT16")
    encoded = quantizer.process_dxo(dxo, Shareable(), FLContext())
    assert encoded.data["weights"].dtype == (np.float16 if format == "numpy" else torch.float16)
    assert encoded.data["flags"] is flags
    assert encoded.data["half"] is half
    assert encoded.get_meta_prop(MetaKey.PROCESSED_ALGORITHM) == "float16"
    assert encoded.get_meta_prop("source_datatype") == {"weights": "float32", "flags": "bool", "half": "float16"}
    assert encoded.get_meta_prop("quantized_flag") is True
    assert quantizer.process_dxo(encoded, Shareable(), FLContext()) is encoded
    shareable = encoded.to_shareable()
    restored = ModelDequantizer().process_dxo(encoded, shareable, FLContext())
    np.testing.assert_array_equal(restored.data["weights"], [-65504.0, -1.25, 2.5, 65504.0])
    assert restored.data["weights"].dtype == (np.float32 if format == "numpy" else torch.float32)
    assert restored.meta == {"round": 3}
    assert from_shareable(shareable).meta == {"round": 3}


@pytest.mark.parametrize("format", ["numpy", "torch"])
@pytest.mark.parametrize("constant", [True, False])
def test_adaptive_filter_roundtrip_restores_source_precision_and_quantized_shape(format, constant):
    values = torch.full((2, 3), -2.0) if constant else torch.linspace(-2, 5, 24).reshape(4, 6)
    original = values.numpy().copy()
    if format == "numpy":
        values = values.numpy()
    encoded = ModelQuantizer("adaquant").process_dxo(
        DXO(DataKind.WEIGHTS, {"weight": values}), Shareable(), FLContext()
    )
    state = encoded.get_meta_prop("quant_state")["weight"]
    decoded = ModelDequantizer().process_dxo(encoded, Shareable(), FLContext())
    tolerance = 0 if constant else state["norm"] / state["quantization_level"] / 2 + 1e-6
    np.testing.assert_allclose(decoded.data["weight"], original, rtol=0, atol=tolerance)
    assert decoded.data["weight"].dtype == (np.float32 if format == "numpy" else torch.float32)


@pytest.mark.parametrize("value", ["invalid", 42])
def test_quantizer_rejects_non_array_parameters(value):
    with pytest.raises(ValueError, match="Invalid source data type"):
        ModelQuantizer().quantization({"w": value}, FLContext())


def test_quantizer_rejects_unsupported_numeric_precision_and_algorithm():
    with pytest.raises(ValueError, match="param_name: w"):
        ModelQuantizer().quantization({"w": np.array([1.0], dtype=np.float64)}, FLContext())
    with pytest.raises(ValueError, match="Invalid quantization type"):
        ModelQuantizer("unknown")
    with pytest.raises(ValueError, match="Invalid quantization type"):
        ModelDequantizer().process_dxo(
            DXO(DataKind.WEIGHTS, {"w": np.ones(1)}, meta={MetaKey.PROCESSED_ALGORITHM: "unknown"}),
            Shareable(),
            FLContext(),
        )


@pytest.mark.parametrize("fail", [True, False])
def test_shared_data_quantizes_once_and_propagates_result_or_error_to_contending_wrapper(fail):
    values = {"w": np.ones(2, dtype=np.float32)}
    wrappers = [DXO(DataKind.WEIGHTS, values), DXO(DataKind.WEIGHTS, values)]
    quantizer = ModelQuantizer()
    entered, release = threading.Event(), threading.Event()
    lock = ObservedLock()
    quantizer._quantization_locks[id(values)] = lock
    quantizer._data_refcounts[id(values)] = 0
    error = ValueError("controlled quantization failure")
    original = quantizer.quantization

    def gated(params, fl_ctx):
        entered.set()
        release.wait()
        if fail:
            raise error
        return original(params, fl_ctx)

    quantizer.quantization = MagicMock(side_effect=gated)
    results, errors = [], []

    def process(dxo):
        try:
            results.append(quantizer.process_dxo(dxo, Shareable(), FLContext()))
        except ValueError as caught:
            errors.append(caught)

    with ThreadGroup(release) as workers:
        workers.start(process, wrappers[0])
        assert entered.wait(10)
        workers.start(process, wrappers[1])
        assert lock.blocked.wait(10)
        assert quantizer._data_refcounts[id(values)] == 2
    assert quantizer.quantization.call_count == 1
    if fail:
        assert errors == [error, error]
        assert results == []
    else:
        assert not errors
        assert len(results) == 2 and results[0] is results[1]
        assert results[0].get_meta_prop("quantized_flag") is True
    assert quantizer._quantization_locks == {}
    assert quantizer._data_refcounts == {}
    assert quantizer._quantized_results == {}
    assert quantizer._quantization_errors == {}


@pytest.mark.parametrize("format", ["numpy", "torch"])
@pytest.mark.parametrize("algorithm", ["blockwise8", "float4", "normfloat4"])
def test_bitsandbytes_cpu_kernel_roundtrip_with_device_transfer_boundary_controlled(monkeypatch, format, algorithm):
    pytest.importorskip("bitsandbytes", minversion="0.50.2")
    # Device placement is the controlled boundary; actual bitsandbytes kernels and state serialization run.
    monkeypatch.setattr(torch.Tensor, "cuda", lambda self, *args, **kwargs: self)
    values = torch.linspace(-2, 2, 128, dtype=torch.float32).reshape(8, 16)
    original = values.numpy().copy()
    if format == "numpy":
        values = values.numpy()
    encoded = ModelQuantizer(algorithm).process_dxo(DXO(DataKind.WEIGHTS, {"w": values}), Shareable(), FLContext())
    assert encoded.data["w"].dtype == (np.uint8 if format == "numpy" else torch.uint8)
    state = encoded.get_meta_prop("quant_state")["w"]
    assert "absmax" in state
    assert isinstance(state["absmax"], np.ndarray if format == "numpy" else torch.Tensor)
    if algorithm != "blockwise8":
        assert state["quant_type"] == ("fp4" if algorithm == "float4" else "nf4")
    decoded = ModelDequantizer().process_dxo(encoded, Shareable(), FLContext())
    np.testing.assert_allclose(decoded.data["w"], original, atol=0.35, rtol=0)
    assert decoded.data["w"].shape == original.shape
