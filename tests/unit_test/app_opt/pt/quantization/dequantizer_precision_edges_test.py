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

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("bitsandbytes")

from nvflare.apis.fl_context import FLContext  # noqa: E402
from nvflare.app_opt.pt.quantization.dequantizer import ModelDequantizer  # noqa: E402


@pytest.mark.parametrize("format,dtype", [("numpy", "float16"), ("torch", "float16"), ("torch", "bfloat16")])
def test_adaptive_dequantization_without_quant_state_restores_exact_source_precision(format, dtype):
    values = np.array([-2.0, 0.5, 4.0], dtype=np.float32)
    if format == "torch":
        values = torch.as_tensor(values)
    params = {"w": values}
    dequantizer = ModelDequantizer()
    dequantizer.log_info = Mock()
    result = dequantizer.dequantization(params, {"w": {}}, "adaquant", {"w": dtype}, FLContext())
    assert result is params
    if format == "numpy":
        assert result["w"].dtype == np.float16
        np.testing.assert_array_equal(result["w"], [-2.0, 0.5, 4.0])
    else:
        assert result["w"].dtype == getattr(torch, dtype)
        torch.testing.assert_close(result["w"].float(), torch.tensor([-2.0, 0.5, 4.0]))
