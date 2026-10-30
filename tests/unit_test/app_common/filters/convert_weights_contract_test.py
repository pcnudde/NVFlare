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

import numpy as np
import pytest

from nvflare.apis.dxo import DXO, DataKind, MetaKey
from nvflare.apis.fl_constant import FLContextKey
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.app_common.filters.convert_weights import ConvertWeights


@pytest.mark.parametrize("direction", [ConvertWeights.WEIGHTS_TO_DIFF, ConvertWeights.DIFF_TO_WEIGHTS])
def test_weight_conversion_uses_original_task_base_and_preserves_unmatched_parameters(direction):
    convert = ConvertWeights(direction)
    ctx = FLContext()
    base = np.array([1.0, 3.0])
    ctx.set_prop(FLContextKey.TASK_DATA, DXO(DataKind.WEIGHTS, {"w": base}).to_shareable(), private=True, sticky=False)
    kind = DataKind.WEIGHTS if direction == ConvertWeights.WEIGHTS_TO_DIFF else DataKind.WEIGHT_DIFF
    result = DXO(kind, {"w": np.array([4.0, 8.0]), "new": np.array([9.0])})
    assert convert.process_dxo(result, Shareable(), ctx) is result
    np.testing.assert_array_equal(result.data["w"], [3, 5] if direction == ConvertWeights.WEIGHTS_TO_DIFF else [5, 11])
    np.testing.assert_array_equal(result.data["new"], [9])
    np.testing.assert_array_equal(base, [1, 3])
    assert result.data_kind == (
        DataKind.WEIGHT_DIFF if direction == ConvertWeights.WEIGHTS_TO_DIFF else DataKind.WEIGHTS
    )


@pytest.mark.parametrize("bad", ["missing", "not-shareable", "no-dxo", "kind", "processed", "empty"])
def test_invalid_or_processed_task_bases_leave_result_unmodified(bad):
    convert = ConvertWeights(ConvertWeights.WEIGHTS_TO_DIFF)
    ctx = FLContext()
    task = DXO(
        DataKind.METRICS if bad == "kind" else DataKind.WEIGHTS,
        {} if bad == "empty" else {"w": 1},
        meta={MetaKey.PROCESSED_ALGORITHM: "encrypted"} if bad == "processed" else {},
    ).to_shareable()
    if bad == "not-shareable":
        task = "invalid"
    elif bad == "no-dxo":
        task = Shareable()
    if bad != "missing":
        ctx.set_prop(FLContextKey.TASK_DATA, task, private=True, sticky=False)
    result = DXO(DataKind.WEIGHTS, {"w": 5})
    assert convert.process_dxo(result, Shareable(), ctx) is None
    assert result.data == {"w": 5} and result.data_kind == DataKind.WEIGHTS


@pytest.mark.parametrize("direction", [ConvertWeights.WEIGHTS_TO_DIFF, ConvertWeights.DIFF_TO_WEIGHTS])
@pytest.mark.parametrize("reason", ["kind", "processed"])
def test_processed_or_wrong_kind_results_are_not_converted(direction, reason):
    convert = ConvertWeights(direction)
    ctx = FLContext()
    ctx.set_prop(FLContextKey.TASK_DATA, DXO(DataKind.WEIGHTS, {"w": 2}).to_shareable(), private=True, sticky=False)
    appropriate = DataKind.WEIGHTS if direction == ConvertWeights.WEIGHTS_TO_DIFF else DataKind.WEIGHT_DIFF
    wrong = DataKind.WEIGHT_DIFF if direction == ConvertWeights.WEIGHTS_TO_DIFF else DataKind.WEIGHTS
    result = DXO(
        wrong if reason == "kind" else appropriate,
        {"w": 5},
        meta={MetaKey.PROCESSED_ALGORITHM: "private"} if reason == "processed" else {},
    )
    assert convert.process_dxo(result, Shareable(), ctx) is None
    assert result.data == {"w": 5}
    with pytest.raises(ValueError, match="invalid convert direction"):
        ConvertWeights("invalid")
