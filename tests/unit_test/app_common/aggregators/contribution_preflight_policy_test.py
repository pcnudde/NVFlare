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
from nvflare.apis.fl_context import FLContext
from nvflare.app_common.aggregators.dxo_aggregator import DXOAggregator
from nvflare.app_common.app_constant import AppConstants


@pytest.mark.parametrize(
    "failure",
    ["not-dxo", "unsupported", "wrong-kind", "algorithm", "wrong-round", "missing-data", "duplicate", "shape"],
)
def test_rejected_contribution_never_changes_accepted_weights_history_or_algorithm(failure):
    aggregator = DXOAggregator(expected_data_kind=DataKind.WEIGHTS)
    ctx = FLContext()
    ctx.set_prop(AppConstants.CURRENT_ROUND, 3)
    first = DXO(
        DataKind.WEIGHTS,
        {"w": np.array([1.0, 3.0])},
        {MetaKey.PROCESSED_ALGORITHM: "first", MetaKey.NUM_STEPS_CURRENT_ROUND: 1},
    )
    assert aggregator.accept(first, "accepted", 3, ctx)
    history = list(aggregator.aggregation_helper.get_history())
    dxo = DXO(DataKind.WEIGHTS, {"w": np.array([5.0, 7.0])})
    if failure == "not-dxo":
        dxo = {}
    elif failure == "unsupported":
        dxo = DXO(DataKind.COLLECTION, {})
    elif failure == "wrong-kind":
        dxo = DXO(DataKind.WEIGHT_DIFF, {"w": np.array([1.0])})
    elif failure == "algorithm":
        dxo.set_meta_prop(MetaKey.PROCESSED_ALGORITHM, "second")
    elif failure == "missing-data":
        dxo.data = None
    elif failure == "shape":
        dxo.data = {"w": np.array([1.0])}
    assert (
        aggregator.can_accept(
            dxo, "accepted" if failure == "duplicate" else "rejected", 2 if failure == "wrong-round" else 3, ctx
        )
        is False
    )
    assert aggregator.aggregation_helper.get_history() == history
    assert aggregator.processed_algorithm == "first"
    np.testing.assert_array_equal(aggregator.aggregate(ctx).data["w"], [1.0, 3.0])
