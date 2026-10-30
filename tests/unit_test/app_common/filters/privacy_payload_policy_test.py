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
from nvflare.apis.filter import ContentBlockedException
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.app_common.filters.dxo_blocker import DXOBlocker
from nvflare.app_common.filters.percentile_privacy import PercentilePrivacy


def test_percentile_privacy_applies_per_step_cutoff_clipping_and_scalar_passthrough():
    privacy = PercentilePrivacy(percentile=50, gamma=0.5)
    data = {"w": np.array([-2.0, -0.2, 0.4, 3.0]), "scalar": np.array(4.0)}
    original = data["w"].copy()
    dxo = DXO(DataKind.WEIGHT_DIFF, data, {MetaKey.NUM_STEPS_CURRENT_ROUND: 2})
    result = privacy.process_dxo(dxo, Shareable(), FLContext())
    assert result is dxo
    np.testing.assert_array_equal(result.data["w"], [-1.0, 0, 0, 1.0])
    assert result.data["scalar"] == 4.0
    np.testing.assert_array_equal(data["w"], original)


@pytest.mark.parametrize("percentile,gamma", [(10, 0), (-1, 1), (101, 1)])
def test_disabled_or_invalid_percentile_policy_preserves_payload(percentile, gamma):
    privacy = PercentilePrivacy(percentile=percentile, gamma=gamma)
    dxo = DXO(DataKind.WEIGHTS, {"w": np.array([1.0])})
    assert privacy.process_dxo(dxo, Shareable(), FLContext()) is None
    np.testing.assert_array_equal(dxo.data["w"], [1.0])


@pytest.mark.parametrize(
    "allow,kind,blocked",
    [
        (True, DataKind.WEIGHTS, False),
        (True, DataKind.METRICS, True),
        (False, DataKind.WEIGHTS, True),
        (False, DataKind.METRICS, False),
    ],
)
def test_dxo_blocking_supports_allow_and_deny_lists(allow, kind, blocked):
    blocker = DXOBlocker([DataKind.WEIGHTS], allow_data_kinds=allow)
    dxo = DXO(kind, {"value": 1})
    if blocked:
        with pytest.raises(ContentBlockedException, match=kind):
            blocker.process_dxo(dxo, Shareable(), FLContext())
    else:
        assert blocker.process_dxo(dxo, Shareable(), FLContext()) is None
        assert dxo.data == {"value": 1}


@pytest.mark.parametrize("kinds", [[], "WEIGHTS", [DataKind.WEIGHTS, 1]])
def test_dxo_blocker_rejects_ambiguous_configuration(kinds):
    with pytest.raises(ValueError, match="data_kinds"):
        DXOBlocker(kinds)
