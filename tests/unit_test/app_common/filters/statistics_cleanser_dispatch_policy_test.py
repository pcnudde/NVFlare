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

from unittest.mock import Mock

import pytest

from nvflare.apis.dxo import DXO, DataKind
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.app_common.app_constant import StatisticsConstants as StC
from nvflare.app_common.filters.statistics_privacy_filter import StatisticsPrivacyFilter
from nvflare.app_common.statistics.statistics_privacy_cleanser import StatisticsPrivacyCleanser
from nvflare.fuel.utils import fobs


@pytest.mark.parametrize("component", ["disabled", "invalid", "valid"])
def test_cleanser_lookup_skips_disabled_and_rejects_components_outside_privacy_protocol(component, monkeypatch):
    engine = Mock()
    cleanser = Mock(spec=StatisticsPrivacyCleanser)
    engine.get_component.return_value = (
        None if component == "disabled" else object() if component == "invalid" else cleanser
    )
    ctx = FLContext()
    monkeypatch.setattr(ctx, "get_engine", lambda: engine)
    privacy = StatisticsPrivacyFilter(["cleanser"])
    if component == "invalid":
        with pytest.raises(ValueError, match="StatisticsPrivacyCleanser"):
            privacy.get_cleansers(["cleanser"], ctx)
    else:
        assert privacy.get_cleansers(["cleanser"], ctx) == ([cleanser] if component == "valid" else [])
    engine.get_component.assert_called_once_with("cleanser")


def test_statistics_dispatch_cleanses_in_order_and_preserves_withheld_feature_report(monkeypatch):
    privacy = StatisticsPrivacyFilter(["cleanser"])
    cleanser = Mock(spec=StatisticsPrivacyCleanser)
    original = {"dataset": {"count": 1}}
    cleaned = {"dataset": {}}
    cleanser.apply.return_value = cleaned, True
    cleanser.withheld_features.return_value = {"dataset": ["small"]}
    engine = Mock()
    engine.get_component.return_value = cleanser
    ctx = FLContext()
    monkeypatch.setattr(ctx, "get_engine", lambda: engine)
    task = "summary"
    dxo = DXO(DataKind.STATISTICS, {StC.STATISTICS_TASK_KEY: task, task: fobs.dumps(original)})
    result = privacy.process_dxo(dxo, Shareable(), ctx)
    assert fobs.loads(result.data[task]) == cleaned
    assert fobs.loads(result.data[StC.STATS_WITHHELD_FEATURES]) == {"dataset": ["small"]}
    cleanser.apply.assert_called_once_with(original, ctx.get_identity_name())
    assert privacy.process_dxo(DXO(DataKind.WEIGHTS, {}), Shareable(), ctx) is None
