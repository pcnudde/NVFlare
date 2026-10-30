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
from nvflare.apis.event_type import EventType
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.app_common.aggregators.intime_accumulate_model_aggregator import InTimeAccumulateWeightedAggregator


@pytest.mark.parametrize("problem", ["malformed", "unsupported", "return-code", "no-expected-members"])
def test_rejected_collection_cannot_reach_any_member_accumulator(monkeypatch, problem):
    context = FLContext()
    aggregator = InTimeAccumulateWeightedAggregator(expected_data_kind={"weights": DataKind.WEIGHTS})
    aggregator.handle_event(EventType.START_RUN, context)
    member = aggregator.dxo_aggregators["weights"]
    accept = Mock()
    monkeypatch.setattr(member, "accept", accept)
    if problem == "malformed":
        payload = Shareable()
    elif problem == "unsupported":
        payload = DXO(DataKind.PSI, {}).to_shareable()
    elif problem == "return-code":
        payload = DXO(DataKind.COLLECTION, {"weights": DXO(DataKind.WEIGHTS, {"w": 1})}).to_shareable()
        payload.set_return_code("ERROR")
    else:
        payload = DXO(DataKind.COLLECTION, {"weights": "not-a-dxo"}).to_shareable()
    assert aggregator.accept(payload, context) is False
    accept.assert_not_called()
    assert member.aggregation_helper.get_history() == []
