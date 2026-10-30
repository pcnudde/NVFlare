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

from nvflare.apis.dxo import DXO, DataKind, from_shareable
from nvflare.apis.fl_constant import ReservedKey, ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.app_common.aggregators.assembler import Assembler
from nvflare.app_common.aggregators.collect_and_assemble_aggregator import CollectAndAssembleAggregator
from nvflare.app_common.aggregators.dxo_collector import DXOCollector
from nvflare.app_common.app_constant import AppConstants


class ConcatenateAssembler(Assembler):
    def __init__(self):
        super().__init__(DataKind.WEIGHTS)

    def get_model_params(self, dxo):
        return list(dxo.data["vectors"])

    def assemble(self, data, fl_ctx):
        return DXO(DataKind.WEIGHTS, {"vectors": [v for values in data.values() for v in values]})


def contribution(name="one", round_number=3, data_kind=DataKind.WEIGHTS):
    result = DXO(data_kind, {"vectors": [1, 2]}).to_shareable()
    result.set_peer_props({ReservedKey.IDENTITY_NAME: name})
    result.add_cookie(AppConstants.CONTRIBUTION_ROUND, round_number)
    return result


@pytest.fixture
def collection():
    assembler = ConcatenateAssembler()
    engine = Mock()
    engine.get_component.return_value = assembler
    ctx = FLContext()
    ctx.set_prop(ReservedKey.ENGINE, engine, private=True, sticky=True)
    ctx.set_prop(AppConstants.CURRENT_ROUND, 3, private=True, sticky=True)
    return CollectAndAssembleAggregator("assembler"), assembler, engine, ctx


def test_collection_accepts_each_site_once_and_resets_between_rounds(collection):
    aggregator, assembler, engine, ctx = collection
    assembler.initialize(ctx)
    assert aggregator.accept(contribution(), ctx)
    assert not aggregator.accept(contribution(), ctx)
    assert aggregator.accept(contribution("two"), ctx)
    engine.get_component.assert_called_once_with("assembler")
    result = from_shareable(aggregator.aggregate(ctx))
    assert result.data_kind == DataKind.WEIGHTS
    assert result.data == {"vectors": [1, 2, 1, 2]}
    assert assembler.collection == {}
    assert aggregator.accept(contribution(), ctx)


@pytest.mark.parametrize("failure", ["invalid-dxo", "return-code", "wrong-kind", "stale-round"])
def test_collection_rejects_unusable_contributions_without_polluting_state(collection, failure):
    aggregator, assembler, _, ctx = collection
    data = contribution()
    if failure == "invalid-dxo":
        data = Shareable()
    elif failure == "return-code":
        data.set_return_code(ReturnCode.EXECUTION_EXCEPTION)
    elif failure == "wrong-kind":
        data = contribution(data_kind=DataKind.METRICS)
    elif failure == "stale-round":
        data = contribution(round_number=2)
    assert not aggregator.accept(data, ctx)
    assert assembler.collection == {}


def test_dxo_collection_preserves_site_payloads_and_requires_peer_identity():
    collector = DXOCollector()
    ctx = FLContext()
    peer = FLContext()
    ctx.set_peer_context(peer)
    assert not collector.accept(Shareable(), ctx)
    assert not collector.accept(contribution(), ctx)
    peer.set_prop(ReservedKey.IDENTITY_NAME, "one", private=False, sticky=True)
    assert collector.accept(contribution(), ctx)
    peer.set_prop(ReservedKey.IDENTITY_NAME, "two", private=False, sticky=True)
    assert collector.accept(contribution("two", data_kind=DataKind.METRICS), ctx)
    result = from_shareable(collector.aggregate(ctx))
    assert result.data_kind == DataKind.COLLECTION
    assert set(result.data) == {"one", "two"}
    assert result.data["one"].data == {"vectors": [1, 2]}
    assert result.data["two"].data_kind == DataKind.METRICS
    collector.reset(ctx)
    assert from_shareable(collector.aggregate(ctx)).data == {}
