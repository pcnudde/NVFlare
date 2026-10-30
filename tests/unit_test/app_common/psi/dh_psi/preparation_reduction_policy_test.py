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

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from nvflare.apis.dxo import DXO, DataKind
from nvflare.apis.fl_constant import ReservedKey
from nvflare.apis.fl_context import FLContext
from nvflare.apis.signal import Signal
from nvflare.app_common.app_constant import PSIConst
from nvflare.app_common.psi.dh_psi.dh_psi_workflow import DhPSIWorkFlow, SiteSize


@pytest.fixture
def workflow(monkeypatch):
    result = DhPSIWorkFlow()
    result.fl_ctx, result.abort_signal = FLContext(), Signal()
    result.fl_ctx.set_prop(ReservedKey.ENGINE, SimpleNamespace(get_clients=lambda: ["one", "two"]))
    function = DhPSIWorkFlow.forward_pass
    monkeypatch.setattr(function, "time_taken", getattr(function, "time_taken", None), raising=False)
    return result


@pytest.mark.parametrize("mode", ["empty", "negative", "malformed", "abort"])
def test_preparation_refuses_empty_or_malformed_counts_and_respects_transport_abort(workflow, monkeypatch, mode):
    if mode == "empty":
        values = {}
    else:
        values = {
            site: DXO(DataKind.METRICS, {PSIConst.ITEMS_SIZE: -1 if mode == "negative" else 2})
            for site in ("one", "two")
        }
        if mode == "malformed":
            values = []

    def broadcast(**kwargs):
        if mode == "abort":
            workflow.abort_signal.trigger(True)
        return values

    operator = SimpleNamespace(broadcast_and_wait=Mock(side_effect=broadcast))
    monkeypatch.setattr(workflow, "_new_broadcast_operator", lambda: operator)
    if mode == "abort":
        assert workflow.prepare_sites(workflow.abort_signal) is None
    else:
        with pytest.raises(RuntimeError, match="no item|malformed"):
            workflow.prepare_sites(workflow.abort_signal)
        if mode == "empty":
            assert workflow.abort_signal.triggered
    assert workflow.ordered_sites == []
    operator.broadcast_and_wait.assert_called_once()


@pytest.mark.parametrize("sites", [[SiteSize("one", 2)], [SiteSize("one", 2), SiteSize("two", 3)]])
def test_forward_pass_uses_singleton_directly_and_delegates_only_multi_site_reduction(workflow, monkeypatch, sites):
    reduce = Mock(return_value=sites[0])
    monkeypatch.setattr(workflow, "parallel_forward_pass", reduce)
    processed = {}
    assert workflow.forward_pass(sites, processed) == sites[0]
    assert reduce.call_count == int(len(sites) > 1)
    workflow.abort_signal.trigger(True)
    assert workflow.forward_pass(sites, processed) is None
    assert reduce.call_count == int(len(sites) > 1)


def test_pairwise_and_final_intersection_decode_each_expected_response(workflow, monkeypatch):
    values = {"two": DXO(DataKind.METRICS, {PSIConst.ITEMS_SIZE: 2})}
    operator = SimpleNamespace(multicasts_and_wait=Mock(return_value=values))
    monkeypatch.setattr(workflow, "_new_broadcast_operator", lambda: operator)
    assert workflow.pairwise_intersect([SiteSize("one", 2), SiteSize("two", 3)], {"one": "response"}) == {"two": 2}
    assert workflow.calculate_intersections({"two": "response"}) == {"two": 2}
    assert operator.multicasts_and_wait.call_count == 2
    assert workflow._required_values({"one": "message", "extra": "ignored"}, ["one"], "phase") == {"one": "message"}
    with pytest.raises(RuntimeError, match="malformed"):
        workflow._required_values([], ["one"], "phase")
