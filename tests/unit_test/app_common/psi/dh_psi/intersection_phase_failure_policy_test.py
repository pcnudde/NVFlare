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
from nvflare.apis.signal import Signal
from nvflare.app_common.app_constant import PSIConst
from nvflare.app_common.psi.dh_psi.dh_psi_workflow import DhPSIWorkFlow, SiteSize


@pytest.fixture
def workflow(monkeypatch):
    workflow = DhPSIWorkFlow()
    workflow.initialize(FLContext(), controller=Mock())
    workflow.abort_signal = Signal()
    for name in ("forward_pass", "backward_pass", "parallel_backward_pass"):
        function = getattr(DhPSIWorkFlow, name)
        monkeypatch.setattr(function, "time_taken", getattr(function, "time_taken", None), raising=False)
    return workflow


@pytest.mark.parametrize("phase", ["before", "forward", "backward", "complete"])
def test_intersection_run_checks_abort_before_using_next_phase_results(workflow, monkeypatch, phase):
    sites = [SiteSize("one", 2), SiteSize("two", 3)]
    workflow.ordered_sites = sites

    def forward(ordered, processed):
        processed.update(one=2)
        if phase == "forward":
            workflow.abort_signal.trigger(True)
        return sites[0]

    def backward(ordered, intersection):
        if phase == "backward":
            workflow.abort_signal.trigger(True)
        return {"two": 2}

    first, second, timing = Mock(side_effect=forward), Mock(side_effect=backward), Mock()
    monkeypatch.setattr(workflow, "forward_pass", first)
    monkeypatch.setattr(workflow, "backward_pass", second)
    monkeypatch.setattr(workflow, "log_pass_time_taken", timing)
    if phase == "before":
        workflow.abort_signal.trigger(True)
    result = workflow.run(workflow.abort_signal)
    assert first.call_count == int(phase != "before")
    assert second.call_count == int(phase in ("backward", "complete"))
    assert timing.call_count == int(phase == "complete")
    assert workflow.backward_processed == ({"two": 2} if phase == "complete" else {})
    assert result is (None if phase == "complete" else False)


@pytest.mark.parametrize(
    "phase", ["before", "pairwise_setup", "pairwise_requests", "pairwise_responses", "pairwise_intersect"]
)
def test_forward_reduction_never_runs_next_protocol_step_after_cancel(workflow, monkeypatch, phase):
    names = ["pairwise_setup", "pairwise_requests", "pairwise_responses", "pairwise_intersect"]
    boundaries = {}
    for name in names:

        def boundary(*args, current=name):
            if phase == current:
                workflow.abort_signal.trigger(True)
            return {"one": "encrypted"}

        boundaries[name] = Mock(side_effect=boundary)
        monkeypatch.setattr(workflow, name, boundaries[name])
    if phase == "before":
        workflow.abort_signal.trigger(True)
    processed = {}
    assert workflow.parallel_forward_pass([SiteSize("one", 2), SiteSize("two", 3)], processed) is None
    called = [] if phase == "before" else names[: names.index(phase) + 1]
    assert [name for name in names if boundaries[name].called] == called
    assert processed == {}


@pytest.mark.parametrize("phase", ["before", "single", "parallel-abort", "complete"])
def test_backward_pass_returns_only_complete_protocol_results(workflow, monkeypatch, phase):
    sites = [SiteSize("one", 2), SiteSize("two", 3)]

    def parallel(*args):
        if phase == "parallel-abort":
            workflow.abort_signal.trigger(True)
        return {"two": 2}

    boundary = Mock(side_effect=parallel)
    boundary.time_taken = 5
    monkeypatch.setattr(workflow, "parallel_backward_pass", boundary)
    if phase == "before":
        workflow.abort_signal.trigger(True)
    if phase == "single":
        sites = sites[:1]
    result = workflow.backward_pass(sites, sites[0])
    assert result == ({"two": 2} if phase == "complete" else {})
    assert boundary.call_count == int(phase in ("parallel-abort", "complete"))


def test_intersection_counts_must_agree_and_empty_sites_are_excluded(workflow):
    with pytest.raises(RuntimeError, match="smaller"):
        workflow.check_processed_sites(SiteSize("one", 3), {"two": 2})
    workflow.backward_processed = {"two": 4}
    with pytest.raises(RuntimeError, match="equal"):
        workflow.check_final_intersection_sizes(SiteSize("one", 3))
    assert workflow.get_ordered_sites(
        {
            "empty": DXO(DataKind.METRICS, {}),
            "large": DXO(DataKind.METRICS, {PSIConst.ITEMS_SIZE: 5}),
            "small": DXO(DataKind.METRICS, {PSIConst.ITEMS_SIZE: 2}),
        }
    ) == [SiteSize("small", 2), SiteSize("large", 5)]


def test_preparation_abort_never_contacts_sites(workflow, monkeypatch):
    prepare = Mock()
    monkeypatch.setattr(workflow, "prepare_sites", prepare)
    workflow.abort_signal.trigger(True)
    assert workflow.pre_process(workflow.abort_signal) is False
    prepare.assert_not_called()
