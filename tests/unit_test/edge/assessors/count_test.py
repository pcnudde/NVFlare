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

import pytest

from nvflare.apis.fl_context import FLContext
from nvflare.edge.assessor import Assessment
from nvflare.edge.assessors import count
from nvflare.edge.assessors.count import CountAssessor
from nvflare.edge.assessors.sgap import SGAPAssessor
from tests.timing_utils import isolate_time


@pytest.mark.parametrize(
    "elapsed,num_updates,expected",
    [
        (0, 0, Assessment.CONTINUE),
        (10, 1, Assessment.CONTINUE),
        (10.1, 0, Assessment.TASK_DONE),
        (0, 2, Assessment.TASK_DONE),
        (0, 4, Assessment.WORKFLOW_DONE),
        (0, 8, Assessment.WORKFLOW_DONE),
    ],
)
def test_count_and_deadline_choose_task_or_workflow_completion(monkeypatch, elapsed, num_updates, expected):
    clock = isolate_time(monkeypatch, count)
    monkeypatch.setattr(clock, "time", lambda: 100 + elapsed)
    assessor = CountAssessor("", "aggr", "", min_count=2, max_count=4, timeout=10)
    assessor._start_time = 100
    assessor.aggregator = Mock()
    assessor.aggregator.get_count.return_value = num_updates

    assert assessor.assess(FLContext()) is expected
    if elapsed > 10:
        assessor.aggregator.get_count.assert_not_called()


def test_task_lifecycle_tracks_start_time_and_clears_it_after_aggregation(monkeypatch):
    clock = isolate_time(monkeypatch, count)
    monkeypatch.setattr(clock, "time", lambda: 123)
    monkeypatch.setattr(SGAPAssessor, "start_task", lambda self, ctx: {"round": 1})
    end = Mock()
    monkeypatch.setattr(SGAPAssessor, "end_task", end)
    assessor = CountAssessor("", "aggr", "", 2, 4, 10)
    ctx = FLContext()

    assert assessor.start_task(ctx) == {"round": 1}
    assert assessor._start_time == 123
    assessor.end_task(ctx)
    assert assessor._start_time is None
    end.assert_called_once_with(ctx)
