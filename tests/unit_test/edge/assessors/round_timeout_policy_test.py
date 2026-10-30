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

from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.app_common.abstract.model import make_model_learnable
from nvflare.app_common.app_constant import AppConstants
from nvflare.edge.assessor import Assessment
from nvflare.edge.assessors import timeout as subject
from tests.timing_utils import isolate_time


def test_round_timeout_is_strict_and_end_resets_clock_after_model_aggregation(monkeypatch):
    assessor, ctx = subject.TimeoutAssessor("generator", "aggregator", "", 10), FLContext()
    clock = isolate_time(monkeypatch, subject)
    monkeypatch.setattr(clock, "time", Mock(side_effect=[5.0, 15.0, 15.01]))
    for name in ("log_info", "fire_event", "fire_event_with_data"):
        monkeypatch.setattr(assessor, name, Mock())
    assessor.shareable_gen, assessor.aggregator = Mock(), Mock()
    task = Shareable({"weights": [1]})
    assessor.shareable_gen.learnable_to_shareable.return_value = task
    model = make_model_learnable({"weights": [2]}, {})
    assessor.shareable_gen.shareable_to_learnable.return_value = model
    assert assessor.start_task(ctx) is task
    assert assessor.do_assessment(ctx) is Assessment.CONTINUE
    assert assessor.do_assessment(ctx) is Assessment.TASK_DONE
    assessor.end_task(ctx)
    assert assessor._start_time is None
    assessor.aggregator.reset.assert_called_once_with(ctx)
    assert ctx.get_prop(AppConstants.GLOBAL_MODEL) is model
