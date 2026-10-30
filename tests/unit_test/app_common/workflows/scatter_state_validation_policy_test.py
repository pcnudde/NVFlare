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

from nvflare.apis.fl_context import FLContext
from nvflare.app_common.app_constant import AppConstants
from nvflare.app_common.workflows import scatter_and_gather as subject
from nvflare.widgets.info_collector import InfoCollector


@pytest.mark.parametrize(
    "options,error",
    [
        ({"min_clients": "1"}, TypeError),
        ({"min_clients": 0}, ValueError),
        ({"aggregator_id": None}, TypeError),
        ({"persistor_id": 1}, TypeError),
        ({"shareable_generator_id": None}, TypeError),
        ({"train_task_name": 2}, TypeError),
        ({"task_check_period": "fast"}, TypeError),
        ({"task_check_period": 0}, ValueError),
    ],
)
def test_scatter_configuration_rejects_invalid_types_and_nonpositive_budgets(options, error):
    with pytest.raises(error):
        subject.ScatterAndGather(**options)


def test_scatter_persisted_state_round_trip_restores_round_and_weights():
    source, restored, context = subject.ScatterAndGather(), subject.ScatterAndGather(), FLContext()
    source._current_round, source._start_round, source._num_rounds = 3, 1, 7
    source._global_weights = {"weights": {"w": 5}}
    checkpoint = source.get_persist_state(context)
    restored.restore(checkpoint, context)
    assert restored.get_persist_state(context) == checkpoint
    assert restored._global_weights is checkpoint["global_weights"]


def test_scatter_memory_cleanup_runs_only_at_configured_round_boundary(monkeypatch):
    controller = subject.ScatterAndGather(memory_gc_rounds=2)
    cleanup = Mock()
    monkeypatch.setattr(subject, "cleanup_memory", cleanup)
    controller._maybe_cleanup_memory()
    controller._current_round = 0
    controller._maybe_cleanup_memory()
    cleanup.assert_not_called()
    controller._current_round = 1
    controller._maybe_cleanup_memory()
    cleanup.assert_called_once_with()


def test_scatter_statistics_collector_is_type_checked_and_end_run_marks_finished(monkeypatch):
    controller, context = subject.ScatterAndGather(), FLContext()
    controller.stop_controller(context)
    assert controller._phase == AppConstants.PHASE_FINISHED
    context.set_prop(InfoCollector.CTX_KEY_STATS_COLLECTOR, object())
    with pytest.raises(TypeError, match="GroupInfoCollector"):
        controller.handle_event(InfoCollector.EVENT_TYPE_GET_STATS, context)
