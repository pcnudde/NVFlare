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

from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.app_common.abstract.learnable_persistor import LearnablePersistor
from nvflare.app_common.abstract.model import make_model_learnable
from nvflare.app_common.abstract.shareable_generator import ShareableGenerator
from nvflare.app_common.app_constant import AppConstants
from nvflare.app_common.workflows.cyclic_ctl import CyclicController


@pytest.mark.parametrize(
    "options,error",
    [
        ({"num_rounds": "1"}, TypeError),
        ({"task_assignment_timeout": 0.5}, TypeError),
        ({"persistor_id": None}, TypeError),
        ({"shareable_generator_id": None}, TypeError),
        ({"task_name": 3}, TypeError),
        ({"order": "invalid"}, ValueError),
    ],
)
def test_cyclic_configuration_rejects_invalid_round_task_and_order_types(options, error):
    with pytest.raises(error):
        CyclicController(**options)


def test_cyclic_insufficient_client_admission_reports_panic_under_loaded_model(monkeypatch):
    controller, context = CyclicController(), FLContext()
    persistor, generator = Mock(spec=LearnablePersistor), Mock(spec=ShareableGenerator)
    model = make_model_learnable({"w": 1}, {})
    persistor.load.return_value = model
    controller._engine = Mock()
    controller._engine.get_component.side_effect = [persistor, generator]
    controller._engine.get_clients.return_value = [SimpleNamespace(name="only-site")]
    panic = Mock()
    monkeypatch.setattr(controller, "system_panic", panic)
    monkeypatch.setattr(controller, "fire_event", Mock())
    controller.start_controller(context)
    panic.assert_called_once_with("Not enough client sites.", context)
    assert context.get_prop(AppConstants.GLOBAL_MODEL) is model


def test_cyclic_unknown_result_is_dropped_without_modifying_persisted_model(monkeypatch):
    controller, context = CyclicController(), FLContext()
    marker = object()
    controller._last_learnable = marker
    warning = Mock()
    monkeypatch.setattr(controller, "log_warning", warning)
    controller.process_result_of_unknown_task(SimpleNamespace(name="site"), "unknown", "task", Shareable(), context)
    assert controller._last_learnable is marker
    assert "Dropped result" in warning.call_args.args[1]
