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

from nvflare.apis.event_type import EventType
from nvflare.apis.fl_context import FLContext
from nvflare.app_common.abstract.learnable_persistor import LearnablePersistor
from nvflare.app_common.abstract.model import make_model_learnable
from nvflare.app_common.app_event_type import AppEventType
from nvflare.edge.assessors import model_update as subject
from tests.timing_utils import isolate_time


@pytest.fixture
def runtime(monkeypatch):
    assessor = subject.ModelUpdateAssessor("persistor", "model", "devices", 5, device_wait_timeout=4)
    ctx, engine = FLContext(), Mock()
    monkeypatch.setattr(ctx, "get_engine", lambda: engine)
    monkeypatch.setattr(assessor, "system_panic", Mock())
    monkeypatch.setattr(assessor, "fire_event", Mock())
    monkeypatch.setattr(assessor, "log_error", Mock())
    monkeypatch.setattr(assessor, "log_info", Mock())
    return assessor, ctx, engine


@pytest.mark.parametrize("fault", [None, "persistor", "model", "devices", "bad-model"])
def test_start_validates_components_before_initializing_model_and_emitting_ready(runtime, fault):
    assessor, ctx, engine = runtime
    persistor, model, devices = Mock(spec=LearnablePersistor), Mock(), Mock()
    components = {"persistor": persistor, "model": model, "devices": devices}
    if fault in components:
        components[fault] = object() if fault == "persistor" else None
    persistor.load.return_value = object() if fault == "bad-model" else make_model_learnable({"w": [1, 2]}, {})
    engine.get_component.side_effect = components.get
    assessor._handle_start_run(EventType.START_RUN, ctx)
    if fault:
        assessor.system_panic.assert_called_once()
        model.initialize_model.assert_not_called()
        assessor.fire_event.assert_not_called()
    else:
        payload, context = model.initialize_model.call_args.args
        assert payload.data == {"w": [1, 2]} and context is ctx
        assessor.fire_event.assert_called_once_with(AppEventType.INITIAL_MODEL_LOADED, ctx)
        assessor.system_panic.assert_not_called()


def test_device_timeout_inspection_failure_is_contained_and_status_reports_wait_state(runtime, monkeypatch):
    assessor, ctx, _ = runtime
    clock = isolate_time(monkeypatch, subject)
    monkeypatch.setattr(clock, "time", lambda: 10)
    assessor.device_wait_start_time = 1
    assessor.device_manager = Mock()
    assessor.device_manager.get_usable_devices.side_effect = RuntimeError("inspection")
    assert assessor._is_device_wait_timeout_exceeded(ctx) is False
    assert "inspection" in assessor.log_error.call_args.args[1]
    assessor.device_manager.get_usable_devices.side_effect = None
    assessor.device_manager.get_usable_devices.return_value = []
    assessor.device_manager.available_devices = {"one": 1}
    assessor.device_manager.device_selection_size = 2
    assessor.device_status_log_interval = 1
    assessor._last_device_status_log_time = 0
    assessor.device_wait_timeout = None
    assessor._log_device_status(ctx)
    assert "waiting indefinitely" in assessor.log_info.call_args.args[1]
