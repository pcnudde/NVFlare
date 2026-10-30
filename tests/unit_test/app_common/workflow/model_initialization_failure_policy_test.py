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

from nvflare.apis.client import Client
from nvflare.apis.fl_context import FLContext
from nvflare.apis.signal import Signal
from nvflare.app_common.abstract.fl_model import FLModel
from nvflare.app_common.abstract.model import make_model_learnable
from nvflare.app_common.app_constant import AppConstants
from nvflare.app_common.app_event_type import AppEventType
from nvflare.app_common.workflows.base_model_controller import BaseModelController


class ModelWorkflow(BaseModelController):
    def run(self):
        self.action()


@pytest.fixture
def runtime(monkeypatch):
    controller = ModelWorkflow()
    ctx = FLContext()
    controller.fl_ctx = ctx
    controller.abort_signal = Signal()
    controller.engine = Mock()
    controller.engine.get_clients.return_value = [Client("one", "1"), Client("two", "2")]
    controller.action = Mock()
    for name in ("info", "warning", "exception", "panic", "event", "error", "broadcast", "broadcast_and_wait"):
        monkeypatch.setattr(controller, name, Mock())
    return controller, ctx


@pytest.mark.parametrize("failure", [False, True])
def test_control_flow_publishes_training_start_and_turns_run_failure_into_panic(runtime, failure):
    controller, ctx = runtime
    if failure:
        controller.action.side_effect = ValueError("training failed")
    abort = Signal()
    controller.control_flow(abort, ctx)
    assert controller.fl_ctx is ctx and controller.abort_signal is abort
    controller.action.assert_called_once_with()
    controller.event.assert_called_once_with(AppEventType.TRAINING_STARTED)
    assert controller.panic.call_count == controller.exception.call_count == int(failure)
    if failure:
        assert "training failed" in controller.panic.call_args.args[0]


@pytest.mark.parametrize("source", ["persistor", "context", "empty-allowed", "missing", "invalid", "none-configured"])
def test_model_initialization_validates_persistor_data_and_resolves_context_fallback(runtime, source):
    controller, ctx = runtime
    controller.persistor = Mock()
    original = make_model_learnable({"w": [1.0]}, {"source": "persistor"})
    controller.persistor.load.return_value = original
    if source == "invalid":
        controller.persistor.load.return_value = {"not": "learnable"}
    elif source in ("context", "empty-allowed", "missing"):
        controller.persistor.load.return_value = make_model_learnable({}, {})
        controller._allow_empty_global_weights = source == "empty-allowed"
        ctx.set_prop(AppConstants.GLOBAL_MODEL, make_model_learnable({"w": [2.0]} if source == "context" else {}, {}))
    elif source == "none-configured":
        controller.persistor = None
    model = controller.load_model()
    if source in ("invalid", "missing"):
        assert model is None
        controller.panic.assert_called_once()
        controller.event.assert_not_called()
    else:
        expected = {"w": [1.0]} if source == "persistor" else {"w": [2.0]} if source == "context" else {}
        assert model.params == expected
        assert ctx.get_prop(AppConstants.GLOBAL_MODEL)["weights"] == expected
        controller.event.assert_called_once_with(AppEventType.INITIAL_MODEL_LOADED)
        controller.panic.assert_not_called()


@pytest.mark.parametrize("blocking", [False, True])
def test_model_broadcast_normalizes_client_objects_and_preserves_task_error_budget(runtime, blocking):
    controller, ctx = runtime
    model = FLModel(params={"w": [1]}, current_round=3)
    callback = Mock()
    clients = [Client("one", "1"), "two"]
    result = controller.broadcast_model(model, targets=clients, min_responses=1, blocking=blocking, callback=callback)
    send = controller.broadcast_and_wait if blocking else controller.broadcast
    send.assert_called_once()
    call = send.call_args.kwargs
    assert call["targets"] == ["one", "two"] and call["min_responses"] == 1
    assert call["fl_ctx"] is ctx and call["task"].data.get_header(AppConstants.CURRENT_ROUND) == 3
    assert controller._current_min_responses == 1 and controller._current_num_targets == 2
    assert controller._current_failed_clients == set()
    assert result == ([] if blocking else None)
    if blocking:
        assert "expected responses" in controller.warning.call_args.args[0]


@pytest.mark.parametrize("kwargs", [{"task_name": 2}, {"data": {}}, {"blocking": False}, {"timeout": -1}])
def test_broadcast_rejects_invalid_contract_before_transport(runtime, kwargs):
    controller, _ = runtime
    arguments = {"data": FLModel(params={}), **kwargs}
    with pytest.raises((TypeError, ValueError)):
        controller.broadcast_model(**arguments)
    controller.broadcast.assert_not_called()
    controller.broadcast_and_wait.assert_not_called()


def test_model_save_preserves_weights_metadata_and_announces_persistence_boundaries(runtime):
    controller, ctx = runtime
    controller.persistor = Mock()
    model = FLModel(params={"w": [3]}, meta={"round": 4})
    controller.save_model(model)
    learnable, context = controller.persistor.save.call_args.args
    assert context is ctx and learnable == make_model_learnable({"w": [3]}, {"round": 4})
    assert [c.args[0] for c in controller.event.call_args_list] == [
        AppEventType.BEFORE_LEARNABLE_PERSIST,
        AppEventType.AFTER_LEARNABLE_PERSIST,
    ]
    controller.persistor = None
    controller.save_model(model)
    controller.error.assert_called_once_with("persistor not configured, model will not be saved")
