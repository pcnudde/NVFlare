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

import copy
from unittest.mock import Mock

import numpy as np
import pytest

from nvflare.apis.dxo import DXO, DataKind, from_shareable
from nvflare.apis.fl_context import FLContext
from nvflare.apis.signal import Signal
from nvflare.app_common.abstract.aggregator import Aggregator
from nvflare.app_common.abstract.learnable_persistor import LearnablePersistor
from nvflare.app_common.abstract.model import make_model_learnable
from nvflare.app_common.abstract.shareable_generator import ShareableGenerator
from nvflare.app_common.app_constant import AlgorithmConstants, AppConstants
from nvflare.app_common.app_event_type import AppEventType
from nvflare.app_common.workflows import scatter_and_gather as sag
from nvflare.app_common.workflows import scatter_and_gather_scaffold as subject


@pytest.fixture
def runtime(monkeypatch):
    controller = subject.ScatterAndGatherScaffold(min_clients=1, num_rounds=1)
    aggregator, persistor, generator = (
        Mock(spec=Aggregator),
        Mock(spec=LearnablePersistor),
        Mock(spec=ShareableGenerator),
    )
    persistor.load.return_value = make_model_learnable({"w": np.array([2.0, 4.0])}, {})
    generator.shareable_to_learnable.return_value = make_model_learnable({"w": np.array([8.0, 10.0])}, {})
    engine = Mock()
    engine.get_component.side_effect = {
        "aggregator": aggregator,
        "persistor": persistor,
        "shareable_generator": generator,
    }.get
    controller._engine = engine
    controller.fire_event = Mock()
    controller.fire_event_with_data = Mock()
    controller.system_panic = Mock()
    controller.broadcast_and_wait = Mock()
    monkeypatch.setattr(subject, "gc", Mock())
    ctx = FLContext()
    controller.start_controller(ctx)
    return controller, aggregator, persistor, generator, engine, ctx


def assembled_result(include_weights=True, include_controls=True):
    parts = {}
    if include_weights:
        parts[AppConstants.MODEL_WEIGHTS] = DXO(DataKind.WEIGHTS, {"w": np.array([8.0, 10.0])})
    if include_controls:
        parts[AlgorithmConstants.SCAFFOLD_CTRL_DIFF] = DXO(DataKind.WEIGHT_DIFF, {"w": np.array([0.25, -0.5])})
    return DXO(DataKind.COLLECTION, parts).to_shareable()


def test_scaffold_round_exchanges_model_and_controls_and_persists_after_update(runtime):
    controller, aggregator, persistor, generator, engine, ctx = runtime
    np.testing.assert_array_equal(controller._global_ctrl_weights["w"], [0, 0])
    aggregator.aggregate.return_value = assembled_result()
    sent = []
    controller.broadcast_and_wait.side_effect = lambda **kwargs: sent.append(copy.deepcopy(kwargs["task"].data))
    controller.control_flow(Signal(), ctx)
    task = controller.broadcast_and_wait.call_args.kwargs["task"]
    incoming = from_shareable(sent[0])
    assert incoming.data_kind == DataKind.COLLECTION
    np.testing.assert_array_equal(incoming.data[AppConstants.MODEL_WEIGHTS].data["w"], [2, 4])
    np.testing.assert_array_equal(incoming.data[AlgorithmConstants.SCAFFOLD_CTRL_GLOBAL].data["w"], [0, 0])
    assert task.data.get_cookie(AppConstants.CONTRIBUTION_ROUND) == 0
    np.testing.assert_array_equal(controller._global_ctrl_weights["w"], [0.25, -0.5])
    np.testing.assert_array_equal(ctx.get_prop(AlgorithmConstants.SCAFFOLD_CTRL_GLOBAL)["w"], [0.25, -0.5])
    assert controller._global_weights is generator.shareable_to_learnable.return_value
    persistor.save.assert_called_once_with(controller._global_weights, ctx)
    engine.persist_components.assert_called_once_with(ctx, completed=False)
    aggregator.reset.assert_called_once_with(ctx)
    assert controller._current_round == 1
    assert controller._phase == AppConstants.PHASE_FINISHED
    assert AppEventType.ROUND_DONE in [c.args[0] for c in controller.fire_event.call_args_list]
    controller.system_panic.assert_not_called()


@pytest.mark.parametrize("failure", ["missing-weights", "missing-controls", "aggregate-error", "aborted"])
def test_scaffold_invalid_or_aborted_round_never_persists_partial_state(runtime, failure):
    controller, aggregator, persistor, generator, _, ctx = runtime
    signal = Signal()
    aggregator.aggregate.return_value = assembled_result(failure != "missing-weights", failure != "missing-controls")
    if failure == "aggregate-error":
        aggregator.aggregate.side_effect = RuntimeError("aggregate")
    elif failure == "aborted":
        signal.trigger(True)
    controller.control_flow(signal, ctx)
    persistor.save.assert_not_called()
    generator.shareable_to_learnable.assert_not_called()
    assert controller.system_panic.called is (failure == "aggregate-error")
    assert controller._current_round == 0


@pytest.mark.parametrize("component", ["aggregator", "shareable_generator", "persistor", "loaded-model"])
def test_sag_start_rejects_wrong_component_contracts_and_invalid_model(component):
    controller = sag.ScatterAndGather(persistor_id="persistor")
    components = {
        "aggregator": Mock(spec=Aggregator),
        "shareable_generator": Mock(spec=ShareableGenerator),
        "persistor": Mock(spec=LearnablePersistor),
    }
    components["persistor"].load.return_value = {}
    if component != "loaded-model":
        components[component] = object()
    controller._engine = Mock()
    controller._engine.get_component.side_effect = components.get
    controller.system_panic = Mock()
    controller.start_controller(FLContext())
    controller.system_panic.assert_called_once()


def test_sag_empty_persisted_weights_fall_back_to_context_global_model():
    controller = sag.ScatterAndGather(persistor_id="persistor")
    components = {
        "aggregator": Mock(spec=Aggregator),
        "shareable_generator": Mock(spec=ShareableGenerator),
        "persistor": Mock(spec=LearnablePersistor),
    }
    components["persistor"].load.return_value = make_model_learnable({}, {})
    controller._engine = Mock()
    controller._engine.get_component.side_effect = components.get
    controller.fire_event = Mock()
    controller.system_panic = Mock()
    ctx = FLContext()
    model = make_model_learnable({"w": [3]}, {})
    ctx.set_prop(AppConstants.GLOBAL_MODEL, model, private=True, sticky=True)
    controller.start_controller(ctx)
    assert controller._global_weights is model
    controller.fire_event.assert_called_once_with(AppEventType.INITIAL_MODEL_LOADED, ctx)
    controller.system_panic.assert_not_called()
