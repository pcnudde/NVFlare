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
from nvflare.apis.shareable import Shareable
from nvflare.apis.signal import Signal
from nvflare.app_common.abstract.response_processor import ResponseProcessor
from nvflare.app_common.app_constant import AppConstants
from nvflare.app_common.response_processors.global_weights_initializer import GlobalWeightsInitializer, WeightMethod
from nvflare.app_common.workflows.broadcast_and_process import BroadcastAndProcess


@pytest.mark.parametrize("method,expected", [(WeightMethod.FIRST, 2), (WeightMethod.CLIENT, 5)])
def test_broadcast_initialization_selects_first_or_designated_client_and_publishes_sticky_model(method, expected):
    processor = GlobalWeightsInitializer(weight_method=method, client_name="two")
    controller = BroadcastAndProcess(processor, "weights", clients=["one", "two"], timeout=4, min_responses_required=2)
    controller.system_panic = Mock()
    ctx = FLContext()
    callbacks = []

    def broadcast(**kwargs):
        task = kwargs["task"]
        assert task.name == "weights" and task.timeout == 4
        assert kwargs["targets"] == ["one", "two"] and kwargs["min_responses"] == 2
        for name, value in (("one", 2), ("two", 5)):
            client_task = SimpleNamespace(
                task=task, client=SimpleNamespace(name=name), result=DXO(DataKind.WEIGHTS, {"w": value}).to_shareable()
            )
            task.result_received_cb(client_task, ctx)
            callbacks.append(client_task)

    controller.broadcast_and_wait = Mock(side_effect=broadcast)
    controller.control_flow(Signal(), ctx)
    assert ctx.get_prop(AppConstants.GLOBAL_MODEL)["weights"] == {"w": expected}
    detail = ctx.get_prop_detail(AppConstants.GLOBAL_MODEL)
    assert detail["private"] and detail["sticky"]
    assert all(task.result is None for task in callbacks)
    controller.system_panic.assert_not_called()
    processor.create_task_data("weights", ctx)
    assert processor.final_weights is None and processor.final_process(ctx) is False


@pytest.mark.parametrize(
    "value",
    [
        object(),
        Shareable(),
        DXO(DataKind.METRICS, {"loss": 1}).to_shareable(),
        DXO(DataKind.WEIGHTS, {}).to_shareable(),
    ],
)
def test_initializer_rejects_missing_or_wrong_model_weights_without_selecting_a_provider(value):
    processor = GlobalWeightsInitializer()
    assert not processor.process_client_response(SimpleNamespace(name="one"), "weights", value, FLContext())
    assert processor.final_weights is None


@pytest.mark.parametrize(
    "kwargs",
    [
        {"weight_method": "bad"},
        {"weight_method": WeightMethod.CLIENT},
        {"weight_method": WeightMethod.CLIENT, "client_name": ["one"]},
    ],
)
def test_provider_selection_configuration_requires_supported_method_and_single_client(kwargs):
    with pytest.raises(ValueError):
        GlobalWeightsInitializer(**kwargs)


@pytest.mark.parametrize("mode", ["no-engine", "wrong-processor", "ok"])
def test_controller_resolves_processor_component_before_broadcasting(mode):
    ctx = FLContext()
    engine = Mock()
    ctx.set_prop(ReservedKey.ENGINE, engine if mode != "no-engine" else None)
    processor = GlobalWeightsInitializer()
    engine.get_component.return_value = processor if mode == "ok" else object()
    controller = BroadcastAndProcess("initializer", "weights")
    controller.system_panic = Mock()
    controller.start_controller(ctx)
    assert controller.system_panic.called is (mode != "ok")
    if mode == "ok":
        assert controller.processor is processor
        engine.get_component.assert_called_once_with("initializer")


@pytest.mark.parametrize("failure", ["payload", "response", "final"])
def test_controller_panics_on_processor_failure_and_clears_received_payload(failure):
    processor = Mock(spec=ResponseProcessor)
    processor.create_task_data.return_value = "invalid" if failure == "payload" else Shareable()
    processor.final_process.return_value = failure != "final"
    processor.process_client_response.return_value = failure != "response"
    controller = BroadcastAndProcess(processor, "weights")
    controller.system_panic = Mock()
    controller.broadcast_and_wait = Mock()
    ctx = FLContext()
    if failure == "response":
        task = SimpleNamespace(name="weights")
        received = SimpleNamespace(task=task, result=Shareable({"w": 3}), client=SimpleNamespace(name="one"))
        controller._process_client_response(received, ctx)
        assert received.result is None
    else:
        controller.control_flow(Signal(), ctx)
    controller.system_panic.assert_called_once()
    if failure == "payload":
        controller.broadcast_and_wait.assert_not_called()
