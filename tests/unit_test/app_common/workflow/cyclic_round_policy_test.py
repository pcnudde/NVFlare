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
from nvflare.apis.controller_spec import TaskCompletionStatus
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.apis.signal import Signal
from nvflare.app_common.abstract.fl_model import FLModel
from nvflare.app_common.abstract.learnable_persistor import LearnablePersistor
from nvflare.app_common.abstract.model import make_model_learnable
from nvflare.app_common.abstract.shareable_generator import ShareableGenerator
from nvflare.app_common.app_constant import AppConstants
from nvflare.app_common.app_event_type import AppEventType
from nvflare.app_common.workflows import cyclic as modern
from nvflare.app_common.workflows import cyclic_ctl as subject


@pytest.fixture
def runtime(monkeypatch):
    controller = subject.CyclicController(num_rounds=2, memory_gc_rounds=1)
    persistor, generator = Mock(spec=LearnablePersistor), Mock(spec=ShareableGenerator)
    persistor.load.return_value = make_model_learnable({"w": [1]}, {})
    generator.learnable_to_shareable.side_effect = lambda *_: Shareable()
    engine = Mock()
    engine.get_component.side_effect = {"persistor": persistor, "shareable_generator": generator}.get
    engine.get_clients.return_value = [Client("one", "1"), Client("two", "2")]
    controller._engine = engine
    controller.fire_event = Mock()
    controller.system_panic = Mock()
    controller.get_client_disconnect_time = Mock(return_value=None)
    monkeypatch.setattr(subject, "cleanup_memory", Mock())
    ctx = FLContext()
    controller.start_controller(ctx)
    return controller, persistor, generator, engine, ctx


def test_cyclic_completed_relays_publish_round_done_persist_snapshot_and_restore_state(runtime):
    controller, persistor, _, engine, ctx = runtime
    sent = []

    def relay(**kwargs):
        task = kwargs["task"]
        sent.append((task.data.get_header(AppConstants.CURRENT_ROUND), [c.name for c in kwargs["targets"]]))
        task.completion_status = TaskCompletionStatus.OK

    controller.relay_and_wait = Mock(side_effect=relay)
    controller.control_flow(Signal(), ctx)
    assert sent == [(0, ["one", "two"]), (1, ["one", "two"])]
    assert persistor.save.call_count == 2
    assert engine.persist_components.call_count == 2
    assert subject.cleanup_memory.call_count == 2
    assert [c.args[0] for c in controller.fire_event.call_args_list].count(AppEventType.ROUND_DONE) == 2
    state = controller.get_persist_state(ctx)
    restored = subject.CyclicController()
    restored.restore(state, ctx)
    assert restored._start_round == restored._current_round == 1
    assert restored._end_round == 2
    assert restored._last_learnable is state["last_learnable"]
    controller.stop_controller(ctx)
    persistor.save.assert_called_with(learnable=state["last_learnable"], fl_ctx=ctx)


@pytest.mark.parametrize("order", [["two", "one"], subject.RelayOrder.RANDOM_WITHOUT_SAME_IN_A_ROW])
def test_cyclic_order_respects_explicit_sites_and_avoids_repeating_boundary_client(runtime, monkeypatch, order):
    controller, _, _, _, ctx = runtime
    controller._order = order
    controller._last_client = controller._participating_clients[0]
    monkeypatch.setattr(subject.random, "shuffle", Mock())
    assert [c.name for c in controller._get_relay_orders(ctx)] == ["two", "one"]
    assert controller._last_client.name == "one"


@pytest.mark.parametrize("failure", ["disconnected", "missing-required", "relay-error", "aborted", "done"])
def test_cyclic_incomplete_round_never_reports_success(runtime, failure):
    controller, persistor, _, _, ctx = runtime
    controller.relay_and_wait = Mock(side_effect=RuntimeError("relay") if failure == "relay-error" else None)
    signal = Signal()
    if failure == "disconnected":
        controller.get_client_disconnect_time.return_value = 1
    elif failure == "missing-required":
        controller._order = ["absent"]
    elif failure == "aborted":
        signal.trigger(True)
    elif failure == "done":
        controller._is_done = True
    controller.control_flow(signal, ctx)
    assert AppEventType.ROUND_DONE not in [
        c.args[0] if c.args else c.kwargs.get("event_type") for c in controller.fire_event.call_args_list
    ]
    persistor.save.assert_not_called()
    assert controller.system_panic.called is (failure in ["disconnected", "missing-required", "relay-error"])


def test_modern_cyclic_passes_each_clients_updated_model_to_the_next_and_saves_each_round(monkeypatch):
    controller = modern.Cyclic(num_clients=2, num_rounds=2, memory_gc_rounds=1)
    controller.load_model = Mock(return_value=FLModel(params={"w": 0}))
    controller.sample_clients = Mock(return_value=["one", "two"])
    observed = []

    def train(**kwargs):
        model = kwargs["data"]
        observed.append((kwargs["targets"], model.params["w"], model.current_round))
        return [FLModel(params={"w": model.params["w"] + 1}, meta={"trained_by": kwargs["targets"][0]})]

    controller.send_model_and_wait = Mock(side_effect=train)
    saved = []
    controller.save_model = Mock(side_effect=lambda model: saved.append(model.params["w"]))
    monkeypatch.setattr(modern, "cleanup_memory", Mock())
    controller.run()
    assert observed == [(["one"], 0, 0), (["two"], 1, 0), (["one"], 2, 1), (["two"], 3, 1)]
    assert saved == [2, 4]
    assert modern.cleanup_memory.call_count == 2
