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

from nvflare.apis.controller_spec import Task
from nvflare.apis.fl_constant import ReservedKey
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import ReturnCode, Shareable, make_reply
from nvflare.apis.signal import Signal
from nvflare.app_common.abstract.learnable_persistor import LearnablePersistor
from nvflare.app_common.abstract.shareable_generator import ShareableGenerator
from nvflare.app_common.app_constant import AppConstants
from nvflare.app_common.app_event_type import AppEventType
from nvflare.app_common.workflows.splitnn_workflow import SplitNNConstants, SplitNNController
from nvflare.widgets.info_collector import GroupInfoCollector, InfoCollector


@pytest.mark.parametrize(
    "name",
    [
        "num_rounds",
        "start_round",
        "task_timeout",
        "persistor_id",
        "shareable_generator_id",
        "init_model_task_name",
        "train_task_name",
    ],
)
def test_splitnn_configuration_rejects_wrong_types(name):
    with pytest.raises(TypeError):
        SplitNNController(**{name: []})


@pytest.mark.parametrize("name", ["num_rounds", "start_round"])
def test_negative_round_configuration_is_rejected(name):
    with pytest.raises(ValueError):
        SplitNNController(**{name: -1})


@pytest.fixture
def runtime():
    ctl = SplitNNController(num_rounds=3, start_round=1, batch_size=16)
    persistor = Mock(spec=LearnablePersistor)
    persistor.load.return_value = {"weights": [1]}
    generator = Mock(spec=ShareableGenerator)
    generator.learnable_to_shareable.return_value = Shareable({"weights": [1]})
    engine = Mock()
    engine.get_component.side_effect = {ctl.persistor_id: persistor, ctl.shareable_generator_id: generator}.get
    engine.get_clients.return_value = [SimpleNamespace(name="site-1"), SimpleNamespace(name="site-2")]
    ctx = FLContext()
    ctx.set_prop(ReservedKey.ENGINE, engine, private=True, sticky=True)
    ctl.system_panic = Mock()
    ctl.fire_event = Mock()
    ctl.broadcast_and_wait = Mock()
    return ctl, ctx, persistor, generator, engine


def test_start_loads_global_model_and_broadcasts_two_phase_training_contract(runtime):
    ctl, ctx, persistor, generator, _ = runtime
    ctl.start_controller(ctx)
    assert ctx.get_prop(AppConstants.GLOBAL_MODEL) == {"weights": [1]}
    assert ctx.get_prop(AppConstants.NUM_ROUNDS) == 3
    assert ctx.get_prop(AppConstants.START_ROUND) == 1
    ctl.fire_event.assert_called_once_with(AppEventType.INITIAL_MODEL_LOADED, ctx)
    signal = Signal()
    ctl.control_flow(signal, ctx)
    calls = ctl.broadcast_and_wait.call_args_list
    assert len(calls) == 2
    initialize, training = [c.kwargs["task"] for c in calls]
    assert initialize.name == ctl.init_model_task_name
    assert initialize.data == generator.learnable_to_shareable.return_value
    assert training.name == ctl.train_task_name
    assert training.data.get_header(AppConstants.NUM_ROUNDS) == 3
    assert training.data.get_header(SplitNNConstants.BATCH_SIZE) == 16
    assert training.data.get_header(SplitNNConstants.TARGET_NAMES) == ["site-1", "site-2"]
    assert all(c.kwargs["min_responses"] == 2 and c.kwargs["abort_signal"] is signal for c in calls)
    assert ctl._phase == AppConstants.PHASE_FINISHED
    ctl.system_panic.assert_not_called()
    persistor.load.assert_called_once_with(ctx)


@pytest.mark.parametrize("bad", ["persistor", "generator"])
def test_start_reports_wrong_component_contracts(runtime, bad):
    ctl, ctx, persistor, generator, engine = runtime
    if bad == "persistor":
        persistor = Mock()
    else:
        generator = Mock()
    engine.get_component.side_effect = {ctl.persistor_id: persistor, ctl.shareable_generator_id: generator}.get
    ctl.start_controller(ctx)
    ctl.system_panic.assert_called_once()
    assert ("Persistor" if bad == "persistor" else "Shareable generator") in ctl.system_panic.call_args.args[0]


@pytest.mark.parametrize(
    "rc",
    [
        ReturnCode.MISSING_PEER_CONTEXT,
        ReturnCode.BAD_PEER_CONTEXT,
        ReturnCode.EXECUTION_EXCEPTION,
        ReturnCode.TASK_UNKNOWN,
        ReturnCode.EXECUTION_RESULT_ERROR,
        ReturnCode.TASK_DATA_FILTER_ERROR,
        ReturnCode.TASK_RESULT_FILTER_ERROR,
    ],
)
@pytest.mark.parametrize("ignore", [False, True])
def test_result_error_policy_decides_whether_to_panic(runtime, rc, ignore):
    ctl, ctx, _, _, _ = runtime
    ctl.ignore_result_error = ignore
    task = Task("train", Shareable())
    result = make_reply(rc)
    assert ctl._process_result(SimpleNamespace(task=task, result=result), ctx) is False
    assert task.get_prop(SplitNNConstants.TASK_RESULT) is None
    assert ctl.system_panic.called is not ignore


def test_successful_result_is_attached_to_owning_task(runtime):
    ctl, ctx, _, _, _ = runtime
    task = Task("train", Shareable())
    result = Shareable({"gradient": [1]})
    assert ctl._process_result(SimpleNamespace(task=task, result=result), ctx)
    assert task.get_prop(SplitNNConstants.TASK_RESULT) is result


def test_unexpected_client_and_control_flow_exception_are_reported(runtime):
    ctl, ctx, _, _, engine = runtime
    engine.get_clients.return_value = [SimpleNamespace(name="unexpected")]
    ctl._check_targets(ctx)
    assert "unexpected" in ctl.system_panic.call_args.args[0]
    ctl.system_panic.reset_mock()
    ctl._init_models = Mock(side_effect=ValueError("initialization failed"))
    engine.get_clients.return_value = []
    ctl.control_flow(Signal(), ctx)
    assert "initialization failed" in ctl.system_panic.call_args.args[0]
    ctl.broadcast_and_wait.assert_not_called()


def test_stats_collection_validates_collector_and_reports_workflow_state(runtime):
    ctl, ctx, _, _, _ = runtime
    collector = Mock(spec=GroupInfoCollector)
    ctx.set_prop(InfoCollector.CTX_KEY_STATS_COLLECTOR, collector, private=True, sticky=False)
    ctl.handle_event(InfoCollector.EVENT_TYPE_GET_STATS, ctx)
    assert collector.add_info.call_args.kwargs["info"] == {
        "phase": AppConstants.PHASE_INIT,
        "current_round": None,
        "num_rounds": 3,
    }
    ctx.set_prop(InfoCollector.CTX_KEY_STATS_COLLECTOR, object(), private=True, sticky=False)
    with pytest.raises(TypeError, match="GroupInfoCollector"):
        ctl.handle_event(InfoCollector.EVENT_TYPE_GET_STATS, ctx)
    ctl.stop_controller(ctx)
    assert ctl._phase == AppConstants.PHASE_FINISHED
    ctl.log_warning = Mock()
    ctl.process_result_of_unknown_task(SimpleNamespace(name="site"), "unknown", "id", Shareable(), ctx)
    assert "unknown" in ctl.log_warning.call_args.args[1]
