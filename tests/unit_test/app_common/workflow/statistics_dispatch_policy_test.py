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
from nvflare.apis.event_type import EventType
from nvflare.apis.fl_constant import ReservedKey, ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import make_reply
from nvflare.apis.signal import Signal
from nvflare.app_common.abstract.statistics_spec import Bin, Histogram, HistogramType
from nvflare.app_common.app_constant import StatisticsConstants as StC
from nvflare.app_common.workflows.statistics_controller import StatisticsController
from nvflare.fuel.utils import fobs


@pytest.fixture
def runtime():
    controller = StatisticsController({StC.STATS_COUNT: {}, StC.STATS_SUM: {}, StC.STATS_MEAN: {}}, "writer")
    controller.system_panic, controller.fire_event = Mock(), Mock()
    controller.broadcast_and_wait = Mock()
    ctx = FLContext()
    engine = Mock()
    engine.get_clients.return_value = [SimpleNamespace(name="one"), SimpleNamespace(name="two")]
    ctx.set_prop(ReservedKey.ENGINE, engine, private=True, sticky=True)
    controller.start_controller(ctx)
    return controller, ctx, engine


def client_task(result, task_name=StC.FED_STATS_PRE_RUN, statistic=StC.STATS_1st_STATISTICS):
    return SimpleNamespace(
        client=SimpleNamespace(name="one"),
        task=SimpleNamespace(name=task_name, props={StC.STATISTICS_TASK_KEY: statistic}),
        result=result,
    )


def test_pre_run_serializes_config_and_publishes_successful_handshake(runtime):
    controller, ctx, _ = runtime
    controller.statistic_configs[StC.STATS_HISTOGRAM] = {"*": {"bins": 3}}
    controller.pre_run_task_flow(Signal(), ctx)
    dispatched = controller.broadcast_and_wait.call_args.kwargs["task"]
    names = [item.name for item in fobs.loads(dispatched.data[StC.STATS_TARGET_STATISTICS])]
    assert names == [StC.STATS_COUNT, StC.STATS_SUM, StC.STATS_MEAN, StC.STATS_HISTOGRAM]
    task = client_task(DXO(DataKind.STATISTICS, {"ready": True}).to_shareable())
    controller.results_pre_run_cb(task, ctx)
    assert task.result is None
    assert controller.client_handshake_ok == {"one": True}
    assert ctx.get_prop(StC.PRE_RUN_RESULT)["one"].data == {"ready": True}
    controller.fire_event.assert_called_once_with(EventType.PRE_RUN_RESULT_AVAILABLE, ctx)


@pytest.mark.parametrize(
    "code,panic",
    [(ReturnCode.EXECUTION_EXCEPTION, True), (ReturnCode.EXECUTION_RESULT_ERROR, False), ("custom", False)],
)
def test_failed_handshake_clears_payload_and_applies_client_error_policy(runtime, code, panic):
    controller, ctx, _ = runtime
    task = client_task(make_reply(code))
    controller.results_pre_run_cb(task, ctx)
    assert controller.client_handshake_ok == {"one": False}
    assert task.result is None
    assert controller.system_panic.called is panic


def test_statistics_flow_dispatches_inputs_then_aggregates_reported_client_values(runtime):
    controller, ctx, _ = runtime
    controller.client_statistics = {
        StC.STATS_COUNT: {"one": {"ds": {"age": 2}}, "two": {"ds": {"age": 3}}},
        StC.STATS_SUM: {"one": {"ds": {"age": 4}}, "two": {"ds": {"age": 9}}},
        StC.STATS_MEAN: {"one": {"ds": {"age": 2}}, "two": {"ds": {"age": 3}}},
    }
    controller.statistics_task_flow(Signal(), ctx, StC.STATS_1st_STATISTICS)
    dispatched = controller.broadcast_and_wait.call_args.kwargs["task"]
    assert dispatched.props[StC.STATISTICS_TASK_KEY] == StC.STATS_1st_STATISTICS
    assert controller.global_statistics[StC.STATS_COUNT] == {"ds": {"age": 5}}
    assert controller.global_statistics[StC.STATS_MEAN] == {"ds": {"age": 2.6}}
    assert controller._get_result_cb(StC.STATS_1st_STATISTICS) == controller.results_cb
    assert controller.post_fn(controller.task_name, ctx)


@pytest.mark.parametrize("phase", ["initial", "pre-run", "first", "second"])
def test_control_flow_abort_stops_before_later_statistics_phases(runtime, phase):
    controller, ctx, _ = runtime
    signal = Signal()
    controller.pre_run_task_flow = Mock(side_effect=lambda *_: signal.trigger(True) if phase == "pre-run" else None)

    def run_phase(*args):
        if (phase == "first" and args[-1] == StC.STATS_1st_STATISTICS) or (
            phase == "second" and args[-1] == StC.STATS_2nd_STATISTICS
        ):
            signal.trigger(True)

    controller.statistics_task_flow = Mock(side_effect=run_phase)
    controller.post_fn = Mock()
    if phase == "initial":
        signal.trigger(True)
    assert controller.control_flow(signal, ctx) is False
    assert controller.statistics_task_flow.call_count == {"initial": 0, "pre-run": 0, "first": 1, "second": 2}[phase]
    controller.post_fn.assert_not_called()


def test_aborted_task_flows_do_not_broadcast(runtime):
    controller, ctx, _ = runtime
    signal = Signal()
    signal.trigger(True)
    assert controller.pre_run_task_flow(signal, ctx) is False
    assert controller.statistics_task_flow(signal, ctx, StC.STATS_1st_STATISTICS) is False
    controller.broadcast_and_wait.assert_not_called()


def test_combined_histograms_and_quantiles_preserve_global_values_and_round_local_values(runtime):
    controller, _, _ = runtime
    controller.precision = 2
    controller.statistic_configs = {StC.STATS_HISTOGRAM: {}, StC.STATS_QUANTILE: {}}
    histogram = Histogram(HistogramType.STANDARD, [Bin(0.1234, 1.9876, 4)])
    controller.client_statistics = {
        StC.STATS_HISTOGRAM: {"one": {"ds": {"age": histogram}}},
        StC.STATS_QUANTILE: {"one": {"ds": {"age": {StC.STATS_QUANTILE: {0.5: 1.23456}}}}},
    }
    controller.global_statistics = {
        StC.STATS_HISTOGRAM: {"ds": {"age": histogram}},
        StC.STATS_QUANTILE: {"ds": {"age": {0.5: 2.3456}}},
    }
    result = controller._combine_all_statistics()["age"]
    assert result[StC.STATS_HISTOGRAM]["one"]["ds"] == [Bin(0.12, 1.99, 4)]
    assert result[StC.STATS_HISTOGRAM][StC.GLOBAL]["ds"] == histogram.bins
    assert result[StC.STATS_QUANTILE]["one"]["ds"] == {0.5: 1.23}
    assert result[StC.STATS_QUANTILE][StC.GLOBAL]["ds"] == {0.5: 2.3456}
