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

import json
from unittest.mock import Mock

import pytest

from nvflare.apis.fl_constant import ReservedKey
from nvflare.apis.fl_context import FLContext
from nvflare.apis.signal import Signal
from nvflare.app_common.abstract.statistics_spec import Bin, Histogram, HistogramType
from nvflare.app_common.app_constant import StatisticsConstants as StC
from nvflare.app_common.workflows import hierarchical_statistics_controller as subject


@pytest.fixture
def runtime(tmp_path):
    controller = subject.HierarchicalStatisticsController(
        {StC.STATS_COUNT: {}, StC.STATS_SUM: {}, StC.STATS_MEAN: {}},
        "writer",
        min_clients=2,
        precision=2,
        hierarchy_config="hierarchy.json",
    )
    controller.system_panic = Mock()
    controller.broadcast_and_wait = Mock()
    engine = Mock()
    engine.get_workspace.return_value.get_app_config_dir.return_value = str(tmp_path)
    ctx = FLContext()
    ctx.set_prop(ReservedKey.ENGINE, engine, private=True, sticky=True)
    ctx.set_prop(ReservedKey.RUN_NUM, "job", private=False, sticky=True)
    return controller, ctx, tmp_path


@pytest.mark.parametrize("failure", ["missing-setting", "missing-file", "malformed-json", "permission", "unexpected"])
def test_hierarchy_loading_failures_panic_before_dispatch(runtime, monkeypatch, failure):
    controller, ctx, directory = runtime
    if failure == "missing-setting":
        controller.hierarchy_config = None
    elif failure == "malformed-json":
        (directory / "hierarchy.json").write_text("{invalid", encoding="utf-8")
    elif failure in ("permission", "unexpected"):
        error = PermissionError("denied") if failure == "permission" else RuntimeError("reader")
        monkeypatch.setattr(subject, "open", Mock(side_effect=error), raising=False)
    assert controller.statistics_task_flow(Signal(), ctx, StC.STATS_1st_STATISTICS) is False
    controller.system_panic.assert_called_once()
    controller.broadcast_and_wait.assert_not_called()


def test_hierarchy_flow_dispatches_serialized_inputs_and_aggregates_completed_replies(runtime):
    controller, ctx, directory = runtime
    (directory / "hierarchy.json").write_text(json.dumps({"Sites": ["one", "two"]}), encoding="utf-8")
    controller.client_statistics = {
        StC.STATS_COUNT: {"one": {"ds": {"f": 2}}, "two": {"ds": {"f": 3}}},
        StC.STATS_SUM: {"one": {"ds": {"f": 4}}, "two": {"ds": {"f": 12}}},
        StC.STATS_MEAN: {"one": {"ds": {"f": 2}}, "two": {"ds": {"f": 4}}},
    }
    controller.statistics_task_flow(Signal(), ctx, StC.STATS_1st_STATISTICS)
    kwargs = controller.broadcast_and_wait.call_args.kwargs
    assert kwargs["min_responses"] == 2
    assert kwargs["task"].props == {StC.STATISTICS_TASK_KEY: StC.STATS_1st_STATISTICS}
    assert kwargs["task"].data[StC.STATISTICS_TASK_KEY] == StC.STATS_1st_STATISTICS
    assert controller.global_statistics[StC.GLOBAL][StC.STATS_COUNT] == {"ds": {"f": 5}}
    assert controller.global_statistics[StC.GLOBAL][StC.STATS_MEAN] == {"ds": {"f": 3.2}}
    assert controller.global_statistics["Sites"][0][StC.LOCAL][StC.STATS_COUNT] == {"ds": {"f": 2}}
    controller.system_panic.assert_not_called()


def test_aborted_hierarchy_flow_does_not_dispatch_or_change_statistics(runtime):
    controller, ctx, directory = runtime
    (directory / "hierarchy.json").write_text("{}", encoding="utf-8")
    signal = Signal()
    signal.trigger(True)
    assert controller.statistics_task_flow(signal, ctx, StC.STATS_1st_STATISTICS) is False
    assert controller.global_statistics == {}
    controller.broadcast_and_wait.assert_not_called()


def test_rounding_preserves_nested_hierarchy_and_converts_histogram_bins(runtime):
    controller, _, _ = runtime
    histogram = Histogram(HistogramType.STANDARD, [Bin(0.1234, 1.9876, 3.4567)])
    controller.global_statistics = {
        StC.GLOBAL: {StC.STATS_MEAN: {"ds": {"f": 1.23456}}, StC.STATS_HISTOGRAM: {"ds": {"f": histogram}}},
        "Sites": [{StC.NAME: "one", StC.LOCAL: {StC.STATS_MEAN: {"ds": {"f": 2.34567}}}}],
    }
    result = controller._combine_all_statistics()
    assert result[StC.GLOBAL][StC.STATS_MEAN]["ds"]["f"] == 1.23
    assert result["Sites"][0][StC.LOCAL][StC.STATS_MEAN]["ds"]["f"] == 2.35
    assert result[StC.GLOBAL][StC.STATS_HISTOGRAM]["ds"]["f"] == [Bin(0.12, 1.99, 3.4567)]
    nested = [{StC.LOCAL: {StC.STATS_MEAN: {"ds": {"f": 3.45678}}}}]
    assert controller._recursively_round_global_stats(nested)[0][StC.LOCAL][StC.STATS_MEAN]["ds"]["f"] == 3.46
