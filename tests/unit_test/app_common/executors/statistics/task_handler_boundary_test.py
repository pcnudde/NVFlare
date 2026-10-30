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

from nvflare.apis.fl_constant import ReservedKey, ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.apis.signal import Signal
from nvflare.app_common.abstract.statistics_spec import DataType, Feature, StatisticConfig, Statistics
from nvflare.app_common.app_constant import StatisticsConstants as StC
from nvflare.app_common.executors.statistics.statistics_task_handler import StatisticsTaskHandler
from nvflare.fuel.utils import fobs


@pytest.fixture
def runtime():
    generator = Mock(spec=Statistics)
    engine = Mock()
    engine.get_component.return_value = generator
    ctx = FLContext()
    ctx.set_prop(ReservedKey.ENGINE, engine, private=True, sticky=True)
    handler = StatisticsTaskHandler("stats", precision=2)
    handler.initialize(ctx)
    generator.initialize.assert_called_once_with(ctx)
    assert handler.stats_generator is generator
    return handler, generator, ctx


def test_pre_run_builds_histogram_configuration_and_unknown_task_is_rejected(runtime):
    handler, generator, ctx = runtime
    generator.pre_run.return_value = {"available": True}
    data = Shareable()
    data[StC.STATS_TARGET_STATISTICS] = fobs.dumps(
        [
            StatisticConfig(StC.STATS_COUNT, {}),
            StatisticConfig(StC.STATS_HISTOGRAM, {"*": {"bins": 5}, "age": {"bins": 3, "range": [0, 100]}}),
        ]
    )
    assert handler.execute_task(StC.FED_STATS_PRE_RUN, data, ctx, Signal()) == {"available": True}
    generator.pre_run.assert_called_once_with(
        [StC.STATS_COUNT, StC.STATS_HISTOGRAM], {"*": 5, "age": 3}, {"*": None, "age": [0, 100]}
    )
    with pytest.raises(RuntimeError, match=ReturnCode.TASK_UNKNOWN):
        handler.execute_task("unexpected", data, ctx, Signal())
    handler.finalize(ctx)
    generator.finalize.assert_called_once_with(ctx)


@pytest.mark.parametrize("count,total,expected", [(4, 10.123, 2.53), (None, None, 8.77)])
def test_mean_prefers_count_sum_and_falls_back_to_generator(runtime, count, total, expected):
    handler, generator, ctx = runtime
    generator.count.return_value = count
    generator.sum.return_value = total
    generator.mean.return_value = 8.7654
    assert handler.get_mean("ds", "age", StatisticConfig(StC.STATS_MEAN, {}), Shareable(), ctx) == expected
    assert generator.mean.called is (count is None)


def test_variance_requires_global_values_and_rounds_generator_result(runtime):
    handler, generator, ctx = runtime
    config = StatisticConfig(StC.STATS_VAR, {})
    data = Shareable()
    assert handler.get_variance_with_mean("ds", "age", config, data, ctx) is None
    data[StC.STATS_GLOBAL_MEAN] = {"ds": {"age": 2}}
    data[StC.STATS_GLOBAL_COUNT] = {"ds": {"age": 5}}
    generator.variance_with_mean.return_value = 1.2345
    assert handler.get_variance_with_mean("ds", "age", config, data, ctx) == 1.23
    generator.variance_with_mean.assert_called_once_with("ds", "age", 2, 5)
    assert handler._get_global_value_from_input(StC.STATS_GLOBAL_MEAN, "absent", "age", data) is None


@pytest.mark.parametrize("metric,method", [(StC.STATS_MIN, "get_min_value"), (StC.STATS_MAX, "get_max_value")])
def test_histogram_range_limits_override_local_extremes(runtime, metric, method):
    handler, generator, ctx = runtime
    local = generator.min_value if metric == StC.STATS_MIN else generator.max_value
    local.return_value = -5 if metric == StC.STATS_MIN else 200
    config = StatisticConfig(metric, {"age": {"bins": 2, "range": [0, 100]}})
    assert getattr(handler, method)("ds", "age", config, Shareable(), ctx) == (0 if metric == StC.STATS_MIN else 100)
    local.assert_not_called()
    assert getattr(handler, method)("ds", "age", StatisticConfig(metric, {}), Shareable(), ctx) == local.return_value
    local.assert_called_once_with("ds", "age")


def test_standard_deviation_and_quantiles_are_delegated_with_precision_and_targets(runtime):
    handler, generator, ctx = runtime
    generator.stddev.return_value = 1.23456
    assert handler.get_stddev("ds", "age", StatisticConfig(StC.STATS_STDDEV, {}), Shareable(), ctx) == 1.23
    generator.quantiles.return_value = {0.5: 3}
    config = StatisticConfig(StC.STATS_QUANTILE, {"age": [0.25, 0.5]})
    assert handler.get_quantiles_and_centroids("ds", "age", config, Shareable(), ctx) == {0.5: 3}
    generator.quantiles.assert_called_once_with("ds", "age", [0.25, 0.5])


def test_failed_feature_is_omitted_while_other_feature_and_failure_count_survive(runtime):
    handler, generator, ctx = runtime
    generator.features.return_value = {"ds": [Feature("age", DataType.FLOAT), Feature("name", DataType.STRING)]}
    generator.count.return_value = 4
    generator.failure_count.return_value = 1
    generator.sum.side_effect = RuntimeError("unreadable")
    handler.log_exception = Mock()
    data = Shareable()
    data[StC.STATISTICS_TASK_KEY] = StC.STATS_1st_STATISTICS
    data[StC.STATS_TARGET_STATISTICS] = fobs.dumps([StatisticConfig(StC.STATS_SUM, {})])
    result = handler.execute_task(StC.FED_STATS_TASK, data, ctx, Signal())
    stats = fobs.loads(result[StC.STATS_1st_STATISTICS])
    assert stats[StC.STATS_SUM] == {"ds": {}}
    assert stats[StC.STATS_COUNT] == {"ds": {"age": 4}}
    assert stats[StC.STATS_FAILURE_COUNT] == {"ds": {"age": 1}}
    assert fobs.loads(result[StC.STATS_FEATURES]) == {"ds": [Feature("age", DataType.FLOAT)]}
    handler.log_exception.assert_called_once()
