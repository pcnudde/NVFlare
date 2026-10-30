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

import numpy as np
import pandas as pd
import pytest

from nvflare.app_common.abstract.statistics_spec import DataType
from nvflare.app_common.app_constant import StatisticsConstants
from nvflare.app_opt.statistics.df import df_core_statistics as subject


def test_dataframe_statistics_preserve_dataset_schema_and_ignore_missing_values():
    calculator = subject.DFStatisticsCore(max_bin=8)
    calculator.data = {"train": pd.DataFrame({"value": [1.0, 2.0, 3.0, np.nan], "label": [1, 1, 0, 0]})}
    schema = calculator.features()["train"]
    assert [(item.feature_name, item.data_type) for item in schema] == [
        ("value", DataType.FLOAT),
        ("label", DataType.INT),
    ]
    assert calculator.count("train", "value") == 3
    assert calculator.sum("train", "value") == 6
    assert isinstance(calculator.sum("train", "label"), int)
    assert calculator.mean("train", "value") == 2
    assert calculator.stddev("train", "value") == 1
    assert calculator.variance_with_mean("train", "value", 2, 3) == 1
    assert calculator.min_value("train", "value") == 1
    assert calculator.max_value("train", "value") == 3
    histogram = calculator.histogram("train", "label", 2, 0, 2)
    assert [bucket.sample_count for bucket in histogram.bins] == [2, 2]


def test_dataframe_quantile_digest_exports_real_approximate_distribution():
    pytest.importorskip("fastdigest")
    calculator = subject.DFStatisticsCore(max_bin=20)
    calculator.data = {"train": pd.DataFrame({"value": np.arange(101, dtype=float)})}
    result = calculator.quantiles("train", "value", [0.25, 0.5, 0.75])
    quantiles = result[StatisticsConstants.STATS_QUANTILE]
    assert quantiles[0.25] == pytest.approx(25, abs=2)
    assert quantiles[0.5] == pytest.approx(50, abs=2)
    assert quantiles[0.75] == pytest.approx(75, abs=2)
    assert result[StatisticsConstants.STATS_DIGEST_COORD]
