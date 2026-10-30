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
import json

import numpy as np
import pandas as pd
import pytest

from nvflare.app_common.abstract.statistics_spec import Bin, DataType, Feature, Histogram, HistogramType
from nvflare.app_common.app_constant import StatisticsConstants as StC
from nvflare.app_common.statistics import hierarchical_numeric_stats as subject
from nvflare.app_common.statistics.numpy_utils import NpEncoder, dtype_to_data_type


def test_hierarchy_levels_and_feature_filter_keep_nested_structure_and_numeric_types():
    hierarchy = {"Orgs": [{StC.NAME: "org", "Sites": ["one", "two"]}]}
    assert subject.get_hierarchical_levels(hierarchy) == {"Orgs": 0, "Sites": 1}
    assert subject.get_client_hierarchy(hierarchy, "two") == ["org"]
    assert subject.get_client_hierarchy(hierarchy, "unknown") is None
    features = {
        "train": [Feature("age", DataType.INT), Feature("score", DataType.FLOAT), Feature("name", DataType.STRING)],
        "empty": [],
    }
    assert subject.filter_numeric_features(features) == {"train": features["train"][:2], "empty": []}


def test_histogram_accumulator_combines_repeated_chunks_at_global_org_and_site_levels():
    hierarchy = {"Orgs": [{StC.NAME: "org", "Sites": ["one"]}]}
    histogram = Histogram(HistogramType.STANDARD, [Bin(0, 1, 2), Bin(1, 2, 3)])
    metrics = {StC.STATS_HISTOGRAM: {"one": {"ds": {"age": histogram}}}}
    output = subject.get_output_structure(metrics, StC.STATS_2nd_STATISTICS, [StC.STATS_HISTOGRAM], hierarchy)
    before = copy.deepcopy(hierarchy)
    for _ in range(2):
        subject.get_hierarchical_histograms(
            StC.STATS_HISTOGRAM, "one", metrics[StC.STATS_HISTOGRAM]["one"], output, hierarchy
        )
    expected = Histogram(HistogramType.STANDARD, [Bin(0, 1, 4), Bin(1, 2, 6)])
    assert output[StC.GLOBAL][StC.STATS_HISTOGRAM]["ds"]["age"] == expected
    org = output["Orgs"][0]
    assert org[StC.GLOBAL][StC.STATS_HISTOGRAM]["ds"]["age"] == expected
    assert org["Sites"][0][StC.LOCAL][StC.STATS_HISTOGRAM]["ds"]["age"] == expected
    assert hierarchy == before
    assert subject.remove_unreported_from_output(7, metrics, [StC.STATS_HISTOGRAM]) == []
    assert subject.remove_unreported_from_output({"metadata": "value"}, metrics, [StC.STATS_HISTOGRAM]) == []


def test_numpy_json_encoder_preserves_scalar_and_array_values_and_rejects_unknown_objects():
    encoded = json.dumps({"integer": np.int64(3), "float": np.float32(0.5), "array": np.array([[1, 2]])}, cls=NpEncoder)
    assert json.loads(encoded) == {"integer": 3, "float": 0.5, "array": [[1, 2]]}
    with pytest.raises(TypeError):
        json.dumps(object(), cls=NpEncoder)


@pytest.mark.parametrize(
    "dtype,expected",
    [
        (np.dtype("timedelta64[s]"), DataType.DATETIME),
        (pd.BooleanDtype(), DataType.INT),
        (pd.StringDtype(), DataType.STRING),
    ],
)
def test_numpy_and_nullable_extension_types_keep_statistics_data_classification(dtype, expected):
    assert dtype_to_data_type(dtype) == expected
