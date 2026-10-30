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

import numpy as np
import pytest

from nvflare.apis.dxo import DXO, DataKind
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.app_common.filters import svt_privacy as subject
from nvflare.app_common.filters.exclude_vars import ExcludeVars


@pytest.mark.parametrize("settings", [None, "", 7, ["layer", 7]])
def test_invalid_or_empty_exclusion_configuration_preserves_weights(settings):
    filter = ExcludeVars(settings)
    weights = {"layer": np.array([1, 2])}
    dxo = DXO(DataKind.WEIGHTS, weights)
    assert filter.skip
    assert filter.process_dxo(dxo, Shareable(), FLContext()) is None
    assert dxo.data is weights and list(weights) == ["layer"]


def test_unmatched_exclusion_regex_warns_without_dropping_parameters(monkeypatch):
    filter = ExcludeVars("^absent")
    warning = Mock()
    monkeypatch.setattr(filter, "log_warning", warning)
    dxo = DXO(DataKind.WEIGHTS, {"layer": [1]})
    assert filter.process_dxo(dxo, Shareable(), FLContext()) is dxo
    assert dxo.data == {"layer": [1]}
    assert "No matching layers" in warning.call_args.args[1]


@pytest.mark.parametrize(
    "epsilon,count,threshold,query",
    [
        (0, 1, None, None),
        (1, 0, None, None),
        (1, 1, 0, None),
        (1, 1, None, 0),
        (1, 1, 2, None),
        (1, 1, 0.2, 0.3),
    ],
)
def test_invalid_privacy_split_cannot_allocate_negative_or_excess_budget(epsilon, count, threshold, query):
    with pytest.raises(ValueError):
        subject._compute_epsilon_split(epsilon, count, threshold, query)


@pytest.mark.parametrize("threshold,query", [(None, 0.3), (0.3, None)])
def test_partial_privacy_split_allocates_exact_remainder(threshold, query):
    first, second = subject._compute_epsilon_split(1, 2, threshold, query)
    assert first + second == pytest.approx(1)
    assert first == pytest.approx(0.7 if threshold is None else 0.3)
    assert second == pytest.approx(0.3 if query is not None else 0.7)


@pytest.mark.parametrize("count,noise,release", [(0, 1, None), (1, 1, 0), (1, 0, None)])
def test_invalid_release_budget_is_rejected_before_noise_sampling(count, noise, release):
    with pytest.raises(ValueError):
        subject._compute_release_epsilon(count, noise, release)


def test_empty_weight_array_returns_unchanged_without_spending_privacy_budget():
    filter = subject.SVTPrivacy()
    dxo = DXO(DataKind.WEIGHTS, {"empty": np.array([])})
    assert filter.process_dxo(dxo, Shareable(), FLContext()) is dxo
    assert dxo.data["empty"].size == 0
    assert filter.get_privacy_spent() == 0 and filter.get_privacy_ledger() == []
