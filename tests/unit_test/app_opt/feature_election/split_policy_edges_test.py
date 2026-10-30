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

import json

import numpy as np
import pandas as pd
import pytest

from nvflare.app_opt.feature_election.controller import FeatureElectionController
from nvflare.app_opt.feature_election.feature_election import FeatureElection, load_election_results


@pytest.mark.parametrize("strategy", ["random", "sequential", "non_iid", "dirichlet"])
def test_splitting_is_reproducible_assigns_every_row_once_and_preserves_global_rng(strategy):
    frame = pd.DataFrame(
        {"a": range(120), "b": np.arange(120) * 2, "target": np.arange(120) % 2}, index=np.arange(120) + 1000
    )
    state = np.random.get_state()
    subject = FeatureElection()
    first = subject.prepare_data_splits(
        frame, "target", num_clients=4, split_strategy=strategy, dirichlet_alpha=10, random_state=7
    )
    second = subject.prepare_data_splits(
        frame, "target", num_clients=4, split_strategy=strategy, dirichlet_alpha=10, random_state=7
    )
    all_indices = []
    for (features, labels), (again, _) in zip(first, second):
        assert len(features) > 0
        assert features.index.equals(labels.index)
        assert features.equals(again)
        np.testing.assert_array_equal(features["a"] % 2, labels)
        all_indices.extend(features.index)
    assert sorted(all_indices) == list(frame.index)
    after = np.random.get_state()
    assert state[0] == after[0] and state[2:] == after[2:]
    np.testing.assert_array_equal(state[1], after[1])


@pytest.mark.parametrize("strategy", ["random", "sequential", "non_iid"])
def test_split_rejects_empty_client_partitions(strategy):
    frame = pd.DataFrame({"a": [1], "target": [0]})
    with pytest.raises(ValueError, match="0 samples"):
        FeatureElection().prepare_data_splits(
            frame, "target", num_clients=2, split_strategy=strategy, dirichlet_alpha=1
        )


def test_stratified_split_falls_back_only_for_rare_classes():
    frame = pd.DataFrame({"a": range(10), "target": range(10)})
    parts = FeatureElection().prepare_data_splits(frame, "target", num_clients=2)
    assert sum(len(x) for x, y in parts) == 10
    assert set(parts[0][0].index).isdisjoint(parts[1][0].index)
    with pytest.raises(ValueError, match="test_size"):
        FeatureElection().prepare_data_splits(frame, "target", num_clients=2, split_ratios=[1.0, 0.0])


def test_non_iid_warns_that_explicit_ratios_do_not_control_class_proportions(caplog):
    frame = pd.DataFrame({"a": range(80), "target": np.arange(80) % 2})
    parts = FeatureElection().prepare_data_splits(
        frame, "target", num_clients=2, split_strategy="non_iid", split_ratios=[0.5, 0.5], dirichlet_alpha=20
    )
    assert sum(len(x) for x, y in parts) == 80
    assert "split_ratios is ignored" in caplog.text


@pytest.mark.parametrize(
    "field,value", [("aggregation_mode", "unknown"), ("eval_metric", "unknown"), ("tuning_rounds", -1)]
)
def test_result_loader_rejects_corrupted_configuration_without_mutating_election(tmp_path, field, value):
    path = tmp_path / "results.json"
    payload = {field: value}
    path.write_text(json.dumps(payload))
    subject = FeatureElection(freedom_degree=0.75)
    with pytest.raises(ValueError, match=field):
        subject.load_results(str(path))
    assert subject.freedom_degree == 0.75
    assert load_election_results(str(path)) == payload


@pytest.mark.parametrize("mode", ["count", "columns", "empty_mask"])
def test_simulation_rejects_inconsistent_features_and_all_false_selection(mode):
    values = np.column_stack((np.arange(40), np.arange(40) % 3))
    labels = np.arange(40) % 2
    subject = FeatureElection(
        fs_method="lasso" if mode == "empty_mask" else "mutual_info",
        fs_params={"alpha": 100} if mode == "empty_mask" else {"k": 1},
    )
    if mode == "count":
        clients = [(values, labels), (values[:, :1], labels)]
    elif mode == "columns":
        clients = [
            (pd.DataFrame(values, columns=["a", "b"]), labels),
            (pd.DataFrame(values, columns=["x", "y"]), labels),
        ]
    else:
        clients = [(values, labels)]
    with pytest.raises(
        ValueError,
        match={"count": "same number of features", "columns": "column labels", "empty_mask": "rejected all features"}[
            mode
        ],
    ):
        subject.simulate_election(clients)


@pytest.mark.parametrize("rounds", [0, 1])
def test_auto_tuning_disables_inadequate_search_budget(rounds, caplog):
    subject = FeatureElectionController(auto_tune=True, tuning_rounds=rounds)
    assert subject.tuning_rounds == 0
    assert "auto_tune" in caplog.text
    with pytest.raises(ValueError, match="aggregation_mode"):
        FeatureElectionController(aggregation_mode="unknown")
