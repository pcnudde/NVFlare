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
from unittest.mock import MagicMock

import pytest

from nvflare.apis.dxo import DXO, DataKind, from_shareable
from nvflare.apis.fl_constant import ReservedKey, ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.app_common.abstract.model import ModelLearnableKey, make_model_learnable
from nvflare.app_common.app_constant import AppConstants
from nvflare.app_opt.xgboost.tree_based.bagging_aggregator import XGBBaggingAggregator
from nvflare.app_opt.xgboost.tree_based.shareable_generator import XGBModelShareableGenerator


def _tree_model(value):
    return {
        "learner": {
            "gradient_booster": {
                "model": {
                    "gbtree_model_param": {"num_parallel_tree": "1", "num_trees": "1"},
                    "trees": [{"id": 0, "value": value}],
                    "tree_info": [0],
                    "iteration_indptr": [0, 1],
                }
            }
        }
    }


def _contribution(site="site", round=2, kind=DataKind.WEIGHTS, rc=ReturnCode.OK):
    shareable = DXO(kind, {"model_data": json.dumps(_tree_model(3))}).to_shareable()
    shareable.set_peer_props({ReservedKey.IDENTITY_NAME: site})
    shareable.add_cookie(AppConstants.CONTRIBUTION_ROUND, round)
    shareable.set_return_code(rc)
    return shareable


def test_bagging_collects_unique_round_contributions_and_clears_pending_state():
    subject = XGBBaggingAggregator()
    context = FLContext()
    context.set_prop(AppConstants.CURRENT_ROUND, 2)
    assert subject.accept(_contribution("a"), context)
    assert not subject.accept(_contribution("a"), context)
    assert subject.accept(_contribution("b"), context)
    result = from_shareable(subject.aggregate(context))
    assert result.data_kind == DataKind.WEIGHTS
    assert len(result.data["model_data"]) == 2
    assert result.data["model_data_dict"] == [_tree_model(3), _tree_model(3)]
    assert all(json.loads(value) == _tree_model(3) for value in result.data["model_data"])
    assert subject.history == subject.local_models == subject.local_models_as_dict == []
    assert subject.accept(_contribution("a"), context)


@pytest.mark.parametrize("mode", ["invalid", "wrong_round", "wrong_kind", "error_code"])
def test_bagging_rejects_invalid_contributions_without_retaining_them(mode):
    subject = XGBBaggingAggregator()
    context = FLContext()
    context.set_prop(AppConstants.CURRENT_ROUND, 2)
    kwargs = {
        "wrong_round": {"round": 1},
        "wrong_kind": {"kind": DataKind.METRICS},
        "error_code": {"rc": ReturnCode.EXECUTION_EXCEPTION},
    }.get(mode, {})
    contribution = Shareable() if mode == "invalid" else _contribution(**kwargs)
    assert not subject.accept(contribution, context)
    assert subject.history == subject.local_models == subject.local_models_as_dict == []


@pytest.mark.parametrize("weights", [{}, b"serialized-model", {"learner": {"version": 1}}])
def test_initial_model_exchange_serializes_bagging_and_preserves_cyclic_weights(weights):
    subject = XGBModelShareableGenerator()
    result = from_shareable(subject.learnable_to_shareable(make_model_learnable(weights, {}), FLContext()))
    assert result.data_kind == DataKind.WEIGHTS
    if isinstance(weights, dict) and weights:
        assert json.loads(result.data["model_data"]) == weights
    elif weights:
        assert result.data["model_data"] == weights
    else:
        assert result.data == {}


def test_bagging_updates_append_tree_ids_and_remove_server_only_dict_from_wire():
    subject = XGBModelShareableGenerator()
    context = FLContext()
    base = make_model_learnable(_tree_model(1), {})
    context.set_prop(AppConstants.GLOBAL_MODEL, base)
    update = DXO(
        DataKind.WEIGHTS, {"model_data": ["delta"], "model_data_dict": [_tree_model(2), _tree_model(3)]}
    ).to_shareable()
    assert subject.shareable_to_learnable(update, context) is base
    model = base[ModelLearnableKey.WEIGHTS]["learner"]["gradient_booster"]["model"]
    assert model["gbtree_model_param"]["num_trees"] == "3"
    assert [tree["id"] for tree in model["trees"]] == [0, 1, 2]
    assert [tree["value"] for tree in model["trees"]] == [1, 2, 3]
    cached = subject.learnable_to_shareable(base, context)
    assert from_shareable(cached).data == {"model_data": ["delta"]}


@pytest.mark.parametrize("data", [{}, {"model_data": b"new-model"}])
def test_cyclic_or_empty_update_returns_mutated_base_and_cached_shareable(data):
    subject = XGBModelShareableGenerator()
    context = FLContext()
    base = make_model_learnable(b"base-model", {})
    context.set_prop(AppConstants.GLOBAL_MODEL, base)
    result = subject.shareable_to_learnable(DXO(DataKind.WEIGHTS, data).to_shareable(), context)
    assert result[ModelLearnableKey.WEIGHTS] == (data["model_data"] if data else b"base-model")
    assert from_shareable(subject.shareable).data == data


def test_shareable_generator_rejects_type_kind_and_missing_base():
    subject = XGBModelShareableGenerator()
    context = FLContext()
    with pytest.raises(TypeError, match="Shareable"):
        subject.shareable_to_learnable({}, context)
    subject.system_panic = MagicMock()
    assert subject.shareable_to_learnable(Shareable(), context) is None
    subject.system_panic.assert_called_once_with(reason="No global base model!", fl_ctx=context)
    context.set_prop(AppConstants.GLOBAL_MODEL, make_model_learnable(b"model", {}))
    with pytest.raises(ValueError, match="data_kind"):
        subject.shareable_to_learnable(DXO(DataKind.METRICS, {"loss": 1}).to_shareable(), context)
