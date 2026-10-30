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

from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest

from nvflare.apis.dxo import DXO, DataKind
from nvflare.apis.fl_constant import FLContextKey, ReservedKey
from nvflare.apis.fl_context import FLContext
from nvflare.app_common.abstract.model import ModelLearnableKey, make_model_learnable
from nvflare.app_common.app_constant import AppConstants
from nvflare.app_common.app_event_type import AppEventType
from nvflare.app_common.ccwf.comps.np_file_model_persistor import NPFileModelPersistor
from nvflare.app_common.ccwf.comps.simple_model_shareable_generator import SimpleModelShareableGenerator
from nvflare.app_common.np.constants import NPConstants


def context(root):
    ctx = FLContext()
    workspace = Mock()
    workspace.get_run_dir.return_value = str(root)
    ctx.set_prop(FLContextKey.WORKSPACE_OBJECT, workspace, private=True, sticky=True)
    ctx.set_prop(ReservedKey.RUN_NUM, "job", private=True, sticky=True)
    return ctx


def test_numpy_persistor_inventory_and_loading_follow_last_and_best_checkpoints(tmp_path):
    ctx = context(tmp_path)
    persistor = NPFileModelPersistor()
    assert persistor.get_model("missing", ctx) is None
    model = make_model_learnable({NPConstants.NUMPY_KEY: np.array([2.0, 3.0])}, {})
    persistor.save_model(model, ctx)
    ctx.set_prop(AppConstants.GLOBAL_MODEL, model, private=True, sticky=True)
    persistor.handle_event(AppEventType.GLOBAL_BEST_MODEL_AVAILABLE, ctx)
    inventory = persistor.get_model_inventory(ctx)
    assert set(inventory) == {"best_global_model", "last_global_model"}
    assert all(desc.model_format == "np" and Path(desc.location).is_file() for desc in inventory.values())
    loaded = persistor.get_model("best_global_model", ctx)
    np.testing.assert_array_equal(loaded[ModelLearnableKey.WEIGHTS][NPConstants.NUMPY_KEY], [2, 3])
    assert persistor.get_model("unknown", ctx) is None
    Path(inventory["best_global_model"].location).write_bytes(b"not a numpy checkpoint")
    assert persistor.get_model("best_global_model", ctx) is None


def test_numpy_persistor_handles_inventory_entry_deleted_before_load(tmp_path, monkeypatch):
    persistor = NPFileModelPersistor()
    desc = Mock(location=str(tmp_path / "removed.npy"))
    monkeypatch.setattr(persistor, "get_model_inventory", Mock(return_value={"best": desc}))
    assert persistor.get_model("best", context(tmp_path)) is None


def test_model_shareable_generator_round_trips_weights_and_applies_diffs_to_global_base():
    ctx = FLContext()
    generator = SimpleModelShareableGenerator()
    model = make_model_learnable({"w": np.array([1.0, 2.0])}, {"round": 1})
    wire = generator.learnable_to_shareable(model, ctx)
    restored = generator.shareable_to_learnable(wire, ctx)
    np.testing.assert_array_equal(restored[ModelLearnableKey.WEIGHTS]["w"], [1, 2])
    assert restored[ModelLearnableKey.META] == {"round": 1}
    ctx.set_prop(AppConstants.GLOBAL_MODEL, model, private=True, sticky=True)
    diff = DXO(DataKind.WEIGHT_DIFF, {"w": np.array([2.0, 3.0])}, meta={"round": 2}).to_shareable()
    updated = generator.shareable_to_learnable(diff, ctx)
    assert updated is model
    np.testing.assert_array_equal(updated[ModelLearnableKey.WEIGHTS]["w"], [3, 5])
    assert updated[ModelLearnableKey.META] == {"round": 2}


def test_generator_rejects_wrong_payload_kinds_and_missing_diff_base():
    generator = SimpleModelShareableGenerator()
    ctx = FLContext()
    with pytest.raises(TypeError, match="Shareable"):
        generator.shareable_to_learnable({}, ctx)
    with pytest.raises(ValueError, match="data_kind"):
        generator.shareable_to_learnable(DXO(DataKind.METRICS, {"loss": 1}).to_shareable(), ctx)
    generator.system_panic = Mock()
    assert generator.shareable_to_learnable(DXO(DataKind.WEIGHT_DIFF, {"w": 1}).to_shareable(), ctx) is None
    generator.system_panic.assert_called_once()
    result = generator.shareable_to_learnable(DXO(DataKind.WEIGHTS, {}).to_shareable(), ctx)
    assert ModelLearnableKey.WEIGHTS not in result
