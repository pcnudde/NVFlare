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
import os

import pytest
import torch
from safetensors.torch import load_file, save_file

from nvflare.apis.fl_constant import FLContextKey, WorkspaceConstants
from nvflare.apis.fl_context import FLContext
from nvflare.app_common.abstract.model import ModelLearnableKey, make_model_learnable
from nvflare.app_common.app_constant import AppConstants
from nvflare.app_common.app_event_type import AppEventType
from nvflare.app_opt.pt.lazy_tensor_dict import _LazyRef, safetensors_refs
from nvflare.app_opt.pt.safetensors_model_persistor import PTSafetensorsModelPersistor, load_safetensors_refs


def _write_hf_checkpoint(path, tensors):
    path.mkdir()
    weight_map = {}
    for index, (key, tensor) in enumerate(tensors.items(), start=1):
        shard = f"model-{index:05d}-of-{len(tensors):05d}.safetensors"
        save_file({key: tensor}, path / shard)
        weight_map[key] = shard
    (path / "model.safetensors.index.json").write_text(json.dumps({"metadata": {}, "weight_map": weight_map}))


@pytest.fixture
def fl_ctx(tmp_path):
    ctx = FLContext()
    ctx.set_prop(FLContextKey.APP_ROOT, str(tmp_path / "app"), private=True, sticky=False)
    (tmp_path / "app").mkdir()
    return ctx


def test_load_monolithic_file_returns_unowned_lazy_refs(tmp_path, fl_ctx):
    tensors = {"weight": torch.randn(3, 2), "bias": torch.randn(3)}
    save_file(tensors, tmp_path / "model.safetensors")

    weights = PTSafetensorsModelPersistor(str(tmp_path / "model.safetensors")).load_model(fl_ctx)[
        ModelLearnableKey.WEIGHTS
    ]

    assert all(isinstance(ref, _LazyRef) and ref._temp_ref is None for ref in weights.values())
    assert all(torch.equal(weights[key].materialize(), tensors[key]) for key in tensors)


def test_load_hugging_face_directory_and_index(tmp_path):
    tensors = {"a": torch.ones(2), "b": torch.zeros(3)}
    _write_hf_checkpoint(tmp_path / "hf", tensors)

    for path in (tmp_path / "hf", tmp_path / "hf" / "model.safetensors.index.json"):
        weights = load_safetensors_refs(str(path))
        assert all(torch.equal(weights[key].materialize(), tensors[key]) for key in tensors)


def test_relative_source_path_resolves_against_app_custom_dir(tmp_path, fl_ctx):
    custom = tmp_path / "app" / WorkspaceConstants.CUSTOM_FOLDER_NAME
    custom.mkdir()
    save_file({"a": torch.ones(1)}, custom / "init.safetensors")

    weights = PTSafetensorsModelPersistor("init.safetensors").load_model(fl_ctx)[ModelLearnableKey.WEIGHTS]

    assert torch.equal(weights["a"].materialize(), torch.ones(1))


def test_missing_checkpoint_raises(tmp_path):
    with pytest.raises(ValueError, match="not found"):
        load_safetensors_refs(str(tmp_path / "missing.safetensors"))


def test_save_links_refs_that_cover_one_complete_file(tmp_path, fl_ctx):
    tensors = {"weight": torch.randn(4), "bias": torch.randn(2)}
    aggregate = tmp_path / "aggregate.safetensors"
    save_file(tensors, aggregate)
    persistor = PTSafetensorsModelPersistor(str(aggregate))

    persistor.save_model(make_model_learnable(safetensors_refs(str(aggregate)), {}), fl_ctx)

    saved = tmp_path / "app" / "FL_global_model.safetensors"
    assert os.stat(saved).st_ino == os.stat(aggregate).st_ino
    assert all(torch.equal(load_file(saved)[key], tensors[key]) for key in tensors)
    assert not saved.with_name(saved.name + ".tmp").exists()


def test_save_streams_mixed_tensors_and_refs(tmp_path, fl_ctx):
    save_file({"lazy": torch.arange(6.0), "extra": torch.ones(1)}, tmp_path / "source.safetensors")
    weights = {
        "eager": torch.tensor([[1.0, 2.0]]),
        "lazy": safetensors_refs(str(tmp_path / "source.safetensors"))["lazy"],
    }
    persistor = PTSafetensorsModelPersistor("unused.safetensors")

    persistor.save_model(make_model_learnable(weights, {}), fl_ctx)

    saved = load_file(tmp_path / "app" / "FL_global_model.safetensors")
    assert set(saved) == {"eager", "lazy"}
    assert torch.equal(saved["eager"], weights["eager"])
    assert torch.equal(saved["lazy"], torch.arange(6.0))


def test_failed_save_keeps_previous_checkpoint_and_removes_temp_file(tmp_path, fl_ctx):
    persistor = PTSafetensorsModelPersistor("unused.safetensors")
    persistor.save_model(make_model_learnable({"w": torch.ones(1)}, {}), fl_ctx)

    with pytest.raises(TypeError):
        persistor.save_model(make_model_learnable({"w": [1.0]}, {}), fl_ctx)

    saved = tmp_path / "app" / "FL_global_model.safetensors"
    assert torch.equal(load_file(saved)["w"], torch.ones(1))
    assert os.listdir(tmp_path / "app") == ["FL_global_model.safetensors"]


def test_best_model_event_saves_the_global_model(tmp_path, fl_ctx):
    persistor = PTSafetensorsModelPersistor("unused.safetensors")
    fl_ctx.set_prop(
        AppConstants.GLOBAL_MODEL, make_model_learnable({"w": torch.full((2,), 3.0)}, {}), private=True, sticky=True
    )

    persistor.handle_event(AppEventType.GLOBAL_BEST_MODEL_AVAILABLE, fl_ctx)

    best = load_file(tmp_path / "app" / "best_FL_global_model.safetensors")
    assert torch.equal(best["w"], torch.full((2,), 3.0))
