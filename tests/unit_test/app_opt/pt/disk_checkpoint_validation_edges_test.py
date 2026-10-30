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
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

torch = pytest.importorskip("torch")
safetensors = pytest.importorskip("safetensors.torch")

from nvflare.apis.fl_constant import FLContextKey  # noqa: E402
from nvflare.apis.fl_context import FLContext  # noqa: E402
from nvflare.app_common.abstract.fl_model import FLModel, ParamsType  # noqa: E402
from nvflare.app_common.app_constant import AppConstants  # noqa: E402
from nvflare.app_opt.pt import disk_fedavg_components as module  # noqa: E402
from nvflare.app_opt.pt.lazy_tensor_dict import safetensors_refs  # noqa: E402


@pytest.mark.parametrize("mapping", [{}, []])
def test_disk_checkpoint_index_requires_nonempty_mapping(tmp_path, mapping):
    index = tmp_path / "model.safetensors.index.json"
    index.write_text(json.dumps({"weight_map": mapping}))
    with pytest.raises(ValueError, match="non-empty weight_map"):
        module.DiskFedAvgPersistor._load_refs(str(index))


def test_disk_checkpoint_rejects_outside_shards_missing_tensors_and_non_safetensor_input(tmp_path):
    index = tmp_path / "model.safetensors.index.json"
    index.write_text(json.dumps({"weight_map": {"w": "../outside.safetensors"}}))
    with pytest.raises(ValueError, match="outside the checkpoint directory"):
        module.DiskFedAvgPersistor._load_refs(str(index))
    safetensors.save_file({"different": torch.tensor([1.0])}, tmp_path / "model.safetensors")
    index.write_text(json.dumps({"weight_map": {"w": "model.safetensors"}}))
    with pytest.raises(ValueError, match="missing tensors"):
        module.DiskFedAvgPersistor._load_refs(str(index))
    with pytest.raises(ValueError, match="requires a safetensors checkpoint"):
        module.DiskFedAvgPersistor._load_refs(str(tmp_path / "model.pt"))
    safetensors.save_file({}, tmp_path / "empty.safetensors")
    with pytest.raises(ValueError, match="contains no tensors"):
        module.DiskFedAvgPersistor._load_refs(str(tmp_path / "empty.safetensors"))


def test_rejected_contribution_and_invalid_diff_base_leave_no_partial_aggregation_output(tmp_path):
    ctx = FLContext()
    ctx.set_prop(FLContextKey.APP_ROOT, str(tmp_path))
    aggregator = module.DiskFedAvgAggregator()
    aggregator.fl_ctx = ctx
    with pytest.raises(RuntimeError, match="no accepted contributions"):
        aggregator.aggregate_model()
    with pytest.raises(TypeError, match="string keys and lazy tensor refs"):
        aggregator.accept_model(FLModel(params={"w": torch.tensor([1.0])}, params_type=ParamsType.FULL))
    assert not aggregator._contributions
    source = tmp_path / "input.safetensors"
    safetensors.save_file({"w": torch.tensor([1.0])}, source)
    aggregator.accept_model(FLModel(params=safetensors_refs(str(source)), params_type=ParamsType.DIFF))
    ctx.set_prop(AppConstants.GLOBAL_MODEL, {"weights": {"w": torch.tensor([5.0])}})
    with pytest.raises(TypeError, match="requires a lazy global model"):
        aggregator.aggregate_model()
    assert not aggregator._contributions
    assert not (tmp_path / module.CURRENT_MODEL).exists()
    assert not (tmp_path / (module.CURRENT_MODEL + ".next")).exists()


def test_disk_checkpoint_requires_app_root_before_attempting_to_open_files(monkeypatch):
    monkeypatch.setattr(module, "fobs", SimpleNamespace(register=Mock()))
    with pytest.raises(RuntimeError, match="requires APP_ROOT"):
        module.DiskFedAvgPersistor("initial.safetensors").load_model(FLContext())
