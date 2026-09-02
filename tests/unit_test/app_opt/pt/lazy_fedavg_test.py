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

"""FedAvg with enable_tensor_disk_offload: lazy aggregation end to end, without a real Cell."""

import pytest
import torch
from safetensors.torch import load_file, save_file

import nvflare.app_common.utils.tensor_disk_offload_context as tensor_disk_offload_context
from nvflare.apis.fl_constant import FLContextKey, FLMetaKey, ReservedKey
from nvflare.apis.fl_context import FLContext
from nvflare.apis.job_def import SERVER_SITE_NAME
from nvflare.apis.signal import Signal
from nvflare.app_common.abstract.fl_model import FLModel, ParamsType
from nvflare.app_common.abstract.model import ModelLearnableKey, make_model_learnable
from nvflare.app_common.aggregators.weighted_aggregation_helper import WeightedAggregationHelper
from nvflare.app_common.app_constant import AppConstants
from nvflare.app_common.workflows.fedavg import FedAvg
from nvflare.app_opt.pt.lazy_aggregation import LazyWeightedAggregationHelper
from nvflare.app_opt.pt.lazy_tensor_dict import _LazyRef, safetensors_refs
from nvflare.app_opt.pt.model_persistence_format_manager import PTModelPersistenceFormatManager
from nvflare.app_opt.pt.safetensors_model_persistor import PTSafetensorsModelPersistor
from nvflare.fuel.utils.fobs import FOBSContextKey


class _Cell:
    def __init__(self):
        self.context = {}

    def get_fobs_context(self):
        return dict(self.context)

    def update_fobs_context(self, props):
        self.context.update(props)


class _Engine:
    def __init__(self, cell):
        self.cell = cell

    def get_cell(self):
        return self.cell


@pytest.fixture
def offload_root(tmp_path, monkeypatch):
    """Place the job's offload root under tmp_path so the test can check its cleanup."""
    root = tmp_path / "offload_root"
    original_mkdtemp = tensor_disk_offload_context.tempfile.mkdtemp

    def fake_mkdtemp(prefix="", dir=None, **kwargs):
        if dir is None and prefix.startswith("nvflare_tensor_offload_"):
            root.mkdir()
            return str(root)
        return original_mkdtemp(prefix=prefix, dir=dir, **kwargs)

    monkeypatch.setattr(tensor_disk_offload_context.tempfile, "mkdtemp", fake_mkdtemp)
    return root


def _checkpoint(path, tensors) -> str:
    save_file(tensors, path)
    return str(path)


def _client_params(path, tensors) -> dict:
    save_file(tensors, path)
    return safetensors_refs(str(path))


def _result(params, client_name, steps=1, params_type=ParamsType.FULL, metrics=None) -> FLModel:
    return FLModel(
        params=params,
        params_type=params_type,
        metrics=metrics,
        meta={"client_name": client_name, FLMetaKey.NUM_STEPS_CURRENT_ROUND: steps},
    )


def _controller(tmp_path, persistor, *, num_clients=1, offload=True, **kwargs) -> FedAvg:
    engine = _Engine(_Cell())
    fl_ctx = FLContext()
    fl_ctx.set_prop(FLContextKey.APP_ROOT, str(tmp_path / "app"), private=True, sticky=False)
    fl_ctx.set_prop(ReservedKey.RUN_NUM, "lazy-fedavg-test", private=True, sticky=False)
    fl_ctx.set_prop(ReservedKey.ENGINE, engine, private=True, sticky=False)

    controller = FedAvg(num_clients=num_clients, enable_tensor_disk_offload=offload, **kwargs)
    controller.persistor = persistor
    controller.fl_ctx = fl_ctx
    controller.engine = engine
    controller.abort_signal = Signal()
    controller.event = lambda *_: None
    controller.fire_event_with_data = lambda *_: None
    controller.info = lambda *_: None
    controller.sample_clients = lambda _: [f"site-{index + 1}" for index in range(num_clients)]
    controller.get_num_standing_tasks = lambda: 0
    return controller


def test_offload_aggregates_lazily_and_saves_a_linked_checkpoint(tmp_path, offload_root):
    initial = {"weight": torch.tensor([1.0, 3.0]), "bias": torch.tensor([2.0])}
    persistor = PTSafetensorsModelPersistor(_checkpoint(tmp_path / "initial.safetensors", initial))
    controller = _controller(tmp_path, persistor, num_clients=2, num_rounds=1)
    seen = {}

    def send_model(task_name, targets, data, callback, **kwargs):
        seen["helper"] = type(controller._aggr_helper)
        seen["outbound"] = {type(value) for value in data.params.values()}
        for index, (client, delta, steps) in enumerate((("site-1", 1.0, 1), ("site-2", 3.0, 3)), start=1):
            params = _client_params(
                tmp_path / f"client-{index}.safetensors", {k: v + delta for k, v in initial.items()}
            )
            callback(_result(params, client, steps))

    controller.send_model = send_model
    controller.run()

    assert seen["helper"] is LazyWeightedAggregationHelper
    assert seen["outbound"] == {_LazyRef}
    saved = load_file(tmp_path / "app" / "FL_global_model.safetensors")
    assert torch.equal(saved["weight"], torch.tensor([3.5, 5.5]))
    assert torch.equal(saved["bias"], torch.tensor([4.5]))
    global_model = controller.fl_ctx.get_prop(AppConstants.GLOBAL_MODEL)[ModelLearnableKey.WEIGHTS]
    assert all(isinstance(ref, _LazyRef) for ref in global_model.values())
    assert not offload_root.exists()
    assert controller.engine.cell.context[FOBSContextKey.TENSOR_DISK_OFFLOAD] is False


def test_offload_applies_diff_to_the_lazy_base_model(tmp_path):
    initial = {"weight": torch.tensor([10.0, 20.0])}
    persistor = PTSafetensorsModelPersistor(_checkpoint(tmp_path / "initial.safetensors", initial))
    controller = _controller(tmp_path, persistor, num_rounds=1)

    def send_model(task_name, targets, data, callback, **kwargs):
        diff = _client_params(tmp_path / "diff.safetensors", {"weight": torch.tensor([1.0, 3.0])})
        callback(_result(diff, "site-1", params_type=ParamsType.DIFF))

    controller.send_model = send_model
    controller.run()

    saved = load_file(tmp_path / "app" / "FL_global_model.safetensors")
    assert torch.equal(saved["weight"], torch.tensor([11.0, 23.0]))


def test_early_stopping_saves_only_the_best_model(tmp_path):
    persistor = PTSafetensorsModelPersistor(
        _checkpoint(tmp_path / "initial.safetensors", {"weight": torch.tensor([1.0])})
    )
    controller = _controller(tmp_path, persistor, num_rounds=3, stop_cond="accuracy >= 10", patience=1)
    sent = []

    def send_model(task_name, targets, data, callback, **kwargs):
        current = data.params["weight"].materialize()
        sent.append(current.item())
        params = _client_params(tmp_path / f"round-{data.current_round}.safetensors", {"weight": current + 1})
        callback(_result(params, "site-1", metrics={"accuracy": -float(data.current_round)}))

    controller.send_model = send_model
    controller.run()

    assert controller.current_round == 1
    assert sent == [1.0, 2.0]  # round 1 trained on the unsaved round-0 aggregate
    saved = load_file(tmp_path / "app" / "FL_global_model.safetensors")
    assert torch.equal(saved["weight"], torch.tensor([2.0]))


def test_without_offload_in_memory_aggregation_saves_through_the_persistor(tmp_path):
    initial = {"weight": torch.tensor([1.0, 3.0])}
    persistor = PTSafetensorsModelPersistor(_checkpoint(tmp_path / "initial.safetensors", initial))
    controller = _controller(tmp_path, persistor, num_rounds=1, offload=False)

    def send_model(task_name, targets, data, callback, **kwargs):
        callback(_result({"weight": torch.tensor([3.0, 5.0])}, "site-1", params_type=ParamsType.DIFF))

    controller.send_model = send_model
    controller.run()

    assert type(controller._aggr_helper) is WeightedAggregationHelper
    saved = load_file(tmp_path / "app" / "FL_global_model.safetensors")
    assert torch.equal(saved["weight"], torch.tensor([4.0, 8.0]))


def test_pt_file_persistor_state_accepts_lazy_aggregation_output(tmp_path):
    save_file({"w": torch.tensor([1.0, 2.0])}, tmp_path / "aggregate.safetensors")
    manager = PTModelPersistenceFormatManager({"w": torch.zeros(2)}, allow_numpy_conversion=False)

    manager.update(make_model_learnable(safetensors_refs(str(tmp_path / "aggregate.safetensors")), {}))

    assert torch.equal(manager.var_dict["w"], torch.tensor([1.0, 2.0]))
    assert isinstance(manager.to_persistence_dict()["model"]["w"], torch.Tensor)


def test_pt_recipe_requires_pytorch_exchange_format_for_the_safetensors_persistor(tmp_path):
    from nvflare.app_opt.pt.recipes.fedavg import FedAvgRecipe
    from nvflare.client.config import ExchangeFormat

    persistor = PTSafetensorsModelPersistor(str(tmp_path / "model.safetensors"))

    with pytest.raises(ValueError, match="ExchangeFormat.PYTORCH"):
        FedAvgRecipe(name="lazy", min_clients=1, train_script="train.py", model_persistor=persistor)

    recipe = FedAvgRecipe(
        name="lazy",
        min_clients=1,
        train_script="train.py",
        model_persistor=persistor,
        server_expected_format=ExchangeFormat.PYTORCH,
        enable_tensor_disk_offload=True,
    )
    server_app = recipe._job._deploy_map[SERVER_SITE_NAME]
    assert server_app.app_config.workflows[0].controller.enable_tensor_disk_offload is True
    assert server_app.app_config.components["persistor"] is persistor
