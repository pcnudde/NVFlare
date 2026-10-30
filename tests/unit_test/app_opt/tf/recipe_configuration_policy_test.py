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

from unittest.mock import MagicMock

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")

from nvflare.apis.dxo import DataKind
from nvflare.apis.event_type import EventType
from nvflare.apis.fl_context import FLContext
from nvflare.app_common.abstract.model import make_model_learnable
from nvflare.app_opt.tf.file_model_locator import TFFileModelLocator
from nvflare.app_opt.tf.job_config.model import TFModel
from nvflare.app_opt.tf.model_persistor import TFModelPersistor
from nvflare.app_opt.tf.recipes.cyclic import CyclicRecipe
from nvflare.app_opt.tf.recipes.fedopt import FedOptRecipe
from nvflare.app_opt.tf.recipes.scaffold import ScaffoldRecipe
from nvflare.client.config import ExchangeFormat, TransferType


@pytest.mark.parametrize("source", ["model", "dict", "checkpoint", "custom"])
def test_tf_model_wrapper_registers_correct_persistor_and_checkpoint(source):
    model = tf.keras.Sequential([tf.keras.Input((1,)), tf.keras.layers.Dense(1)])
    config = {"path": "project.Model", "args": {"width": 2}}
    custom = TFModelPersistor(model)
    wrapper = TFModel(
        model=model if source == "model" else config if source == "dict" else None,
        persistor=custom if source == "custom" else None,
        initial_ckpt="/server/source.keras",
    )
    job = MagicMock()
    context = object()
    wrapper.add_to_fed_job(job, context)
    kwargs = job.add_component.call_args.kwargs
    assert kwargs["comp_id"] == "persistor"
    assert kwargs["ctx"] is context
    persistor = kwargs["obj"]
    assert isinstance(persistor, TFModelPersistor)
    if source == "custom":
        assert persistor is custom
    else:
        assert persistor.source_ckpt_file_full_name == "/server/source.keras"
        assert persistor.model is (model if source == "model" else config if source == "dict" else None)
    if source == "dict":
        assert wrapper.model_class_path == "project.Model"
        assert wrapper.model_args == {"width": 2}


def test_tf_model_wrapper_rejects_missing_config_and_wrong_persistor_type():
    with pytest.raises(ValueError, match="path"):
        TFModel({"args": {}})
    with pytest.raises(ValueError, match="Unsupported model configuration"):
        TFModel().add_to_fed_job(MagicMock(), object())
    with pytest.raises(TypeError, match="persistor"):
        TFModel(persistor=object())


@pytest.mark.parametrize("source", ["missing_id", "missing_component", "wrong_type"])
def test_tf_locator_rejects_missing_or_wrong_framework_persistor(source):
    locator = TFFileModelLocator("" if source == "missing_id" else "persistor")
    engine = MagicMock()
    engine.get_component.return_value = None if source == "missing_component" else object()
    context = FLContext()
    context.get_engine = lambda: engine
    with pytest.raises(
        ValueError,
        match={
            "missing_id": "valid tf_persistor_id",
            "missing_component": "No component",
            "wrong_type": "TFModelPersistor",
        }[source],
    ):
        locator.handle_event(EventType.START_RUN, context)


def test_tf_locator_refreshes_inventory_and_converts_loaded_model_to_dxo():
    persistor = TFModelPersistor()
    persistor.get_model_inventory = MagicMock(return_value={"model.weights.h5": object()})
    persistor.get = MagicMock(return_value=make_model_learnable({"layer_nvf_0": np.array([2])}, {"round": 2}))
    engine = MagicMock()
    engine.get_component.return_value = persistor
    context = FLContext()
    context.get_engine = lambda: engine
    locator = TFFileModelLocator("persistor")
    locator.handle_event(EventType.START_RUN, context)
    assert locator.get_model_names(context) == ["model.weights.h5"]
    result = locator.locate_model("model.weights.h5", context)
    assert result.data_kind == DataKind.WEIGHTS
    np.testing.assert_array_equal(result.data["layer_nvf_0"], [2])
    assert result.meta == {"round": 2}
    persistor.get.assert_called_once_with("model.weights.h5", context)
    with pytest.raises(ValueError, match="inventory"):
        locator.locate_model("missing", context)


@pytest.mark.parametrize("recipe_class", [FedOptRecipe, ScaffoldRecipe])
@pytest.mark.parametrize("model_config", [False, True])
def test_tensorflow_recipes_wire_memory_exchange_and_framework_controller(recipe_class, model_config):
    model = (
        {"class_path": "project.Model", "args": {"width": 2}}
        if model_config
        else tf.keras.Sequential([tf.keras.Input((1,)), tf.keras.layers.Dense(1)])
    )
    kwargs = (
        {
            "optimizer_args": {"path": "tensorflow.keras.optimizers.SGD", "args": {"learning_rate": 0.4}},
            "lr_scheduler_args": {
                "path": "tensorflow.keras.optimizers.schedules.CosineDecay",
                "args": {"initial_learning_rate": 0.4, "decay_steps": 7},
            },
        }
        if recipe_class is FedOptRecipe
        else {}
    )
    recipe = recipe_class(
        model=model,
        min_clients=2,
        num_rounds=7,
        train_script=__file__,
        train_args="--epochs 2",
        server_memory_gc_rounds=3,
        client_memory_gc_rounds=4,
        params_transfer_type=TransferType.DIFF,
        **kwargs,
    )
    assert recipe.num_rounds == 7
    assert recipe.params_transfer_type == TransferType.DIFF
    assert recipe.server_expected_format == ExchangeFormat.NUMPY
    server = recipe._job._deploy_map["server"].app_config
    controllers = server.workflows
    assert len(controllers) == 1
    controller = controllers[0].controller
    assert controller.num_rounds == 7
    assert controller.memory_gc_rounds == 3
    if recipe_class is FedOptRecipe:
        assert controller.optimizer_args["args"]["learning_rate"] == 0.4
    if model_config:
        assert recipe.model == {"path": "project.Model", "args": {"width": 2}}


@pytest.mark.parametrize("mode", ["none", "raw", "wrapped", "checkpoint"])
def test_tf_cyclic_persistor_setup_supports_wrappers_and_server_checkpoint(mode):
    model = tf.keras.Sequential([tf.keras.Input((1,)), tf.keras.layers.Dense(1)])
    supplied = None if mode == "none" else TFModel(model) if mode == "wrapped" else model
    if mode == "none":
        with pytest.raises(ValueError, match="model or initial_ckpt"):
            CyclicRecipe(train_script=__file__, min_clients=2)
        return
    recipe = CyclicRecipe(
        model=supplied,
        initial_ckpt="/server/model.keras" if mode == "checkpoint" else None,
        train_script=__file__,
        min_clients=2,
        num_rounds=3,
    )
    server = recipe._job._deploy_map["server"].app_config
    assert isinstance(server.components["persistor"], TFModelPersistor)
    if mode == "checkpoint":
        assert server.components["persistor"].source_ckpt_file_full_name == "/server/model.keras"
    else:
        assert server.components["persistor"].model is model
