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

from nvflare.apis.event_type import EventType
from nvflare.apis.fl_context import FLContext
from nvflare.app_common.abstract.model import ModelLearnableKey
from nvflare.app_opt.tf.model_persistor import TFModelPersistor


def _model():
    model = tf.keras.Sequential([tf.keras.Input((2,)), tf.keras.layers.Dense(1, use_bias=False, name="dense")])
    model.set_weights([np.array([[1.0], [2.0]], dtype=np.float32)])
    return model


def _persistor(tmp_path, model=None, **kwargs):
    subject = TFModelPersistor(model=model, **kwargs)
    context = FLContext()
    engine = MagicMock()
    engine.get_workspace.return_value.get_app_dir.return_value = str(tmp_path)
    context.get_engine = lambda: engine
    subject.handle_event(EventType.START_RUN, context)
    subject.system_panic = MagicMock()
    return subject, context


def test_persistor_roundtrip_and_inventory_preserve_layer_arrays(tmp_path):
    subject, context = _persistor(tmp_path, _model())
    learnable = subject.load_model(context)
    assert list(learnable[ModelLearnableKey.WEIGHTS]) == ["dense_nvf_0"]
    learnable[ModelLearnableKey.WEIGHTS]["dense_nvf_0"] = np.array([[3.0], [4.0]], dtype=np.float32)
    subject.save_model(learnable, context)
    inventory = subject.get_model_inventory(context)
    assert inventory[subject.save_name].location == str(tmp_path / subject.save_name)
    assert inventory[subject.save_name].model_format == "TensorFlow"
    subject.model.set_weights([np.zeros((2, 1), dtype=np.float32)])
    loaded = subject.get_model(subject.save_name, context)
    np.testing.assert_array_equal(loaded[ModelLearnableKey.WEIGHTS]["dense_nvf_0"], [[3], [4]])
    assert subject.get_model("missing.weights.h5", context) is None
    assert subject._get_model_from_location(str(tmp_path / "missing.weights.h5"), context) is None
    assert subject._get_model_from_location(str(tmp_path), context) is None
    reloaded, second_context = _persistor(tmp_path, _model())
    np.testing.assert_array_equal(
        reloaded.load_model(second_context)[ModelLearnableKey.WEIGHTS]["dense_nvf_0"], [[3], [4]]
    )


@pytest.mark.parametrize("full_model", [True, False])
def test_explicit_checkpoint_takes_priority_over_existing_saved_weights(tmp_path, full_model):
    checkpoint = tmp_path / ("initial.keras" if full_model else "initial.weights.h5")
    model = _model()
    if full_model:
        model.save(checkpoint)
    else:
        model.save_weights(checkpoint)
    model.set_weights([np.ones((2, 1), dtype=np.float32) * 9])
    model.save_weights(tmp_path / "tf_model.weights.h5")
    subject, context = _persistor(tmp_path, None if full_model else model, source_ckpt_file_full_name=str(checkpoint))
    loaded = subject.load_model(context)
    np.testing.assert_array_equal(loaded[ModelLearnableKey.WEIGHTS]["dense_nvf_0"], [[1], [2]])
    inventory = subject.get_model_inventory(context)
    assert inventory[checkpoint.name].props == {"source": "initial_ckpt"}
    assert len(inventory) == 2


@pytest.mark.parametrize(
    "case", ["missing_checkpoint", "no_model", "weights_without_model", "saved_without_model", "h5_without_model"]
)
def test_persistor_rejects_unavailable_model_and_checkpoints(tmp_path, case):
    kwargs = {}
    if case == "missing_checkpoint":
        kwargs["source_ckpt_file_full_name"] = str(tmp_path / "missing")
    elif case in ("weights_without_model", "h5_without_model"):
        path = tmp_path / ("source.bin" if case == "weights_without_model" else "source.h5")
        path.write_bytes(b"not-a-model")
        kwargs["source_ckpt_file_full_name"] = str(path)
    elif case == "saved_without_model":
        (tmp_path / "tf_model.weights.h5").write_bytes(b"weights")
    subject, context = _persistor(tmp_path, **kwargs)
    assert subject.load_model(context) is None
    assert subject.system_panic.call_count == 1
    reason = subject.system_panic.call_args.kwargs["reason"]
    assert "model" in reason.lower() or "checkpoint" in reason.lower()


@pytest.mark.parametrize(
    "config, result, expected",
    [
        ({}, None, "path"),
        ({"path": "bad.Class"}, ValueError("bad class"), "Failed to instantiate"),
        ({"path": "object"}, object(), "tf.keras.Model"),
    ],
)
def test_dict_model_configuration_checks_path_constructor_and_framework(
    monkeypatch, tmp_path, config, result, expected
):
    from nvflare.fuel.utils import class_utils

    factory = MagicMock(side_effect=result if isinstance(result, Exception) else None, return_value=result)
    monkeypatch.setattr(class_utils, "instantiate_class", factory)
    subject, context = _persistor(tmp_path, config)
    assert subject.load_model(context) is None
    assert expected in subject.system_panic.call_args.kwargs["reason"]
    assert factory.call_count == int(bool(config))


def test_dict_model_and_delayed_build_use_runtime_shape(monkeypatch, tmp_path):
    from nvflare.fuel.utils import class_utils

    model = tf.keras.Sequential([tf.keras.layers.Dense(1, use_bias=False, name="delayed")])
    model._input_shape = (None, 2)
    factory = MagicMock(return_value=model)
    monkeypatch.setattr(class_utils, "instantiate_class", factory)
    subject, context = _persistor(tmp_path, {"path": "project.Model", "args": {"width": 2}})
    loaded = subject.load_model(context)
    assert model.built
    assert loaded[ModelLearnableKey.WEIGHTS]["delayed_nvf_0"].shape == (2, 1)
    factory.assert_called_once_with("project.Model", {"width": 2})


def test_delayed_build_without_shape_fails_explicitly(tmp_path):
    model = tf.keras.Sequential([tf.keras.layers.Dense(1)])
    subject, context = _persistor(tmp_path, model)
    assert subject.load_model(context) is None
    assert "_input_shape" in subject.system_panic.call_args.kwargs["reason"]
    assert subject.get_model("anything", context) is None


@pytest.mark.parametrize("directory", [True, False])
def test_other_checkpoint_formats_delegate_to_model_loading_boundary(monkeypatch, tmp_path, directory):
    model = _model()
    path = tmp_path / "checkpoint"
    if directory:
        path.mkdir()
    else:
        path.write_bytes(b"weights")
    load = MagicMock(return_value=model)
    monkeypatch.setattr(tf.keras.models, "load_model", load)
    weights = MagicMock()
    monkeypatch.setattr(model, "load_weights", weights)
    subject, context = _persistor(tmp_path, model, source_ckpt_file_full_name=str(path))
    loaded = subject.load_model(context)
    assert "dense_nvf_0" in loaded[ModelLearnableKey.WEIGHTS]
    if directory:
        load.assert_called_once_with(str(path))
        weights.assert_not_called()
    else:
        weights.assert_called_once_with(str(path))
        load.assert_not_called()
