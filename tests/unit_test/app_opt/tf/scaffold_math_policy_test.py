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

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")

from nvflare.app_opt.tf.scaffold import ScaffoldCallback, TFScaffoldHelper, get_lr_values


def _model():
    model = tf.keras.Sequential(
        [
            tf.keras.Input((1,)),
            tf.keras.layers.Dense(1, use_bias=False, name="learned"),
            tf.keras.layers.Dense(1, use_bias=False, trainable=False, name="frozen"),
        ]
    )
    model.set_weights([np.array([[2]], dtype=np.float32), np.array([[3]], dtype=np.float32)])
    return model


def test_scaffold_corrects_only_trainable_weights_and_computes_control_delta():
    model = _model()
    global_model = _model()
    helper = TFScaffoldHelper()
    with pytest.raises(ValueError, match="hasn't been computed"):
        helper.get_delta_controls()
    helper.init(model)
    for weights in helper.c_local.get_weights() + helper.c_global.get_weights():
        np.testing.assert_array_equal(weights, [[0]])
    helper.load_global_controls(
        {"learned_nvf_0": np.array([[0.5]], dtype=np.float32), "frozen_nvf_0": np.array([[0.9]], dtype=np.float32)}
    )
    global_control, local_control = helper.get_params()
    helper.model_update(model, 0.2, global_control, local_control)
    np.testing.assert_allclose(model.get_weights()[0], [[1.9]])
    np.testing.assert_array_equal(model.get_weights()[1], [[3]])
    assert helper.cnt == 1
    model.layers[0].set_weights([np.array([[1.7]], dtype=np.float32)])
    helper.terms_update(model, 0.2, global_control, local_control, global_model)
    controls = helper.get_delta_controls()
    np.testing.assert_allclose(controls["learned_nvf_0"], [[1.0]], atol=1e-6)
    np.testing.assert_array_equal(controls["frozen_nvf_0"], [[3.0]])
    np.testing.assert_allclose(helper.c_local.get_weights()[0], [[1.0]], atol=1e-6)
    helper.get_params()
    assert helper.cnt == 0


def test_scaffold_callback_uses_optimizer_learning_rate_and_epoch_correction(capsys):
    model = _model()
    model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.25), loss="mse")
    helper = TFScaffoldHelper()
    helper.init(model)
    helper.load_global_controls(
        {"learned_nvf_0": np.array([[0.5]], dtype=np.float32), "frozen_nvf_0": np.array([[0]], dtype=np.float32)}
    )
    callback = ScaffoldCallback(helper)
    callback.set_model(model)
    callback.on_epoch_end(2)
    np.testing.assert_allclose(model.get_weights()[0], [[1.875]])
    assert helper.cnt == 1
    assert get_lr_values(model.optimizer) is model.optimizer.learning_rate
    assert "epoch 3" in capsys.readouterr().out
