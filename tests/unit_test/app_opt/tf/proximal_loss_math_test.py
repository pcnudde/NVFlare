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

import pytest

tf = pytest.importorskip("tensorflow")

from nvflare.app_opt.tf.fedprox_loss import TFFedProxLoss  # noqa: E402


def test_fedprox_loss_adds_global_model_distance_and_has_correct_local_gradient():
    local = [tf.Variable([2.0, 5.0]), tf.Variable([[3.0]])]
    global_weights = [tf.constant([1.0, 1.0]), tf.constant([[1.0]])]
    loss = TFFedProxLoss(local, global_weights, mu=0.2, loss_fun=tf.keras.losses.MeanSquaredError())
    assert float(loss.model_difference_norm(global_weights, local)) == pytest.approx(21)
    with tf.GradientTape() as tape:
        result = loss(tf.constant([1.0, 3.0]), tf.constant([2.0, 1.0]))
    assert float(result) == pytest.approx(2.5 + 0.1 * 21)
    gradients = tape.gradient(result, local)
    assert gradients[0].numpy().tolist() == pytest.approx([0.2, 0.8])
    assert float(gradients[1][0, 0]) == pytest.approx(0.4)
    baseline = TFFedProxLoss(local, global_weights, mu=0, loss_fun=tf.keras.losses.MeanSquaredError())
    assert float(baseline([1.0, 3.0], [2.0, 1.0])) == pytest.approx(2.5)
