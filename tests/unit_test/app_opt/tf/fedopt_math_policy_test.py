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

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")

from nvflare.app_common.abstract.fl_model import FLModel, ParamsType
from nvflare.app_opt.tf import fedopt_ctl as module


@pytest.mark.parametrize("stage", ["optimizer", "scheduler", "none"])
def test_tf_fedopt_wires_schedule_before_parent_workflow(monkeypatch, stage):
    subject = module.FedOpt(
        num_rounds=5,
        optimizer_args={"path": "tensorflow.keras.optimizers.SGD"},
        lr_scheduler_args={"path": "schedule", "args": {"decay_steps": None}},
    )
    assert subject.lr_scheduler_args["args"]["decay_steps"] == 5
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.5)
    schedule = tf.keras.optimizers.schedules.CosineDecay(initial_learning_rate=0.5, decay_steps=5)
    subject.build_component = MagicMock(
        side_effect=(
            ValueError("bad optimizer")
            if stage == "optimizer"
            else [optimizer, ValueError("bad scheduler") if stage == "scheduler" else schedule]
        )
    )
    subject.panic = MagicMock()
    subject.exception = MagicMock()
    parent = MagicMock()
    monkeypatch.setattr(module.FedAvg, "run", parent)
    subject.run()
    if stage == "none":
        parent.assert_called_once_with()
        assert subject.optimizer is optimizer
        assert subject.lr_scheduler is schedule
        assert subject.optimizer_args["args"] == {}
    else:
        parent.assert_not_called()
        assert subject.panic.call_count == 1
        assert ("optimizer" if stage == "optimizer" else "lr_scheduler") in subject.panic.call_args.args[0]


@pytest.mark.parametrize("full", [True, False])
def test_tf_fedopt_applies_negative_delta_gradient_and_averages_non_trainable_weights(full):
    model = tf.keras.Sequential(
        [
            tf.keras.Input((1,)),
            tf.keras.layers.Dense(1, use_bias=False, name="learned"),
            tf.keras.layers.Dense(1, use_bias=False, trainable=False, name="frozen"),
        ]
    )
    model.set_weights([np.array([[1.0]], dtype=np.float32), np.array([[4.0]], dtype=np.float32)])
    subject = module.FedOpt(
        optimizer_args={"path": "SGD"}, lr_scheduler_args={"path": "schedule", "args": {"decay_steps": 1}}
    )
    subject.persistor = SimpleNamespace(model=model)
    subject.optimizer = tf.keras.optimizers.SGD(learning_rate=0.5)
    subject.info = MagicMock()
    original = FLModel(
        params={
            "learned_nvf_0": np.array([[1.0]], dtype=np.float32),
            "frozen_nvf_0": np.array([[4.0]], dtype=np.float32),
        }
    )
    update = FLModel(
        params={
            "learned_nvf_0": np.array([[3.0 if full else 2.0]], dtype=np.float32),
            "frozen_nvf_0": np.array([[7.0 if full else 3.0]], dtype=np.float32),
        },
        params_type=ParamsType.FULL if full else ParamsType.DIFF,
        meta={"round": 1},
        metrics={"loss": 0.2},
    )
    result = subject.update_model(original, update)
    assert result is original
    np.testing.assert_allclose(result.params["learned_nvf_0"], [[2.0]])
    np.testing.assert_allclose(result.params["frozen_nvf_0"], [[7.0]])
    assert result.meta == update.meta
    assert result.metrics == update.metrics
    positive = subject._to_tf_params_list({"w": np.array([2.0])})
    np.testing.assert_array_equal(positive[0].numpy(), [2.0])
