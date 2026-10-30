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

from unittest.mock import Mock

import numpy as np
import pytest

from nvflare.apis.dxo import DXO, DataKind, from_shareable
from nvflare.apis.fl_constant import ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.signal import Signal
from nvflare.app_common.app_constant import AppConstants
from nvflare.app_common.ccwf.comps import np_trainer as subject
from nvflare.app_common.np.constants import NPConstants
from nvflare.fuel.utils import deprecated


@pytest.fixture(autouse=True)
def isolate_deprecation_cache(monkeypatch):
    monkeypatch.setattr(deprecated, "_WARNED_DEPRECATION_MESSAGES", set(deprecated._WARNED_DEPRECATION_MESSAGES))


def test_numpy_trainer_rejects_non_numeric_delta():
    with pytest.raises(TypeError, match="delta"):
        subject.NPTrainer(delta="invalid")


def test_numpy_validation_dispatch_preserves_metric_envelope_and_bounds_fake_accuracy(monkeypatch):
    trainer = subject.NPTrainer()
    uniform = Mock(return_value=0.42)
    monkeypatch.setattr(subject.random, "uniform", uniform)
    model = DXO(DataKind.WEIGHTS, {NPConstants.NUMPY_KEY: np.array([1, 2])}).to_shareable()
    result = trainer.execute(AppConstants.TASK_VALIDATION, model, FLContext(), Signal())
    dxo = from_shareable(result)
    assert dxo.data_kind == DataKind.METRICS and dxo.data == {"val_accuracy": 0.42}
    uniform.assert_called_once_with(0.1, 1.0)


def test_aborted_numpy_train_cannot_modify_received_model():
    trainer = subject.NPTrainer()
    weights = np.array([1, 2])
    model = DXO(DataKind.WEIGHTS, {NPConstants.NUMPY_KEY: weights}).to_shareable()
    abort = Signal()
    abort.trigger(True)
    result = trainer.execute(AppConstants.TASK_TRAIN, model, FLContext(), abort)
    assert result.get_return_code() == ReturnCode.TASK_ABORTED
    np.testing.assert_array_equal(weights, [1, 2])
