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

torch = pytest.importorskip("torch")

from nvflare.apis.fl_context import FLContext
from nvflare.app_opt.pt.ditto import PTDittoHelper
from nvflare.app_opt.pt.model_reader_writer import PTModelReaderWriter


def _helper(tmp_path, **kwargs):
    model = torch.nn.Linear(2, 1, bias=False)
    options = dict(
        criterion=torch.nn.MSELoss(),
        model=model,
        optimizer=torch.optim.SGD(model.parameters(), lr=0.1),
        device=torch.device("cpu"),
        app_dir=str(tmp_path),
    )
    options.update(kwargs)
    return PTDittoHelper(**options)


@pytest.mark.parametrize("field", ["criterion", "model", "optimizer", "device"])
def test_ditto_rejects_non_torch_training_components(tmp_path, field):
    with pytest.raises(ValueError, match=field + " component must be torch"):
        _helper(tmp_path, **{field: object()})


def test_ditto_saves_latest_and_best_independently_and_restores_personalized_state(tmp_path):
    helper = _helper(tmp_path)
    assert helper.device == torch.device("cpu")
    initial = {"weight": torch.tensor([[1.0, 2.0]])}
    helper.load_model(initial)
    assert torch.equal(helper.model.weight, initial["weight"])
    helper.epoch_global = 3
    helper.update_metric_save_model(0.7)
    with torch.no_grad():
        helper.model.weight.fill_(5)
    helper.epoch_global = 4
    helper.update_metric_save_model(0.6)
    latest = torch.load(helper.model_file_path, weights_only=True)
    best = torch.load(helper.best_model_file_path, weights_only=True)
    assert latest["epoch"] == 4
    assert torch.equal(latest["model"]["weight"], torch.tensor([[5.0, 5.0]]))
    assert best["epoch"] == 3
    assert best["best_metric"] == 0.7
    assert torch.equal(best["model"]["weight"], initial["weight"])
    restored = _helper(tmp_path)
    restored.load_model({"weight": torch.zeros(1, 2)})
    assert restored.epoch_of_start_time == 4
    assert restored.best_metric == 0.7
    assert torch.equal(restored.model.weight, latest["model"]["weight"])


@pytest.mark.parametrize("wrapped", [True, False])
def test_model_reader_writer_extracts_and_applies_tiny_real_model(wrapped):
    model = torch.nn.Linear(2, 1, bias=False)
    network = SimpleNamespace(module=model) if wrapped else model
    reader = PTModelReaderWriter()
    params = {"weight": np.array([[3.0, 4.0]], dtype=np.float32)}
    operations = reader.apply_model(network, wrapped, params, FLContext())
    assert len(operations) == 1
    extracted = reader.extract_model(network, wrapped, {}, FLContext())
    np.testing.assert_array_equal(extracted["weight"], params["weight"])
    assert torch.equal(model.weight, torch.tensor([[3.0, 4.0]]))


def test_model_reader_writer_reports_conversion_and_loading_failures():
    reader = PTModelReaderWriter()
    model = MagicMock()
    model.state_dict.return_value = {"weight": SimpleNamespace(cpu=MagicMock(side_effect=ValueError("bad tensor")))}
    with pytest.raises(ValueError, match="bad tensor"):
        reader.extract_model(model, False, {}, FLContext())
    model = torch.nn.Linear(2, 1, bias=False)
    with pytest.raises(RuntimeError, match="load_state_dict Exception"):
        reader.apply_model(model, False, {"weight": np.zeros((3, 3))}, FLContext())
