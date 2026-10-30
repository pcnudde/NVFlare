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
from unittest.mock import Mock

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from nvflare.apis.event_type import EventType  # noqa: E402
from nvflare.apis.fl_context import FLContext  # noqa: E402
from nvflare.app_common.abstract.model import make_model_learnable  # noqa: E402
from nvflare.app_opt.pt import file_model_locator as locator  # noqa: E402
from nvflare.app_opt.pt.file_model_persistor import PTFileModelPersistor  # noqa: E402
from nvflare.app_opt.pt.he_model_reader_writer import HEPTModelReaderWriter  # noqa: E402


@pytest.fixture(autouse=True)
def preserve_cpu_rng():
    with torch.random.fork_rng(devices=[]):
        yield


@pytest.mark.parametrize("wrapped", [False, True])
def test_decrypted_flat_parameters_are_reshaped_and_applied_without_overwriting_unreceived_bias(wrapped):
    network = torch.nn.Linear(2, 2)
    with torch.no_grad():
        network.bias.fill_(9)
    data = {"weight": np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)}
    target = SimpleNamespace(module=network) if wrapped else network
    changed = HEPTModelReaderWriter().apply_model(target, wrapped, data, FLContext())
    assert len(changed) == 1
    np.testing.assert_array_equal(changed[0], [[1, 2], [3, 4]])
    np.testing.assert_array_equal(data["weight"], [[1, 2], [3, 4]])
    torch.testing.assert_close(network.weight, torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
    torch.testing.assert_close(network.bias, torch.tensor([9.0, 9.0]))


def test_decrypted_parameter_shape_failure_keeps_model_unchanged():
    network = torch.nn.Linear(2, 2)
    old = network.weight.detach().clone()
    with pytest.raises(RuntimeError, match="reshaping Exception"):
        HEPTModelReaderWriter().apply_model(network, False, {"weight": np.zeros(3)}, FLContext())
    torch.testing.assert_close(network.weight, old)


@pytest.mark.parametrize("component", [None, object()])
def test_locator_rejects_missing_or_wrong_persistor_before_registering_serializer(monkeypatch, component):
    monkeypatch.setattr(locator, "fobs", SimpleNamespace(register=Mock()))
    ctx = Mock(spec=FLContext)
    ctx.get_engine.return_value.get_component.return_value = component
    with pytest.raises(ValueError, match="No component found|must be PTFileModelPersistor"):
        locator.PTFileModelLocator("saved").handle_event(EventType.START_RUN, ctx)
    locator.fobs.register.assert_not_called()


def test_locator_returns_typed_learnable_from_refreshed_named_inventory(monkeypatch):
    register = Mock()
    monkeypatch.setattr(locator, "fobs", SimpleNamespace(register=register))
    persistor = Mock(spec=PTFileModelPersistor)
    persistor.get_model_inventory.return_value = {"best": {"path": "saved.pt"}}
    persistor.get.return_value = make_model_learnable({"weight": np.array([3.0, 4.0])}, {"round": 4})
    ctx = Mock(spec=FLContext)
    ctx.get_engine.return_value.get_component.return_value = persistor
    item = locator.PTFileModelLocator("saved")
    item.handle_event(EventType.START_RUN, ctx)
    assert item.get_model_names(ctx) == ["best"]
    result = item.locate_model("best", ctx)
    np.testing.assert_array_equal(result.data["weight"], [3, 4])
    assert result.meta == {"round": 4}
    persistor.get.assert_called_once_with("best", ctx)
    register.assert_called_once_with(locator.TensorDecomposer)
    with pytest.raises(ValueError, match="does not contain"):
        item.locate_model("absent", ctx)
