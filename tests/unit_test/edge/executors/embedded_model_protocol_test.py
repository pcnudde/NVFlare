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

import base64
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

torch = pytest.importorskip("torch")

from nvflare.apis.dxo import DXO, DataKind
from nvflare.apis.fl_context import FLContext
from nvflare.edge.constants import CookieKey, MsgKey
from nvflare.edge.executors import et_edge_model_executor as subject
from nvflare.edge.model_protocol import ModelBufferType, ModelEncoding, ModelExchangeFormat, ModelNativeFormat
from nvflare.edge.models import model as model_module
from nvflare.edge.mud import BaseState
from nvflare.edge.web.models.device_info import DeviceInfo
from nvflare.edge.web.models.result_report import ResultReport


@pytest.fixture(autouse=True)
def preserve_torch_rng():
    with torch.random.fork_rng(devices=[]):
        yield


def test_embedded_model_payload_loads_weights_before_export_and_declares_encoding(monkeypatch):
    model = model_module.DeviceModel(torch.nn.Linear(2, 2))
    executor = subject.ETEdgeModelExecutor(model, (1, 2), (1,), "factory", 3)
    expected = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

    def export(net, input_shape, output_shape):
        assert net is model
        assert input_shape == (1, 2) and output_shape == (1,)
        torch.testing.assert_close(net.net.weight, expected)
        torch.testing.assert_close(net.net.bias, torch.tensor([0.5, 1.0]))
        return b"embedded-model-buffer"

    monkeypatch.setattr(subject, "export_model_to_bytes", export)
    state = BaseState(4, DXO(DataKind.WEIGHTS, {"weight": expected.numpy(), "bias": [0.5, 1.0]}), 0, None)
    result = executor._convert_task(state, SimpleNamespace(id="task"), FLContext())
    assert result["kind"] == DataKind.APP_DEFINED
    assert base64.b64decode(result["data"]) == b"embedded-model-buffer"
    assert result["meta"] == {
        MsgKey.MODEL_VERSION: 4,
        ModelExchangeFormat.MODEL_BUFFER_TYPE: ModelBufferType.EXECUTORCH,
        ModelExchangeFormat.MODEL_BUFFER_NATIVE_FORMAT: ModelNativeFormat.BINARY,
        ModelExchangeFormat.MODEL_BUFFER_ENCODING: ModelEncoding.BASE64,
    }


@pytest.mark.parametrize("cookie", [None, {}, {"unrelated": 1}])
def test_embedded_model_result_requires_model_version_before_tensor_conversion(cookie):
    executor = subject.ETEdgeModelExecutor(Mock(), (1, 2), (1,), "factory", 3)
    report = ResultReport(DeviceInfo("device"), None, "job", "task", cookie=cookie)
    with pytest.raises(ValueError, match="cookie"):
        executor._convert_device_result_to_model_update(report, SimpleNamespace(id="task"), FLContext())


def test_embedded_result_rejects_tensor_shape_mismatch():
    executor = subject.ETEdgeModelExecutor(Mock(), (1, 2), (1,), "factory", 3)
    report = ResultReport(
        DeviceInfo("device"),
        None,
        "job",
        "task",
        cookie={CookieKey.MODEL_VERSION: 2},
        result={"meta": {}, "data": {"net.weight": {"data": [1, 2], "sizes": [2, 2]}}},
    )
    with pytest.raises(RuntimeError, match="shape"):
        executor._convert_device_result_to_model_update(report, SimpleNamespace(id="task"), FLContext())


def test_model_wrapper_returns_differentiable_loss_and_detached_predictions():
    net = torch.nn.Linear(2, 2)
    wrapped = model_module.DeviceModel(net)
    data, labels = torch.tensor([[1.0, 2.0], [2.0, 1.0]]), torch.tensor([0, 1])
    loss, predictions = wrapped(data, labels)
    torch.testing.assert_close(loss, torch.nn.functional.cross_entropy(net(data), labels))
    torch.testing.assert_close(predictions, net(data).argmax(dim=1))
    assert loss.requires_grad and not predictions.requires_grad
    loss.backward()
    assert net.weight.grad is not None and torch.isfinite(net.weight.grad).all()


def test_export_bytes_builds_input_and_integer_label_examples(monkeypatch):
    model = Mock()
    export = Mock(return_value=SimpleNamespace(buffer=b"buffer"))
    monkeypatch.setattr(model_module, "export_model", export)
    assert model_module.export_model_to_bytes(model, (2, 3), (2,)) == b"buffer"
    net, example, label = export.call_args.args
    assert net is model and example.shape == (2, 3)
    assert label.shape == (2,) and label.dtype == torch.int64
    torch.testing.assert_close(label, torch.ones(2, dtype=torch.int64))
