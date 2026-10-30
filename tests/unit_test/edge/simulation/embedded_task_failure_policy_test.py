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
from unittest.mock import Mock

import pytest

pytest.importorskip("torch")

from nvflare.apis.dxo import DXO, DataKind
from nvflare.edge.model_protocol import ModelBufferType, ModelEncoding, ModelExchangeFormat, ModelNativeFormat
from nvflare.edge.simulation import et_task_processor as subject
from nvflare.edge.web.models.task_response import TaskResponse


class Processor(subject.ETTaskProcessor):
    def create_dataset(self, path):
        raise AssertionError("no dataset I/O is expected for dispatch-policy tests")


def task():
    data = DXO(
        DataKind.APP_DEFINED,
        base64.b64encode(b"model-bytes").decode(),
        meta={
            ModelExchangeFormat.MODEL_BUFFER_TYPE: ModelBufferType.EXECUTORCH,
            ModelExchangeFormat.MODEL_BUFFER_NATIVE_FORMAT: ModelNativeFormat.BINARY,
            ModelExchangeFormat.MODEL_BUFFER_ENCODING: ModelEncoding.BASE64,
        },
    )
    return TaskResponse("OK", task_id="task", task_name="train", task_data=data.to_dict())


@pytest.mark.parametrize("stage,import_error", [("load", False), ("load", True), ("train", False), ("train", True)])
def test_embedded_dispatch_preserves_missing_backend_and_wraps_runtime_failure(monkeypatch, stage, import_error):
    processor, loader = Processor("unused"), Mock()
    training = Mock()
    monkeypatch.setattr(subject, "_load_for_executorch_for_training_from_buffer", loader)
    monkeypatch.setattr(processor, "run_training", training)
    error = ImportError("backend absent") if import_error else ValueError("backend failed")
    (loader if stage == "load" else training).side_effect = error
    with pytest.raises(ImportError if import_error else RuntimeError) as raised:
        processor.process_task(task())
    if import_error:
        assert raised.value is error
    else:
        assert raised.value.__cause__ is error
    loader.assert_called_once_with(b"model-bytes")
    assert training.call_count == int(stage == "train")


def test_embedded_dispatch_validates_task_type_before_backend_and_preserves_model_metadata(monkeypatch):
    processor, loader = Processor("unused", {"epoch": 3}), Mock()
    monkeypatch.setattr(subject, "_load_for_executorch_for_training_from_buffer", loader)
    train = Mock(return_value={"net.weight": {"data": [0.5], "sizes": [1]}})
    monkeypatch.setattr(processor, "run_training", train)
    request = task()
    request.task_name = "evaluate"
    with pytest.raises(ValueError, match="Unsupported task type"):
        processor.process_task(request)
    loader.assert_not_called()
    request.task_name = "train"
    result = processor.process_task(request)
    assert result["kind"] == "et_tensor_diff" and result["data"] == train.return_value
    assert result["meta"] == request.task_data["meta"]
    train.assert_called_once_with(loader.return_value, total_epochs=3)
