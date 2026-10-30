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

from unittest.mock import Mock

import numpy as np
import pytest
import torch

from nvflare.apis.event_type import EventType
from nvflare.apis.fl_constant import FLContextKey
from nvflare.apis.fl_context import FLContext
from nvflare.app_common.app_constant import AppConstants
from nvflare.edge.widgets import evaluator as subject
from nvflare.edge.widgets.evaluator import GlobalEvaluator


@pytest.fixture
def evaluator(monkeypatch):
    monkeypatch.setattr(subject, "ThreadPoolExecutor", Mock(return_value=Mock()))
    monkeypatch.setattr(subject, "DEVICE", torch.device("cpu"))
    return GlobalEvaluator(torch.nn.Identity(), custom_dataset={"data": [[5, 0], [0, 5], [0, 5]], "labels": [0, 1, 0]})


@pytest.mark.parametrize(
    "kwargs,error",
    [
        ({}, "Must provide"),
        ({"custom_dataset": {}, "torchvision_dataset": {}}, "Cannot provide both"),
        ({"custom_dataset": {}, "model_path": 7}, "model_path must"),
        ({"custom_dataset": {}, "model_path": {}}, "must contain 'path'"),
        ({"custom_dataset": {}, "eval_frequency": 0}, "eval_frequency"),
        ({"custom_dataset": {}, "eval_frequency": "1"}, "eval_frequency"),
    ],
)
def test_rejects_ambiguous_or_invalid_evaluation_configuration(kwargs, error):
    kwargs = {"model_path": torch.nn.Identity(), **kwargs}
    with pytest.raises((TypeError, ValueError), match=error):
        GlobalEvaluator(**kwargs)


def test_custom_dataset_metrics_use_all_examples_and_switch_model_to_eval(evaluator):
    evaluator._create_data_loader()
    evaluator.model = evaluator.model_path
    metrics = evaluator._eval_model()
    assert metrics["accuracy"] == pytest.approx(200 / 3)
    assert evaluator.batch_size == 3
    assert evaluator.model.training is False
    data, labels = next(iter(evaluator.data_loader))
    assert data.dtype == torch.float32 and labels.dtype == torch.long
    assert data.shape == (3, 2) and labels.tolist() == [0, 1, 0]


def test_mismatched_custom_data_and_missing_loader_fail_safely(evaluator):
    assert evaluator._eval_model() == {"accuracy": 0.0}
    evaluator.custom_dataset = {"data": [[1]], "labels": [0, 1]}
    with pytest.raises(ValueError, match="same length"):
        evaluator._create_data_loader()
    evaluator.model = torch.nn.Identity()
    evaluator.data_loader = []
    assert evaluator._eval_model() == {"accuracy": 0.0}


def test_tensor_conversion_keeps_tensor_and_numpy_storage_and_handles_lists(evaluator):
    tensor = torch.tensor([2.0])
    assert evaluator._to_tensor(tensor) is tensor
    array = np.array([3.0], dtype=np.float32)
    converted = evaluator._to_tensor(array)
    array[0] = 9
    assert converted.item() == 9
    assert evaluator._to_tensor([1, 2]).tolist() == [1, 2]


@pytest.mark.parametrize(
    "config", ["torch.nn.Identity", {"path": "torch.nn.Linear", "args": {"in_features": 2, "out_features": 1}}]
)
def test_model_class_paths_and_constructor_arguments_are_applied(evaluator, config):
    evaluator.model_path = config
    model = evaluator._instantiate_model(FLContext())
    assert isinstance(model, torch.nn.Identity if isinstance(config, str) else torch.nn.Linear)
    if isinstance(model, torch.nn.Linear):
        assert model.in_features == 2 and model.out_features == 1


@pytest.mark.parametrize("path", ["missing_module.Model", "torch.nn.MissingModel", "InvalidPath"])
def test_model_import_failure_panics_and_stops_initialization(evaluator, path):
    evaluator.system_panic = Mock()
    evaluator.model_path = path
    assert evaluator._instantiate_model(FLContext()) is None
    assert path in evaluator.system_panic.call_args.kwargs["reason"]
    evaluator._initialize(EventType.START_RUN, FLContext())
    assert evaluator.model is None and evaluator.data_loader is None
    assert "Failed to instantiate" in evaluator.system_panic.call_args.kwargs["reason"]


def test_initialization_uses_application_directory_and_end_run_closes_owned_resources(evaluator, monkeypatch):
    import torch.utils.tensorboard as tensorboard

    writer = Mock()
    factory = Mock(return_value=writer)
    monkeypatch.setattr(tensorboard, "SummaryWriter", factory)
    ctx = FLContext()
    ctx.set_prop(FLContextKey.APP_ROOT, "/app/job")
    evaluator._initialize(EventType.START_RUN, ctx)
    assert evaluator._is_initialized()
    assert evaluator.model is evaluator.model_path
    factory.assert_called_once_with(log_dir="/app/job")
    evaluator._handle_end_run(EventType.END_RUN, ctx)
    evaluator._thread_pool.shutdown.assert_called_once_with(wait=True)
    writer.close.assert_called_once_with()


@pytest.mark.parametrize("configuration", [{}, {"name": "NoSuchDataset", "path": "/data"}])
def test_torchvision_dataset_config_rejects_missing_or_unknown_names(evaluator, configuration):
    evaluator.torchvision_dataset = configuration
    with pytest.raises(ValueError, match="torchvision"):
        evaluator._create_data_loader()


def test_torchvision_dataset_setup_preserves_download_transform_and_loader_options(evaluator, monkeypatch):
    dataset = [0, 1]
    factory = Mock(return_value=dataset)
    monkeypatch.setattr(subject.torchvision.datasets, "UnitDataset", factory, raising=False)
    loader = Mock()
    monkeypatch.setattr(subject.torch.utils.data, "DataLoader", loader)
    evaluator.torchvision_dataset = {"name": "UnitDataset", "path": "/data"}
    evaluator._create_data_loader()
    assert factory.call_args.kwargs["root"] == "/data"
    assert factory.call_args.kwargs["train"] is False and factory.call_args.kwargs["download"] is True
    assert isinstance(factory.call_args.kwargs["transform"], subject.transforms.Compose)
    loader.assert_called_once_with(dataset, batch_size=4, shuffle=False, num_workers=2)


@pytest.mark.parametrize(
    "weights,round_number,ready,submitted",
    [
        ({"w": [1]}, 2, True, True),
        ({"w": [1]}, 1, True, False),
        ({}, 2, True, False),
        (None, 2, True, False),
        ({"w": [1]}, 2, False, False),
    ],
)
def test_evaluation_submits_only_initialized_matching_rounds_with_weights(
    evaluator, weights, round_number, ready, submitted
):
    evaluator.eval_frequency = 2
    if ready:
        evaluator.model, evaluator.data_loader, evaluator.tb_writer = torch.nn.Identity(), [], Mock()
    ctx = FLContext()
    ctx.set_prop(AppConstants.CURRENT_ROUND, round_number)
    ctx.set_prop(AppConstants.GLOBAL_MODEL, {"weights": weights} if weights is not None else None)
    evaluator.evaluate("updated", ctx)
    assert evaluator._thread_pool.submit.called is submitted
    if submitted:
        evaluator._thread_pool.submit.assert_called_once_with(evaluator._evaluate_async, weights, 2, "eval_round_2")


def test_evaluation_loads_numpy_weights_and_writes_zero_based_round_metrics(evaluator):
    evaluator.model = torch.nn.Linear(2, 1, bias=False)
    evaluator.tb_writer = Mock()
    evaluator._eval_model = Mock(return_value={"accuracy": 90.0, "loss": 0.25})
    evaluator._evaluate_async({"weight": np.array([[3, 4]], dtype=np.float32)}, 3, "round-3")
    torch.testing.assert_close(evaluator.model.weight, torch.tensor([[3.0, 4.0]]))
    assert evaluator.tb_writer.add_scalar.call_args_list == [(("accuracy", 90.0, 2),), (("loss", 0.25, 2),)]


@pytest.mark.parametrize("case", ["no_model", "no_weights", "bad_weights"])
def test_evaluation_failure_never_records_partial_metrics(evaluator, caplog, case):
    evaluator.model = None if case == "no_model" else torch.nn.Linear(2, 1)
    evaluator.tb_writer = Mock()
    weights = {} if case == "no_weights" else {"bad-key": [1]}
    evaluator._evaluate_async(weights, 2, "round-2")
    evaluator.tb_writer.add_scalar.assert_not_called()
    if case == "bad_weights":
        assert "Error during evaluation" in caplog.text
