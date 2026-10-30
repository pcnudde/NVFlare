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

torch = pytest.importorskip("torch")

from nvflare.apis.dxo import DXO, DataKind, MetaKey
from nvflare.apis.event_type import EventType
from nvflare.apis.fl_context import FLContext
from nvflare.app_common.abstract.fl_model import FLModel
from nvflare.app_opt.pt import fedopt as generator_module
from nvflare.app_opt.pt import fedopt_ctl as controller_module


@pytest.mark.parametrize("model", [None, object()])
def test_fedopt_generator_rejects_missing_or_non_module_source(model):
    subject = generator_module.PTFedOptModelShareableGenerator(source_model="source", device="cpu")
    subject.system_panic = MagicMock()
    context = FLContext()
    context.get_engine = lambda: SimpleEngine(model)
    subject.handle_event(EventType.START_RUN, context)
    assert subject.system_panic.call_count == 1
    assert subject.optimizer is None


class SimpleEngine:
    def __init__(self, model):
        self.model = model

    def get_component(self, name):
        assert name == "source"
        return self.model


@pytest.mark.parametrize("stage", ["optimizer", "scheduler"])
def test_fedopt_generator_reports_component_build_failure(stage):
    model = torch.nn.Linear(1, 1)
    subject = generator_module.PTFedOptModelShareableGenerator(
        source_model=model, optimizer_args={"name": "SGD"}, lr_scheduler_args={"name": "schedule"}, device="cpu"
    )
    engine = MagicMock()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    engine.build_component.side_effect = (
        ValueError("bad optimizer") if stage == "optimizer" else [optimizer, ValueError("bad scheduler")]
    )
    context = FLContext()
    context.get_engine = lambda: engine
    subject.system_panic = MagicMock()
    subject.handle_event(EventType.START_RUN, context)
    assert stage in subject.system_panic.call_args.args[0]
    assert engine.build_component.call_count == (1 if stage == "optimizer" else 2)
    if stage == "scheduler":
        assert subject.lr_scheduler_args["args"]["optimizer"] is optimizer
        assert subject.optimizer_name == "SGD"
        assert subject.lr_scheduler_name == "schedule"


@pytest.mark.parametrize(
    "kind, processed, expected",
    [(DataKind.WEIGHTS, None, "data_kind"), (DataKind.WEIGHT_DIFF, "encrypted", "processed by")],
)
def test_fedopt_generator_rejects_full_or_preprocessed_updates(kind, processed, expected):
    subject = generator_module.PTFedOptModelShareableGenerator(device="cpu")
    subject.system_panic = MagicMock()
    dxo = DXO(kind, data={"weight": np.ones(1)})
    if processed:
        dxo.set_meta_prop(MetaKey.PROCESSED_ALGORITHM, processed)
    result = subject.shareable_to_learnable(dxo.to_shareable(), FLContext())
    assert result == {}
    assert expected in subject.system_panic.call_args.args[0]


@pytest.mark.parametrize("source", [None, object()])
def test_fedopt_controller_rejects_missing_or_non_module_source(source):
    subject = controller_module.FedOpt(source_model="source", device="cpu")
    subject.get_component = MagicMock(return_value=source)
    subject.panic = MagicMock()
    subject.run()
    subject.get_component.assert_called_once_with("source")
    assert subject.panic.call_count == 1


@pytest.mark.parametrize("stage", ["optimizer", "lr_scheduler", "none"])
def test_fedopt_controller_wires_components_before_parent_workflow(monkeypatch, stage):
    model = torch.nn.Linear(1, 1, bias=False)
    subject = controller_module.FedOpt(
        source_model=model,
        device="cpu",
        optimizer_args={"path": "torch.optim.SGD"},
        lr_scheduler_args={"path": "torch.optim.lr_scheduler.StepLR"},
    )
    optimizer = torch.optim.SGD(model.parameters(), lr=0.25)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
    subject.build_component = MagicMock(
        side_effect=(
            ValueError("optimizer failure")
            if stage == "optimizer"
            else [optimizer, ValueError("scheduler failure") if stage == "lr_scheduler" else scheduler]
        )
    )
    subject.panic = MagicMock()
    subject.exception = MagicMock()
    parent_run = MagicMock()
    monkeypatch.setattr(controller_module.FedAvg, "run", parent_run)
    subject.run()
    if stage == "none":
        parent_run.assert_called_once_with()
        assert subject.optimizer is optimizer
        assert subject.lr_scheduler is scheduler
        assert subject.lr_scheduler_args["args"]["optimizer"] is optimizer
        assert list(subject.optimizer_args["args"]["params"])[0] is model.weight
    else:
        parent_run.assert_not_called()
        assert subject.panic.call_count == 1
        assert stage in subject.panic.call_args.args[0]


def test_fedopt_controller_updates_trainable_weights_and_averages_buffers():
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor([1.0]))
            self.register_buffer("running_mean", torch.tensor([4.0]))

    model = Model()
    subject = controller_module.FedOpt(source_model=model, device="cpu")
    subject.torch_model = model
    subject.optimizer = torch.optim.SGD(model.parameters(), lr=0.5)
    subject.lr_scheduler = torch.optim.lr_scheduler.StepLR(subject.optimizer, step_size=1, gamma=0.5)
    subject.info = MagicMock()
    original = FLModel(params={"weight": np.array([1.0]), "running_mean": np.array([4.0])})
    update = FLModel(
        params={"weight": np.array([2.0], dtype=np.float32), "running_mean": np.array([3.0], dtype=np.float32)},
        meta={"round": 3},
        metrics={"loss": 0.1},
    )
    result = subject.update_model(original, update)
    assert result is original
    np.testing.assert_allclose(result.params["weight"], [2.0])
    np.testing.assert_allclose(result.params["running_mean"], [7.0])
    assert result.meta == update.meta
    assert result.metrics == update.metrics
    assert subject.optimizer.param_groups[0]["lr"] == 0.25
    assert "fedopt layers: 1, fedavg layers: 1" in subject.info.call_args.args[0]
