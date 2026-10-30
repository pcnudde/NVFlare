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
import torch
from torch.utils.data import DataLoader, TensorDataset

from nvflare.apis.dxo import DXO, DataKind, from_shareable
from nvflare.apis.event_type import EventType
from nvflare.apis.fl_constant import ReservedKey, ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.apis.signal import Signal
from nvflare.app_opt.p2p.executors import distributed_gradient_descent as dgd
from nvflare.app_opt.p2p.executors import gradient_tracking as gt
from nvflare.app_opt.p2p.types import Config, LocalConfig, Neighbor, Network, Node
from nvflare.app_opt.p2p.utils import utils
from nvflare.app_opt.p2p.utils.metrics import compute_loss_over_dataset


class _DGD(dgd.DGDExecutor):
    def __init__(self, model, loader):
        super().__init__(model=model, loss=torch.nn.MSELoss(), train_dataloader=loader, test_dataloader=loader)


class _GT(gt.GTExecutor):
    def __init__(self, model, loader):
        super().__init__(model=model, loss=torch.nn.MSELoss(), train_dataloader=loader, test_dataloader=loader)


@pytest.fixture
def model_and_loader():
    model = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(1.0)
    loader = DataLoader(TensorDataset(torch.tensor([[1.0], [2.0]]), torch.zeros(2, 1)), batch_size=1)
    return model, loader


@pytest.mark.parametrize("kind,expected_weight", [("dgd", 0.55), ("gt", 0.503125)])
def test_distributed_updates_follow_consensus_and_gradient_math_with_controlled_peer_boundary(
    monkeypatch, tmp_path, model_and_loader, kind, expected_weight
):
    monkeypatch.chdir(tmp_path)
    model, loader = model_and_loader
    module, cls = (dgd, _DGD) if kind == "dgd" else (gt, _GT)
    monkeypatch.setattr(module, "get_device", lambda: torch.device("cpu"))
    executor = cls(model, loader)
    ctx = FLContext()
    engine = MagicMock()
    ctx.set_prop(ReservedKey.ENGINE, engine)
    ctx.set_prop(ReservedKey.IDENTITY_NAME, "site-1")
    executor.handle_event(EventType.START_RUN, ctx)
    assert executor.id == 1
    assert engine.register_aux_message_handler.call_args.kwargs["topic"] == "send_value"
    config = DXO(DataKind.WEIGHTS, {"neighbors": [{"id": "site-2", "weight": 0.25}]}).to_shareable()
    assert executor.execute("config", config, ctx, Signal()).get_return_code() == ReturnCode.OK
    assert executor._weight == 0.75 and executor.neighbors[0].id == "site-2"
    exchanged = []

    def exchange(context, value, iteration):
        exchanged.append(iteration)
        neighbor = [torch.tensor([[2.0]])]
        if kind == "gt":
            neighbor = {"parameters": neighbor, "tracker": [torch.zeros(1, 1)]}
        executor.neighbors_values[iteration] = {"site-2": neighbor}

    executor._exchange_values = exchange
    task = DXO(DataKind.WEIGHTS, {"iterations": 3, "stepsize": 0.1}).to_shareable()
    assert executor.execute("run_algorithm", task, ctx, Signal()).get_return_code() == ReturnCode.OK
    assert exchanged == [0, 1, 2]
    assert model.weight.item() == pytest.approx(expected_weight, abs=1e-6)
    assert executor.neighbors_values == {}
    assert len(executor.train_loss_sequence) == len(executor.test_loss_sequence) == 2
    assert executor.train_loss_sequence[0][1] == pytest.approx(2.5)
    assert torch.load("train_loss_sequence.pt", weights_only=True).shape == (2, 2)
    assert torch.load("test_loss_sequence.pt", weights_only=True).shape == (2, 2)
    if kind == "gt":
        assert executor.old_gradient[0].item() == pytest.approx(1.00625, abs=1e-6)
        assert executor.tracker[0].item() == pytest.approx(-1.8375, abs=1e-6)
    assert executor.execute("unknown", Shareable(), ctx, Signal()).get_return_code() == ReturnCode.TASK_UNKNOWN


@pytest.mark.parametrize("kind", ["dgd", "gt"])
def test_algorithm_abort_does_not_exchange_or_modify_parameters(monkeypatch, model_and_loader, kind):
    model, loader = model_and_loader
    module, cls = (dgd, _DGD) if kind == "dgd" else (gt, _GT)
    monkeypatch.setattr(module, "get_device", lambda: torch.device("cpu"))
    executor = cls(model, loader)
    executor._iterations = 3
    executor._exchange_values = MagicMock()
    signal = Signal()
    signal.trigger(True)
    executor.run_algorithm(FLContext(), Shareable(), signal)
    executor._exchange_values.assert_not_called()
    assert model.weight.item() == 1.0


@pytest.mark.parametrize("kind", ["dgd", "gt"])
def test_tensor_message_round_trip_preserves_values(monkeypatch, model_and_loader, kind):
    model, loader = model_and_loader
    module, cls = (dgd, _DGD) if kind == "dgd" else (gt, _GT)
    monkeypatch.setattr(module, "get_device", lambda: torch.device("cpu"))
    executor = cls(model, loader)
    value = list(model.parameters())
    if kind == "gt":
        value = {"parameters": value, "tracker": [torch.tensor([[3.0]])]}
    with torch.no_grad():
        wire = executor._to_message(value)
    restored = executor._from_message(wire)
    if kind == "gt":
        np.testing.assert_array_equal(wire["tracker"][0], [[3.0]])
        assert restored["parameters"][0].item() == 1.0
        assert restored["tracker"][0].item() == 3.0
    else:
        assert restored[0].item() == 1.0


def test_sync_handler_records_peer_identity_and_exchange_timeout_panics(monkeypatch, model_and_loader):
    model, loader = model_and_loader
    monkeypatch.setattr(dgd, "get_device", lambda: torch.device("cpu"))
    executor = _DGD(model, loader)
    executor.neighbors = [Neighbor("site-2", 0.5)]
    ctx = FLContext()
    engine = MagicMock()
    ctx.set_prop(ReservedKey.ENGINE, engine)
    request = DXO(DataKind.WEIGHTS, {"iteration": 2, "value": [np.array([[4.0]], dtype=np.float32)]}).to_shareable()
    request.set_peer_props({ReservedKey.IDENTITY_NAME: "site-2"})
    reply = executor._handle_neighbor_value("send_value", request, ctx)
    assert reply.get_return_code() == ReturnCode.OK and executor.sync_waiter.is_set()
    assert executor.neighbors_values[2]["site-2"][0].item() == 4.0
    executor._exchange_values(ctx, [torch.tensor([[1.0]])], 2)
    sent = engine.send_aux_request.call_args.kwargs
    assert sent["targets"] == ["site-2"] and sent["topic"] == "send_value"
    assert from_shareable(sent["request"]).data["iteration"] == 2
    executor.sync_waiter = MagicMock()
    executor.sync_waiter.wait.return_value = False
    executor.system_panic = MagicMock()
    executor._exchange_values(ctx, [torch.tensor([[1.0]])], 3)
    executor.sync_waiter.wait.assert_called_once_with(timeout=10)
    executor.system_panic.assert_called_once()


def test_network_configuration_normalizes_nested_wire_dictionaries():
    config = Config(
        network={"nodes": [{"id": "a", "neighbors": [{"id": "b", "weight": 0.5}]}]}, extra={"iterations": 4}
    )
    assert isinstance(config.network, Network)
    assert isinstance(config.network.nodes[0], Node)
    assert isinstance(config.network.nodes[0].neighbors[0], Neighbor)
    assert config.__dict__ == {
        "extra": {"iterations": 4},
        "network": {"nodes": [{"id": "a", "neighbors": [{"id": "b", "weight": 0.5}]}]},
    }
    local = LocalConfig([Neighbor("a", 0.5), {"id": "b", "weight": 0.25}], extra={"x": 1})
    assert [n.weight for n in local.neighbors] == [0.5, 0.25]
    assert local.extra == {"x": 1}
    assert Network([Node("a")]).__dict__ == {"nodes": [{"id": "a", "neighbors": []}]}


@pytest.mark.parametrize("cuda,mps,expected", [(True, False, "cuda"), (False, True, "mps"), (False, False, "cpu")])
def test_device_selection_prefers_cuda_then_mps_then_cpu(monkeypatch, cuda, mps, expected):
    monkeypatch.setattr(utils.torch.cuda, "is_available", lambda: cuda)
    monkeypatch.setattr(utils.torch.backends.mps, "is_available", lambda: mps)
    assert utils.get_device().type == expected


def test_dataset_loss_is_sample_weighted_and_requires_inputs(model_and_loader):
    model, loader = model_and_loader
    assert compute_loss_over_dataset(model, torch.nn.MSELoss(), loader, "cpu") == pytest.approx(2.5)
    with pytest.raises(ValueError, match="All arguments"):
        compute_loss_over_dataset()
