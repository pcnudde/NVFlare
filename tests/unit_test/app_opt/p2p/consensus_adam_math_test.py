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

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from nvflare.apis.dxo import DXO, DataKind  # noqa: E402
from nvflare.apis.fl_context import FLContext  # noqa: E402
from nvflare.apis.signal import Signal  # noqa: E402
from nvflare.app_opt.p2p.executors import gradient_tracking as gt  # noqa: E402
from nvflare.app_opt.p2p.executors.consensus import ConsensusExecutor  # noqa: E402
from nvflare.app_opt.p2p.executors.gtadam import GTADAMExecutor  # noqa: E402


class TinyAdam(GTADAMExecutor):
    def __init__(self):
        data = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(torch.tensor([[1.0]]), torch.tensor([[0.0]])),
            generator=torch.Generator().manual_seed(17),
        )
        gt.GTExecutor.__init__(self, torch.nn.Linear(1, 1), torch.nn.MSELoss(), data, data)


def test_consensus_updates_weighted_neighbor_values_preserves_history_and_releases_round_cache(tmp_path, monkeypatch):
    executor = ConsensusExecutor(initial_value=2.0)
    executor._weight = 0.5
    executor.neighbors = [SimpleNamespace(id=1, weight=0.5)]
    sent = []

    def exchange(ctx, value, iteration):
        sent.append(value)
        executor.neighbors_values[iteration] = {1: 6.0}

    monkeypatch.setattr(executor, "_exchange_values", exchange)
    executor.run_algorithm(FLContext(), DXO(DataKind.APP_DEFINED, {"iterations": 2}).to_shareable(), Signal())
    assert sent == [2, 4]
    assert executor.value_history == [2, 4, 5]
    assert executor.neighbors_values == {}
    monkeypatch.chdir(tmp_path)
    executor._post_algorithm_run()
    torch.testing.assert_close(torch.load("value_sequence.pt", weights_only=True), torch.tensor([2.0, 4.0, 5.0]))
    abort = Signal()
    abort.trigger(True)
    executor.run_algorithm(FLContext(), DXO(DataKind.APP_DEFINED, {"iterations": 3}).to_shareable(), abort)
    assert executor.value_history == [2, 4, 5]


def test_gtadam_real_initialization_and_clipped_moment_step_match_closed_form(monkeypatch):
    monkeypatch.setattr(gt, "get_device", lambda: torch.device("cpu"))
    torch_state = torch.random.get_rng_state()
    try:
        executor = TinyAdam()
    finally:
        torch.random.set_rng_state(torch_state)
    with torch.no_grad():
        executor.model.weight.fill_(4)
        executor.model.bias.fill_(1)
    config = {"iterations": 2, "stepsize": 0.1, "beta1": 0.5, "beta2": 0.5, "epsilon": 0.01}
    executor._pre_algorithm_run(FLContext(), DXO(DataKind.APP_DEFINED, config).to_shareable(), Signal())
    assert executor.train_loss_sequence[0][1] == pytest.approx(25)
    assert executor.test_loss_sequence[0][1] == pytest.approx(25)
    assert executor._iterations == 2 and executor._stepsize == 0.1
    executor.tracker = [torch.full_like(p, 2) for p in executor.model.parameters()]
    executor.G = torch.tensor(1.0)
    before = [p.detach().clone() for p in executor.model.parameters()]
    executor._update_local_state(0.1)
    for old, current, moment, square in zip(before, executor.model.parameters(), executor.m, executor.v):
        torch.testing.assert_close(moment, torch.ones_like(moment))
        torch.testing.assert_close(square, torch.ones_like(square))
        torch.testing.assert_close(current, old - 0.1 / np.sqrt(1.01))
