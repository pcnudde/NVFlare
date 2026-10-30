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

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("pytorch_lightning")

from nvflare.app_opt.lightning.callbacks import RestoreState  # noqa: E402


def test_lightning_callback_roundtrip_restores_optimizer_scheduler_and_copies_scaler_state():
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = torch.optim.SGD([parameter], lr=0.2, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.5)
    parameter.grad = torch.tensor([2.0])
    optimizer.step()
    scheduler.step()
    scaler_data = {"scale": 1024, "nested": [1]}
    scaler = Mock()
    scaler.state_dict.return_value = scaler_data

    def restore(checkpoint):
        optimizer.load_state_dict(checkpoint["optimizer_states"][0])

    trainer = SimpleNamespace(
        optimizers=[optimizer],
        scaler=scaler,
        lr_scheduler_configs=[SimpleNamespace(scheduler=scheduler)],
        strategy=SimpleNamespace(load_optimizer_state_dict=Mock(side_effect=restore)),
    )
    callback = RestoreState()
    callback.on_fit_start(trainer, None)
    trainer.strategy.load_optimizer_state_dict.assert_not_called()
    callback.on_fit_end(trainer, None)
    scaler_data["nested"].append(2)
    assert callback.scaler_states[0]["nested"] == [1]
    optimizer.param_groups[0]["lr"] = 9
    scheduler.last_epoch = 20
    callback.on_fit_start(trainer, None)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.1)
    assert scheduler.last_epoch == 1
    scaler.load_state_dict.assert_called_once_with({"scale": 1024, "nested": [1]})
    torch.testing.assert_close(optimizer.state[parameter]["momentum_buffer"], torch.tensor([2.0]))
