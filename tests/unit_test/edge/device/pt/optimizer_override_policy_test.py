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

import pytest

torch = pytest.importorskip("torch")

from nvflare.edge.device.pt.optimizers import AdamOptimizerWrapper, SGDOptimizerWrapper


@pytest.mark.parametrize("wrapper_type", [AdamOptimizerWrapper, SGDOptimizerWrapper])
def test_per_training_optimizer_override_changes_step_without_mutating_configured_defaults(wrapper_type):
    wrapper = wrapper_type(lr=0.1)
    parameter = torch.nn.Parameter(torch.tensor([2.0], device="cpu"))
    optimizer = wrapper.get([parameter], lr=0.2)
    parameter.grad = torch.tensor([1.0], device="cpu")
    optimizer.step()
    torch.testing.assert_close(parameter.detach(), torch.tensor([1.8]), atol=1e-6, rtol=1e-6)
    assert wrapper.kwargs == {"lr": 0.1}
    later = wrapper.get([torch.nn.Parameter(torch.tensor([2.0]))])
    assert later.defaults["lr"] == 0.1 and optimizer.defaults["lr"] == 0.2
