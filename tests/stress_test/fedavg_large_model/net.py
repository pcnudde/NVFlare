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

import torch
import torch.nn as nn

NUM_LAYERS = 50


class StressNet(nn.Module):
    def __init__(self, size_gb: float = 0.1):
        super().__init__()
        total_elements = int(size_gb * (1024**3) / 4)
        per_layer = total_elements // NUM_LAYERS
        remainder = total_elements - per_layer * NUM_LAYERS

        for i in range(NUM_LAYERS):
            n = per_layer + (1 if i < remainder else 0)
            self.register_parameter(f"layer_{i}", nn.Parameter(torch.ones(n), requires_grad=False))

    def forward(self, x):
        return x
