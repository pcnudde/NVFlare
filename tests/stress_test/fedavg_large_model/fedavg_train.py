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

"""Client training script for FedAvg stress test.

Receives weights via FLARE Client API, adds delta=1.0 to all tensors,
and sends back. Deterministic so results are reproducible.
"""

import torch

import nvflare.client as flare

DELTA = 1.0


def main():
    flare.init()

    while flare.is_running():
        model = flare.receive()
        print(f"  [client] Round {model.current_round}: {len(model.params)} tensors")

        for k, v in model.params.items():
            if isinstance(v, torch.Tensor):
                model.params[k] = v + DELTA

        model.meta["NUM_STEPS_CURRENT_ROUND"] = 1
        flare.send(model)


if __name__ == "__main__":
    main()
