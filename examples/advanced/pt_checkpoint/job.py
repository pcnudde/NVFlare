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

"""Export or simulate a FedAvg job with recoverable local fine-tuning."""

import argparse

from client import MODEL_NAME

from nvflare.app_opt.pt.recipes.fedavg import FedAvgRecipe
from nvflare.client.config import ExchangeFormat
from nvflare.recipe import SimEnv


def main():
    parser = argparse.ArgumentParser(description=__doc__, epilog="Use --export --export-dir DIR to export the job.")
    parser.add_argument("--num-clients", type=int, default=1)
    parser.add_argument("--num-rounds", type=int, default=2)
    parser.add_argument("--execution-lifetime", choices=("job", "task"), default="job")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--model", default=MODEL_NAME, help="Pinned Qwen model, or a local snapshot directory")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--checkpoint-every", type=int, default=2)
    args = parser.parse_args()
    recipe = FedAvgRecipe(
        name="pt-checkpoint",
        min_clients=args.num_clients,
        num_rounds=args.num_rounds,
        model={"class_path": "client.QwenModel", "args": {"model_name": args.model}},
        train_script="client.py",
        train_args=[
            "--device",
            args.device,
            "--model",
            args.model,
            "--epochs",
            str(args.epochs),
            "--checkpoint-every",
            str(args.checkpoint_every),
        ],
        execution_lifetime=args.execution_lifetime,
        server_expected_format=ExchangeFormat.PYTORCH,
        enable_tensor_disk_offload=True,
        key_metric="",
    )
    run = recipe.execute(SimEnv(num_clients=args.num_clients))
    print("Result:", run.get_result())


if __name__ == "__main__":
    main()
