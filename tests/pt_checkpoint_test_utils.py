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

"""Small offline workload for the real example's checkpoint/training loop."""

import importlib.util
import os
import signal
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

EXAMPLE = Path(__file__).resolve().parents[1] / "examples/advanced/pt_checkpoint/client.py"
ASSIGNMENT = {"job": "job-1", "site": "site-1", "task": "train", "round": 0}


def load_client():
    spec = importlib.util.spec_from_file_location("checkpoint_client", EXAMPLE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TinyLM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(16, 8)
        self.dropout = torch.nn.Dropout(0.4)
        self.output = torch.nn.Linear(8, 16)

    def forward(self, input_ids, labels):
        hidden = self.dropout(self.embedding(input_ids))
        logits = self.output(hidden)
        loss = torch.nn.functional.cross_entropy(logits[:, :-1].reshape(-1, 16), labels[:, 1:].reshape(-1))
        return SimpleNamespace(loss=loss)


def workload(module):
    module.seed_training(42)
    model = TinyLM()
    inputs = torch.arange(35).reshape(7, 5) % 16
    return model, {"input_ids": inputs, "labels": inputs.clone()}


def train(module, path, assignment=None):
    model, tokens = workload(module)
    return module.train_round(
        model,
        tokens,
        path,
        assignment or ASSIGNMENT,
        epochs=3,
        batch_size=2,
        learning_rate=0.01,
    )


def assert_identical(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_identical(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            assert_identical(a, b)
    else:
        assert left == right


def run_worker():
    """SIGKILL fault injection lives in tests, not in the training example."""
    path, crash_step, phase = Path(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
    torch.set_num_threads(1)
    module = load_client()
    original_save = module.atomic_save
    original_torch_save = torch.save

    class InterruptedWriter:
        def __init__(self, stream):
            self.stream = stream

        def write(self, data):
            self.stream.write(data[:512])
            self.stream.flush()
            os.kill(os.getpid(), signal.SIGKILL)

        def flush(self):
            self.stream.flush()

    def torch_save(state, stream):
        if phase == "during_save" and state["progress"]["step"] == crash_step:
            stream = InterruptedWriter(stream)
        return original_torch_save(state, stream)

    def save(state, destination):
        original_save(state, destination)
        if phase == "after_save" and state["progress"]["step"] == crash_step:
            os.kill(os.getpid(), signal.SIGKILL)

    torch.save = torch_save
    module.atomic_save = save
    train(module, path)


if __name__ == "__main__":
    run_worker()
