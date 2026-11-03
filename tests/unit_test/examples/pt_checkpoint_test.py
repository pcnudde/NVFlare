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

import os

import pytest
import torch

from tests.pt_checkpoint_test_utils import ASSIGNMENT, assert_identical, load_client, train, workload


@pytest.fixture(autouse=True)
def isolated_training_state(monkeypatch):
    threads = torch.get_num_threads()
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", os.environ.get("CUBLAS_WORKSPACE_CONFIG", ":4096:8"))
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", torch.backends.cuda.matmul.allow_tf32)
    torch.set_num_threads(1)
    with torch.random.fork_rng():
        try:
            yield
        finally:
            torch.set_num_threads(threads)
            torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)


@pytest.mark.parametrize("stop_step", [None, 2, 4, 10, 12])
def test_resume_matches_uninterrupted_training(tmp_path, monkeypatch, stop_step):
    module = load_client()
    baseline = tmp_path / "baseline.pt"
    expected = train(module, baseline)
    resumed = tmp_path / "resumed.pt"
    save = module.atomic_save

    def interrupt(state, path):
        if stop_step is None:  # Crash before the first checkpoint is published.
            raise InterruptedError()
        save(state, path)
        if state["progress"]["step"] == stop_step:
            raise InterruptedError()

    with monkeypatch.context() as patch:
        patch.setattr(module, "atomic_save", interrupt)
        with pytest.raises(InterruptedError):
            train(module, resumed)
    assert train(module, resumed) == expected
    assert_identical(torch.load(baseline, weights_only=True), torch.load(resumed, weights_only=True))


def test_default_saves_multiple_times_per_epoch(tmp_path, monkeypatch):
    module = load_client()
    saved_steps = []
    save = module.atomic_save

    def record(state, path):
        save(state, path)
        saved_steps.append(state["progress"]["step"])

    monkeypatch.setattr(module, "atomic_save", record)
    train(module, tmp_path / "latest.pt")  # Three epochs, four batches per epoch.
    assert saved_steps == [2, 4, 6, 8, 10, 12]


@pytest.mark.parametrize("change", ["job", "site", "task", "round", "weights", "data", "batch_size"])
def test_rejects_incompatible_checkpoint(tmp_path, change):
    module = load_client()
    path = tmp_path / "latest.pt"
    train(module, path)
    model, tokens = workload(module)
    assignment = dict(ASSIGNMENT)
    batch_size = 2
    if change in ("job", "site", "task"):
        assignment[change] = "different"
    elif change == "round":
        assignment["round"] = -1
    elif change == "weights":
        with torch.no_grad():
            next(model.parameters()).add_(1)
    elif change == "data":
        tokens["input_ids"][0, 0] = 1
    else:
        batch_size = 3
    with pytest.raises(ValueError, match="checkpoint"):
        module.train_round(
            model,
            tokens,
            path,
            assignment,
            epochs=3,
            batch_size=batch_size,
            learning_rate=0.01,
            checkpoint_every=2,
        )


def test_next_round_uses_new_weights_and_fresh_optimizer(tmp_path):
    module = load_client()
    previous = tmp_path / "previous.pt"
    train(module, previous)
    assignment = {**ASSIGNMENT, "round": 1}
    expected = train(module, tmp_path / "fresh.pt", assignment)
    assert train(module, previous, assignment) == expected
    assert_identical(torch.load(previous, weights_only=True), torch.load(tmp_path / "fresh.pt", weights_only=True))
