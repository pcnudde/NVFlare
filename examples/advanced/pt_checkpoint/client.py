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

"""Plain PyTorch fine-tuning with application-owned, mid-epoch checkpoints."""

import argparse
import hashlib
import math
import os
from pathlib import Path

import torch

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
MODEL_REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"


class QwenModel(torch.nn.Module):
    """The same public model as hello-huggingface, with all parameters trainable."""

    def __init__(self, model_name=MODEL_NAME, revision=MODEL_REVISION):
        super().__init__()
        from transformers import AutoModelForCausalLM

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, revision=revision, torch_dtype=torch.float32, attn_implementation="eager"
        )
        self.model.config.use_cache = False

    def forward(self, **batch):
        return self.model(**batch)


def seed_training(seed):
    # Call before creating the CUDA context; use the same environment on resume.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(seed)


def tensor_digest(tensors):
    """Bind a checkpoint to the actual input weights and tokenized data."""
    digest = hashlib.sha256()
    for name, value in sorted(tensors.items()):
        value = value.detach().cpu().contiguous()
        digest.update(f"{name}:{value.dtype}:{tuple(value.shape)}".encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def atomic_save(state, path):
    """Publish only a complete checkpoint, on the same persistent filesystem."""
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(".tmp")
    with pending.open("wb") as stream:
        torch.save(state, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(pending, path)
    # Persist the rename too. There is one writer: PR #5617 settles the old
    # client/worker before starting its replacement.
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def train_round(
    model,
    tokens,
    checkpoint_path,
    assignment,
    *,
    epochs=2,
    batch_size=1,
    learning_rate=2e-5,
    checkpoint_every=2,
    seed=42,
):
    """Resume this FedAvg round; a new round starts a fresh local optimizer."""
    if min(epochs, batch_size, checkpoint_every) < 1 or learning_rate <= 0:
        raise ValueError("epochs, batch_size, checkpoint_every and learning_rate must be positive")
    size = len(tokens["input_ids"])
    if size == 0:
        raise ValueError("training data is empty")
    batches_per_epoch = math.ceil(size / batch_size)
    total_steps = epochs * batches_per_epoch
    device = next(model.parameters()).device
    binding = {
        "assignment": assignment,
        "input": tensor_digest(model.state_dict()),
        "data": tensor_digest(tokens),
        "settings": (epochs, batch_size, learning_rate, seed, str(device)),
    }
    log_context = (
        f"job={assignment['job']} site={assignment['site']} round={assignment['round']} "
        f"host={os.environ.get('HOSTNAME', 'local')} pid={os.getpid()}"
    )
    print(
        f"Training {log_context} steps={total_steps} checkpoint_every={checkpoint_every} "
        f"input_sha256={binding['input']} data_sha256={binding['data']}",
        flush=True,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, foreach=False)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
    progress = {"step": 0, "order": None, "loss_sum": 0.0}
    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=True) if checkpoint_path.exists() else None
    if saved is not None:
        previous = saved["binding"]["assignment"]
        if any(previous[key] != assignment[key] for key in ("job", "site", "task")):
            raise ValueError("checkpoint belongs to a different job, site or task")
        if previous["round"] > assignment["round"]:
            raise ValueError("checkpoint is from a later round")
        if previous["round"] == assignment["round"]:
            if saved["binding"] != binding:
                raise ValueError("checkpoint input model, data or training settings changed")
            model.load_state_dict(saved["model"])
            optimizer.load_state_dict(saved["optimizer"])
            scheduler.load_state_dict(saved["scheduler"])
            progress = saved["progress"]
            # Restore RNGs last, after model/optimizer construction and loading.
            torch.set_rng_state(saved["torch_rng"])
            if saved["cuda_rng"] is not None:
                torch.cuda.set_rng_state(saved["cuda_rng"], device)
            print(f"Resumed {log_context} step={progress['step']}/{total_steps}", flush=True)
    # Explicit per-round seeding makes a new round independent of CJ lifetime.
    if progress["step"] == 0:
        seed_training(seed + assignment["round"])
    del saved
    model.train()

    while progress["step"] < total_steps:
        batch_index = progress["step"] % batches_per_epoch
        if batch_index == 0:
            progress["order"] = torch.randperm(size)
        indices = progress["order"][batch_index * batch_size : (batch_index + 1) * batch_size]
        # Direct indexing avoids DataLoader workers, prefetch and skipped-batch replay.
        batch = {key: value[indices].to(device) for key, value in tokens.items()}
        optimizer.zero_grad(set_to_none=True)
        loss = model(**batch).loss
        loss.backward()
        optimizer.step()
        scheduler.step()
        progress["step"] += 1
        progress["loss_sum"] += loss.item()
        # A safe point: weights, optimizer, scheduler, RNG and the next batch
        # agree. No pending accumulated gradients or mixed-precision scaler.
        if progress["step"] % checkpoint_every == 0 or progress["step"] == total_steps:
            atomic_save(
                {
                    "binding": binding,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "progress": progress,
                    "torch_rng": torch.get_rng_state(),
                    "cuda_rng": torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
                },
                checkpoint_path,
            )
            print(
                f"Checkpoint {log_context} step={progress['step']}/{total_steps} "
                f"next_epoch={progress['step'] // batches_per_epoch} "
                f"next_batch={progress['step'] % batches_per_epoch}",
                flush=True,
            )
    # Keep the final checkpoint even after send: redelivery must resend the
    # same trained weights/metrics, without executing another local epoch.
    metrics = {"train_loss": progress["loss_sum"] / total_steps}
    print(
        f"Completed {log_context} steps={total_steps} train_loss={metrics['train_loss']:.17g} "
        f"model_sha256={tensor_digest(model.state_dict())}",
        flush=True,
    )
    return metrics, total_steps


def prepare_tokens(model_name):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, revision=MODEL_REVISION)
    texts = [
        "Federated learning keeps training data at each client.",
        "Clients share model updates with the server.",
        "The server aggregates updates into a global model.",
        "Checkpoints preserve training state across interruptions.",
    ] * 4
    tokens = dict(tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=64))
    tokens["labels"] = tokens["input_ids"].clone()
    tokens["labels"][tokens["attention_mask"] == 0] = -100
    return tokens


def main():
    import nvflare.client as flare

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=MODEL_NAME, help="Pinned Qwen model, or a local snapshot directory")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--checkpoint-every", type=int, default=2, help="Save every N optimizer steps")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    torch.set_num_threads(1)
    seed_training(42)
    flare.init()
    model = QwenModel(args.model).to(args.device)
    tokens = prepare_tokens(args.model)
    while flare.is_running():
        incoming = flare.receive()
        if not flare.is_train():
            raise ValueError("this example only handles FedAvg train assignments")
        assignment = {
            "job": flare.get_job_id(),
            "site": flare.get_site_name(),
            "task": "train",
            "round": incoming.current_round,
        }
        model.load_state_dict(incoming.params)
        incoming.params = None
        path = Path(__file__).resolve().parent / "checkpoints" / assignment["job"] / assignment["site"] / "latest.pt"
        metrics, steps = train_round(
            model,
            tokens,
            path,
            assignment,
            epochs=args.epochs,
            checkpoint_every=args.checkpoint_every,
        )
        flare.send(
            flare.FLModel(
                params={key: value.detach().cpu() for key, value in model.state_dict().items()},
                metrics=metrics,
                current_round=assignment["round"],
                meta={"NUM_STEPS_CURRENT_ROUND": steps},
            )
        )


if __name__ == "__main__":
    main()
