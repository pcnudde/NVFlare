# PyTorch mid-epoch checkpointing

[client.py](client.py) fine-tunes pinned `Qwen/Qwen2.5-0.5B-Instruct` on 16 toy texts.
It shows the user checkpoint code needed by [PR #5617](https://github.com/NVIDIA/NVFlare/pull/5617).
**Saves occur every 2 optimizer steps: eight times per 16-batch epoch.**

## Run

```bash
pip install -r requirements.txt
python job.py                              # one-client, two-round simulation
python job.py --export --export-dir ./jobs # submit to a provisioned deployment
```

Add `--device cuda` for one GPU, `--execution-lifetime task` for Kubernetes task
workers, or `--checkpoint-every N` to change the interval **in steps**.

## Resume contract

- Save weights, AdamW, scheduler, shuffle order/cursor, PyTorch CPU/CUDA RNGs and loss.
- Write/fsync a temporary file, then atomically replace `latest.pt`; restore RNGs last.
- Check job/site/round, input/data hashes and settings before restoring.
- Retain completed state to resend a result without retraining. New rounds reset AdamW.

Checkpoints are beside the deployed script at `checkpoints/<job>/<site>/latest.pt`.
Use the [persistent jobs PVC](../../../docs/user_guide/admin_guide/deployment/kubernetes_task_launcher.rst);
resident Kubernetes CJs also need `persist_resident_workspaces: true`. Keep CP/server
running and allow enough task time for restarts. Each FP32 checkpoint is about 6 GB;
allow space for its replacement and keep checkpoint intervals below the kill interval.

Unsaved steps are replayed. Exact continuation requires unchanged hardware/software/data.
This example uses one device and no AMP, accumulation or data workers. Checkpoint their
additional state if adding them, including Python/NumPy RNGs if used.

## Verify

From the repository root:

```bash
python -m pytest -q tests/unit_test/examples/pt_checkpoint_test.py \
    tests/integration_test/fast/pt_checkpoint_resume_test.py
```

The tiny dropout model tests exact state equality across repeated SIGKILLs, including
during a checkpoint write. GPU/Kubernetes execution is not qualified by these CPU tests.
