---
name: nvflare
description: Build, convert, validate, and troubleshoot NVFLARE federated learning jobs using the Recipe API and Client API.
min_flare_version: "2.8.0"
skill_version: "0.1.0"
---

# NVFLARE

Use this skill when the task is to build, convert, validate, export, or
troubleshoot an NVFLARE job.

## API Choice

- Prefer the Recipe API for `job.py`. Do not hand-write app config JSON unless
  no recipe fits the requested workflow.
- Prefer the Client API in the client training script. Do not implement raw
  executors, controllers, `Shareable`, or `DXO` code for normal training-loop
  conversion.
- Use `SimEnv` for local validation. Do not use the deprecated `nvflare
  simulator` command for new jobs.
- Keep a runnable `job.py` at the workspace or project root when the user asks
  for a generated solution. Nested job folders alone are not enough.

## Minimal Recipe Pattern

For PyTorch FedAvg:

```python
from nvflare.app_opt.pt.recipes import FedAvgRecipe
from nvflare.recipe import SimEnv

recipe = FedAvgRecipe(
    name="my_job",
    model=model,
    min_clients=2,
    num_rounds=2,
    train_script="client.py",
)

if __name__ == "__main__":
    recipe.execute(SimEnv(num_clients=2))
```

Export with the same `job.py`:

```bash
python job.py --export --export-dir ./jobs
```

Use `nvflare recipe list --framework pytorch --format json` when recipe choice
is ambiguous. Choose FedAvg only for standard horizontal weighted averaging.

## Minimal Client API Pattern

```python
from nvflare.app_common.abstract.fl_model import FLModel
import nvflare.client as flare

flare.init()
while flare.is_running():
    input_model = flare.receive()
    params = input_model.params

    # Load params into the local model, train or evaluate, then collect updates.
    output_params = params
    metrics = {"accuracy": 0.0}

    flare.send(FLModel(params=output_params, metrics=metrics))
```

For PyTorch in-process Client API jobs, send PyTorch tensor `state_dict`
values when the selected recipe/executor expects PyTorch exchange. Preserve
model constructor arguments explicitly in `job.py` when the model has required
non-default `__init__` parameters.

## Validation

- Inspect the existing training script before editing.
- Run `python job.py` for a local simulation when data and dependencies are
  available.
- Run `python job.py --export --export-dir <dir>` before submission workflows.
- Do not submit to POC or production without explicit user approval.
