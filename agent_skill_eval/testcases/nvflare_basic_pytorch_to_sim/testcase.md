# Convert Centralized PyTorch Image Training to NVFlare Simulation

Difficulty: Basic

## Starting Point

Copy `initial/my_project/` to a clean temporary workspace and start the agent in
the copied `my_project/` folder.

Starting files:

```text
my_project/
├── train.py          # Centralized PyTorch learnable synthetic image training script, no NVFlare imports
├── model.py          # SimpleNetwork CNN definition
└── requirements.txt  # torch
```

Environment:

- Docker image: `nvflare-agent-eval:basic`.
- Agent timeout: 20 minutes.
- Python 3.12 with `torch` installed.
- `nvflare` is not installed in the agent container.
- CPU-only execution must work.

Starting `train.py` contains these functions/classes:

- `StripeDataset`
- `build_loaders`
- `train_one_epoch`
- `evaluate`
- `main`

The unmodified centralized script should reach non-random accuracy on the
learnable synthetic dataset. With the starting defaults, `python train.py
--epochs 2` should usually finish above 0.90 accuracy.

## Prompt

```text
Convert my centralized training script to federated learning using NVIDIA FLARE
with 3 clients and 5 rounds. I want to run it in simulation.
```

## Expected Outcome

The agent should convert the centralized trainer into a runnable NVFlare
federated simulation.

A strong solution:

- Adds `job.py` as the executable NVFlare simulation entry point.
- Adds `client.py` as the NVFlare client training script.
- Uses the NVFlare FedAvg recipe API in `job.py`: `FedAvgRecipe`.
- Uses a simulation environment in `job.py`: `SimEnv`.
- Configures 3 simulated clients and 5 federated rounds.
- Uses `SimpleNetwork` from `model.py`.
- Uses the NVFlare Client API in the client training loop:
  `flare.init()`, `flare.receive()`, and `flare.send()`.
- Reuses or adapts the starting training functions: `build_loaders`,
  `train_one_epoch`, and `evaluate`.
- Reports an accuracy metric from `evaluate` during or after federated training,
  so the result can be compared to the centralized baseline.

## Grading Rubric

Score from 0 to 100.

- FedAvg recipe simulation job, 30 pts: `job.py` imports `FedAvgRecipe`;
  imports `SimEnv`; constructs `FedAvgRecipe` with `min_clients=3`,
  `num_rounds=5`, and `SimpleNetwork()` as `model` or `initial_model`;
  constructs `SimEnv` with `num_clients=3`; calls the recipe execution method
  with that simulation environment.
- Client API integration, 30 pts: `client.py` imports `nvflare.client` as
  `flare`; calls `flare.init()`; calls `flare.receive()` in the federated round
  loop; loads received parameters into `SimpleNetwork`; performs local PyTorch
  training; calls `flare.send()` with `flare.FLModel(...)` containing updated
  model parameters and at least one metric.
- Training adaptation, 20 pts: The FL client reuses or adapts
  `StripeDataset`, `build_loaders`, `train_one_epoch`, and `evaluate`; uses
  `SimpleNetwork` from `model.py`; uses `torch.optim.SGD` and
  `nn.CrossEntropyLoss` or the same optimizer/loss from the starting script;
  works on CPU; does not replace the learnable synthetic dataset with random
  labels.
- Runtime behavior, 15 pts: `model.py`, `client.py`, and `job.py` parse;
  `python job.py` runs the configured simulation without uncaught errors; the
  simulation evidence includes an accuracy metric comparable to the centralized
  baseline, ideally at least 0.80 after 5 rounds.
- Dependency declaration, 5 pts: `requirements.txt` declares an `nvflare`
  dependency compatible with the APIs used.

## Score Caps

Apply these caps after assigning the rubric score:

- Max 60 if the client does not use the NVFlare Client API.
- Max 60 if `job.py` does not use `FedAvgRecipe`.
- Max 55 if there is no runnable simulation job entry point.
- Max 50 if `job.py` or `client.py` has syntax errors.
- Max 75 if the solution does not use `SimpleNetwork` from `model.py`.
- Max 70 if the simulation is not configured for exactly 3 clients and 5
  rounds.

## Evidence To Collect

Before grading, collect:

```bash
python -m py_compile model.py client.py job.py
timeout 600 python job.py
```

## Baseline Evidence To Collect

Before running the agent, collect this from a separate copy of the unmodified
starting workspace:

```bash
python train.py --epochs 2
```

Also inspect `job.py`, `client.py`, `train.py`, and `model.py` when assigning
partial credit.
