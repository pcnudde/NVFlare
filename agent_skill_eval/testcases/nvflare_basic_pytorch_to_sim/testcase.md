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
- Version comparison runs may override this with `nvflare-agent-eval:2.8` or
  `nvflare-agent-eval:2.9-skills`.
- Agent timeout: 20 minutes.
- Python 3.12 with `torch` installed.
- NVFlare availability and version are determined by the selected Docker image.
  If NVFlare is already installed, use that installed package instead of
  changing the environment version.
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
with 3 clients and 5 rounds. I want to run it in simulation. Add job.py as the
simulation entry point. Install or configure only missing dependencies
persistently so `python job.py` works in this environment when you are done.
```

## Expected Outcome

The agent should convert the centralized trainer into a runnable NVFlare
federated simulation.

A strong solution:

- Adds `job.py` as the executable NVFlare simulation entry point.
- Adds or refactors NVFlare client training code. This can be a separate client
  script or a federated mode in the existing trainer.
- Uses the NVFlare FedAvg recipe API in `job.py`: `FedAvgRecipe`.
- Uses a simulation environment in `job.py`: `SimEnv`.
- Configures 3 simulated clients and 5 federated rounds.
- Uses `SimpleNetwork` from `model.py`.
- Uses the NVFlare Client API in the client training loop:
  `flare.init()`, `flare.receive()`, and `flare.send()`.
- Reuses or adapts the starting training functions: `build_loaders`,
  `train_one_epoch`, and `evaluate`.
- Reports an accuracy metric above 0.50 from `evaluate` during or after
  federated training, so the run demonstrates that federated training actually
  happened.
- Uses the NVFlare package already installed in the selected Docker image. If
  NVFlare is missing, installs a persistent dependency without relying on extra
  `PYTHONPATH` values, temporary source trees, or hidden one-off setup commands.

## Grading Rubric

Score from 0 to 100.

- Functional FL job, 70 pts:
  - `python job.py` runs a simulation successfully, 20 pts.
  - The simulation is configured for exactly 3 clients and 5 federated rounds,
    15 pts.
  - The job performs actual federated training with client-local updates and
    server-side aggregation, not just centralized training or fake logs, 15 pts.
  - The FL code uses `SimpleNetwork` from `model.py` and reuses or adapts the
    starting `StripeDataset`, `build_loaders`, `train_one_epoch`, and
    `evaluate` logic, 10 pts.
  - The simulation reports a meaningful accuracy metric above 0.50 during or
    after federated training, 10 pts.
- NVFlare API quality, 20 pts:
  - `job.py` uses the NVFlare FedAvg recipe API with `FedAvgRecipe` and a
    simulation environment with `SimEnv`, 10 pts.
  - The FL client code uses the NVFlare Client API with `flare.init()`,
    `flare.receive()`, and `flare.send()` around local PyTorch training, 10 pts.
- Usability and environment hygiene, 10 pts:
  - Adds a clear root-level `job.py`, keeps the code readable and CPU-safe, and
    does not unnecessarily destroy the starting training script, 5 pts.
  - Uses the NVFlare version already installed in the selected Docker image, or
    if NVFlare is missing, installs a persistent dependency without temporary
    path hacks or hidden setup commands, 5 pts.

## Score Caps

Apply these caps after assigning the rubric score:

- Max 40 if `python job.py` does not complete.
- Max 30 if the result is not actually federated.
- Max 20 for syntax or import errors in core files such as `model.py`, `job.py`,
  `train.py`, or separate FL client code.
- Max 80 if the job works but misses either `FedAvgRecipe` or the NVFlare Client
  API.
- Max 90 if the job works but changes or reinstalls the selected Docker image's
  NVFlare version without being asked.

## Evidence To Collect

Before grading, collect:

```bash
python -m py_compile model.py job.py
timeout 600 python job.py
```

Also inspect `job.py`, `train.py`, `model.py`, and any separate FL client script
when assigning partial credit.
