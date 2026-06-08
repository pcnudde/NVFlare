# Agent Skill Eval

Small harness for running Markdown-based agent skill testcases against agent
CLIs in Docker. The current testcase is
`tests/agent_skill_eval/testcases/nvflare_basic_pytorch_to_sim/`.

## Images

Build the base image and the two NVFlare comparison images from the repo root:

```bash
docker build -f tests/agent_skill_eval/docker/basic/Dockerfile -t nvflare-agent-eval:basic .
docker build -f tests/agent_skill_eval/docker/nvflare-2.8/Dockerfile -t nvflare-agent-eval:2.8 .
docker build -f tests/agent_skill_eval/docker/nvflare-2.9-skills/Dockerfile -t nvflare-agent-eval:2.9-skills .
```

`basic` has Python 3.12, torch, Codex CLI, and Claude Code CLI. `2.8` installs
`nvflare==2.8.0`. `2.9-skills` installs FLARE from the `flare_agent` branch and
pre-installs Codex and Claude skills.

## Run

Run the default comparison:

```bash
python3 tests/agent_skill_eval/run_full_eval.py
```

Run one agent:

```bash
python3 tests/agent_skill_eval/harness.py \
  --agent codex-5.5-xhigh \
  --docker-image nvflare-agent-eval:2.8 \
  --docker-oauth codex
```

Generate a report from one or more run directories:

```bash
python3 tests/agent_skill_eval/report.py tests/agent_skill_eval/runs/<timestamp>
```

## Configuration

- Agents live in `tests/agent_skill_eval/agents.yaml`.
- Public fallback token prices live in `tests/agent_skill_eval/model_costs.yaml`.
- Testcases are Markdown folders with `testcase.md` and `initial/`.
- Use `--runs-per-agent N` for repeated runs and `--parallel N` for concurrent
  runs.
- Use `--docker-image` to compare the same testcase across NVFlare images.
- On a remote host, update the checkout and run `python3 tests/agent_skill_eval/run_full_eval.py`.

The agent-under-test and testcase evidence commands run inside the Docker
container. The Codex grader and separate Codex analysis pass run on the host
after evidence collection. If an agent times out, the run receives score 0 and
evidence/grading is skipped.

Runs are written under `tests/agent_skill_eval/runs/` with `result.json`, logs,
the final copied workspace, and a zip archive for each run. Use `report.py` for
summary tables, cost estimates, and aggregation.

The harness records public agent output. It does not capture hidden
chain-of-thought.
