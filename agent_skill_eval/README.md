# Agent Skill Eval

This is a minimal harness for running the Markdown testcase in
`testcases/nvflare_basic_pytorch_to_sim/` against the configured agent
CLIs.

Build the basic agent container from the repo root:

```bash
docker build -f agent_skill_eval/docker/basic/Dockerfile -t nvflare-agent-eval:basic .
```

The image name is also recorded in each testcase. The basic image has Linux,
Python 3.12, `torch`, and the agent CLIs needed by the harness.
It does not install NVFlare.

Build versioned NVFlare images from the current basic image:

```bash
docker build -f agent_skill_eval/docker/nvflare-2.8/Dockerfile -t nvflare-agent-eval:2.8 .
docker build -f agent_skill_eval/docker/nvflare-2.9-skills/Dockerfile -t nvflare-agent-eval:2.9-skills .
```

The 2.8 image installs `nvflare==2.8.0` from pip. The 2.9 skills image installs
FLARE from `git+https://github.com/chesterxgchen/NVFlare.git@flare_agent` and
pre-installs skills for Codex, Claude, and Hermes.

Run all configured agents:

```bash
python3 agent_skill_eval/harness.py --docker-oauth all
```

By default, the harness runs each selected testcase/agent pair 3 independent
times. Use `--runs-per-agent N` to change that count.

The agents under test are configured in `agent_skill_eval/agents.yaml`, not in
the harness. Use `--agents-file` to point at another YAML file.

Model pricing is configured in `agent_skill_eval/model_costs.yaml`. The file is
structured around prices per 1M tokens and includes separate rates for normal
input, output, cached input, cache creation, and cache reads. The harness prefers
agent CLI-reported cost when available; otherwise it computes a fallback estimate
from this file. Use `--model-costs-file /path/to/model_costs.yaml` to override
it, or `/dev/null` to disable configured estimates.

Cached tokens are handled as separate billing categories. For Anthropic-style
usage (`cache_creation_input_tokens` and `cache_read_input_tokens`), those
counts are added to token totals and priced with their own rates. For
OpenAI-style `cached_tokens`, cached tokens are treated as part of
`input_tokens`; the harness prices `input_tokens - cached_tokens` at the normal
input rate and `cached_tokens` at `cached_input_tokens` when that rate is
configured.

Agent timeouts are configured per testcase with an `Agent timeout` line in
`testcase.md`, for example `- Agent timeout: 20 minutes.`.

The agent-under-test command and testcase evidence commands run in Docker. The
harness mounts that run's copied testcase workspace at `/workspace` and does not
mount the repo or the user's home directory. The agent runs in a named container
that is kept after the run. Final evidence executes in that same container, so it
uses the environment the agent actually created. The Codex grader and analysis
evaluator run locally after evidence collection with read-only access to the
final workspace.

The local environment needs `PyYAML` for agent config loading and the Codex CLI
for grading/analysis. Testcase runtime dependencies such as `torch` and
`nvflare` are resolved inside Docker.

Run one agent:

```bash
python3 agent_skill_eval/harness.py --agent codex-5.5-xhigh --docker-oauth codex
```

Run the same testcase against a specific NVFlare image:

```bash
python3 agent_skill_eval/harness.py --docker-image nvflare-agent-eval:2.8 --docker-oauth codex
python3 agent_skill_eval/harness.py --docker-image nvflare-agent-eval:2.9-skills --docker-oauth codex
```

Run several testcase/agent pairs concurrently:

```bash
python3 agent_skill_eval/harness.py --parallel 4 --docker-oauth codex --docker-claude-keychain
```

Repeat `--testcase /path/to/testcase_dir` to run more than one testcase in the
same invocation. The harness writes each pair under
`runs/<timestamp>/<testcase_id>/<agent_id>/run_XX/`.

List agents:

```bash
python3 agent_skill_eval/harness.py --list-agents
```

Each run writes a timestamped directory under `agent_skill_eval/runs/` with:

- copied workspace and final files
- agent stdout/stderr, streamed while the agent is running
- evidence command stdout/stderr
- Codex grader stdout/stderr, streamed while grading is running, and parsed
  grade
- independent Codex analysis stdout/stderr, streamed while analysis is running,
  plus a parsed five-bullet run summary, FLARE version, achieved accuracy,
  testcase recommendations, and notable observations
- `summary.csv` with score, duration, token usage, and cost when the CLI output
  exposes those fields or `model_costs.yaml` contains matching rates
- `aggregate.csv` with avg/min/max score, duration, token usage, and cost per
  testcase/agent across repeated runs. The same aggregate table is printed at
  the end of the harness run.

For Hermes, the oneshot command prints only the final response. The harness
therefore exports the run container's CLI session store with
`hermes sessions export - --source cli`, saves it as `logs/hermes_sessions.jsonl`,
and normalizes Hermes token/cost fields into the same summary schema.

Generate a self-contained HTML report from one or more run directories:

```bash
python3 agent_skill_eval/report.py agent_skill_eval/runs/<timestamp>
python3 agent_skill_eval/report.py agent_skill_eval/runs/<timestamp-a> agent_skill_eval/runs/<timestamp-b>
```

The harness executes the evidence commands listed in each `testcase.md` inside
the agent container before grading, then stops the container without removing
it. Those commands are testcase-specific; this first testcase runs the generated
NVFlare job. When an evidence command runs `job.py`, the harness first probes
the active `nvflare` package in that same run container and records the runtime
version or import error; reports do not infer NVFlare versions from
`requirements.txt`.

If an agent exceeds the testcase timeout, the harness stops the run container,
skips final evidence and grader/analysis calls for that run, and records a
deterministic score of 0. The stopped container is still kept for debugging; use
`docker start <container-name>` before `docker exec -it <container-name> bash`.

Regrade starts a kept container only long enough to rerun evidence, then stops
it again. This keeps each run recoverable without keeping idle containers in
memory.

The agent container uses Docker's `bridge` network by default so Codex and
Claude can make model calls. The copied project folder is the only mounted
workspace.

For Codex, use `--docker-oauth codex`. The harness mounts only
`~/.codex/auth.json` into the container and the Codex agent commands run with
`--dangerously-bypass-approvals-and-sandbox --ignore-user-config
--ignore-rules`. Docker is the sandbox boundary for the agent under test; the
agent has broad rights inside the container, including shell network access and
the built-in web-search tool. It cannot read the host repo or home directory
because the only host mounts are the copied testcase workspace and the Codex
auth file.

For Claude, a normal host `claude auth login` can be backed by the macOS
keychain and is not portable into Linux Docker as OAuth state. On this machine,
the Anthropic API key is stored in the macOS Keychain under service
`Claude Code`. Use `--docker-claude-keychain` to read that key on the host and
pass it into Docker as `ANTHROPIC_API_KEY` without putting the key value on the
command line:

```bash
python3 agent_skill_eval/harness.py --agent claude-opus-4.8-xhigh --docker-claude-keychain
```

The first run may trigger a macOS Keychain prompt; approve it or choose
`Always Allow`. This command must be run outside the command sandbox because it
needs Keychain and Docker VM access.

You can also pass an already-exported key explicitly:

```bash
python3 agent_skill_eval/harness.py --agent claude-opus-4.8-xhigh --docker-env ANTHROPIC_API_KEY
```

The default Claude agents use `--dangerously-skip-permissions` inside Docker.
The harness does not mount host credential directories by default.

Agent containers run with `--security-opt seccomp=unconfined` by default. This
keeps the image compatible with agent CLIs that create Linux namespaces, though
the default Codex agents rely on Docker rather than Codex's inner sandbox. The
project workspace remains the only writable mount. The harness intentionally
keeps completed agent containers for inspection; remove old eval containers with
Docker when they are no longer needed.

The Codex grader runs at `gpt-5.5` with xhigh reasoning, receives the evidence
output, and has read-only access to the final workspace folder for file
inspection. A separate Codex analysis pass, also `gpt-5.5` xhigh, produces the
five-bullet run summary and testcase recommendations without assigning a score.
Agents are asked to write `AGENT_EVAL_NOTES.md` with public notes on approach,
assumptions, blockers, and skill improvements. The grader and analyzer may use
those notes and public stdout/stderr for process feedback, but should not rely
on hidden chain-of-thought.

Regrade existing runs against the current testcase rubric without rerunning the
agents:

```bash
python3 agent_skill_eval/harness.py --regrade-run-dir agent_skill_eval/runs/<timestamp>
```

Regrade requires the run's kept agent container to still exist. Older runs
created before container retention cannot be regraded because the agent-created
environment no longer exists. Regrade updates each run's existing `result.json`
so summaries and reports use the latest score. Timed-out runs are overwritten
with score 0 and stale evidence is cleared.

The default agents are:

- `codex-5.5-xhigh`
- `codex-5.5-low`
- `claude-opus-4.8-xhigh`
- `claude-sonnet-4.6-low`
