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

Run all configured agents:

```bash
python3 agent_skill_eval/harness.py --docker-oauth all
```

By default, the harness runs each selected testcase/agent pair 5 independent
times. Use `--runs-per-agent N` to change that count.

The agents under test are configured in `agent_skill_eval/agents.yaml`, not in
the harness. Use `--agents-file` to point at another YAML file.

Agent timeouts are configured per testcase with an `Agent timeout` line in
`testcase.md`, for example `- Agent timeout: 20 minutes.`.

Only the agent-under-test command runs in Docker. The harness mounts that
agent's copied testcase workspace at `/workspace` and does not mount the repo or
the user's home directory. Evidence commands and the Codex grader run locally
after the agent finishes.

Local evidence commands use the current Python process, with the repo root added
to `PYTHONPATH`. For this testcase, the local environment still needs `torch`
and `PyYAML` available so the evidence commands and agent config loading can
run.

Run one agent:

```bash
python3 agent_skill_eval/harness.py --agent codex-5.5-xhigh --docker-oauth codex
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
- `summary.csv` with score, duration, token usage, and cost when the CLI output
  exposes those fields
- `aggregate.csv` with avg/min/max score, duration, token usage, and cost per
  testcase/agent across repeated runs. The same aggregate table is printed at
  the end of the harness run.

The harness executes the evidence commands listed in each `testcase.md` before
grading. Those commands are testcase-specific; this first testcase runs the
generated NVFlare job and separately records the centralized baseline listed in
the testcase.

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
project workspace remains the only writable mount.

The Codex grader runs at `gpt-5.5` with xhigh reasoning, receives the evidence
output, and has read-only access to the final workspace folder for file
inspection. Agents are asked to write `AGENT_EVAL_NOTES.md` with public notes on
approach, assumptions, blockers, and skill improvements. The grader may use
those notes and public stdout/stderr for process feedback, but should not rely
on hidden chain-of-thought.

The default agents are:

- `codex-5.5-xhigh`
- `codex-5.5-low`
- `claude-opus-4.8-xhigh`
- `claude-sonnet-4.6-low`
