# Agent Skill Evaluation Testcases

## Recommended Format

Use one human-readable Markdown testcase plus a literal initial workspace
fixture:

```text
agent_skill_eval/
└── testcases/
    └── nvflare_basic_pytorch_to_sim/
        ├── testcase.md
        └── initial/
            └── my_project/
                ├── train.py
                ├── model.py
                └── requirements.txt
```

This is intentionally simple. The testcase should read like something a person
could use to grade an agent run, and Codex can use the same rubric to assign the
score.

## Harness Convention

The harness should:

1. Copy the testcase `initial/` folder to a clean temporary workspace.
2. Start the agent in Docker with only the copied project folder mounted, using
   the Docker image named by the testcase.
3. Give the agent the prompt from `testcase.md`.
4. Collect the evidence commands listed in `testcase.md` locally.
5. Ask local Codex to grade the final workspace using the rubric and score caps in
   `testcase.md`.

The agents under test are listed in `agent_skill_eval/agents.yaml`. The harness
uses the testcase Docker image only for the agent-under-test command; it does
not mount the repo or the user's home directory into that container. Evidence
commands and the Codex grader run locally after the agent finishes. The local
evidence environment must have whatever dependencies the testcase evidence
commands require.

The default Docker security option is `seccomp=unconfined` so Codex's
`workspace-write` sandbox can create Linux namespaces inside the container. This
does not add extra workspace mounts; the copied project folder remains the only
writable project path.

Agent timeouts are testcase-specific. Put a human-readable line such as
`- Agent timeout: 20 minutes.` in the testcase environment section. The harness
default is 30 minutes when the line is omitted.

The evidence commands are testcase-specific and executed by the harness, not
left to the grader. Some testcases may run a generated job; others may run a
CLI command, unit test, validator, or no runtime command at all. The grader
receives the command outputs and gets read-only access to the final workspace so
it can inspect files directly before assigning partial credit.

Agents should also write `AGENT_EVAL_NOTES.md` before finishing. This is a
public engineering note with approach, assumptions, blockers, and skill
improvement ideas. The grader can use those notes and public stdout/stderr to
produce process observations, but should not ask for or score hidden
chain-of-thought.

The minimal harness is `agent_skill_eval/harness.py`:

```bash
docker build -f agent_skill_eval/docker/basic/Dockerfile -t nvflare-agent-eval:basic .
python3 agent_skill_eval/harness.py --parallel 4 --docker-oauth all
```

It writes per-agent workspaces, logs, evidence, grades, token usage, duration,
process observations, skill-improvement suggestions, and a `summary.csv` under
`agent_skill_eval/runs/<timestamp>/`. Each testcase/agent pair has its own
subdirectory, so repeated `--testcase` values and `--parallel` runs do not share
workspaces. The Codex grader uses `gpt-5.5` with xhigh reasoning.

There is no separate `testcase_eval.yaml` for now. A structured schema can be
added later if the harness needs indexing, dashboards, or strict validation, but
the first version should optimize for easy authoring and review.

## Testcase Markdown Shape

Each `testcase.md` should use these sections:

```text
# Title

Difficulty: Basic | Intermediate | Advanced

## Starting Point
Folder to copy, files present, and environment assumptions.

## Prompt
The exact prompt given to the agent.

## Expected Outcome
Short description of what a good solution produces.

## Grading Rubric
Score from 0 to 100 with a small number of point buckets.

## Score Caps
Maximum scores for major failures.

## Evidence To Collect
Commands and files Codex should inspect before grading.
```

Keep the rubric compact. Prefer 5 to 7 point buckets over many tiny criteria.
The goal is consistent Codex grading, not a large checklist that is hard to
author and review.

## First Testcase

The first concrete testcase is:

```text
agent_skill_eval/testcases/nvflare_basic_pytorch_to_sim/testcase.md
```

It evaluates whether an agent can convert a centralized PyTorch synthetic image
training script into an NVFlare simulation with 3 clients and 5 rounds.
