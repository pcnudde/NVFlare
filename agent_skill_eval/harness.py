#!/usr/bin/env python3
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

"""Minimal agent skill evaluation harness.

The harness intentionally keeps testcases human-readable. It copies a testcase
fixture, runs an agent against the prompt in testcase.md, collects evidence, and
asks Codex to grade the final workspace from the same Markdown rubric.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TESTCASE = REPO_ROOT / "agent_skill_eval/testcases/nvflare_basic_pytorch_to_sim"
DEFAULT_OUT_DIR = REPO_ROOT / "agent_skill_eval/runs"
DEFAULT_AGENTS_FILE = REPO_ROOT / "agent_skill_eval/agents.yaml"
GRADER_OUTPUT_SCHEMA = REPO_ROOT / "agent_skill_eval/grader_output_schema.json"
EVIDENCE_TIMEOUT_SECONDS = 600
GRADER_TIMEOUT_SECONDS = 600
DOCKER_AGENT_NETWORK = "bridge"
DOCKER_SECURITY_OPTS = ("seccomp=unconfined",)
DOCKER_CONTAINER_WORKDIR = "/workspace"
CLAUDE_KEYCHAIN_SERVICE = "Claude Code"

GRADER_COMMAND = [
    "codex",
    "--ask-for-approval",
    "never",
    "exec",
    "--model",
    "gpt-5.5",
    "-c",
    'model_reasoning_effort="xhigh"',
    "--sandbox",
    "read-only",
    "--ignore-user-config",
    "--ignore-rules",
    "--skip-git-repo-check",
    "--json",
    "--output-schema",
    "{schema_file}",
    "--output-last-message",
    "{grade_file}",
    "--cd",
    "{workdir}",
    "-",
]

TOKEN_KEYS = {
    "input_tokens",
    "output_tokens",
    "total_tokens",
    "prompt_tokens",
    "completion_tokens",
    "cache_creation_input_tokens",
    "cache_read_input_tokens",
    "cached_tokens",
    "reasoning_tokens",
}
COST_KEYS = {"cost_usd", "total_cost_usd"}


@dataclass
class CommandResult:
    command: list[str]
    returncode: int | None
    duration_seconds: float
    stdout: str
    stderr: str
    timed_out: bool = False

    def to_record(self, stdout_path: Path | None = None, stderr_path: Path | None = None) -> dict[str, Any]:
        return {
            "command": self.command,
            "returncode": self.returncode,
            "duration_seconds": round(self.duration_seconds, 3),
            "timed_out": self.timed_out,
            "stdout_path": str(stdout_path) if stdout_path else None,
            "stderr_path": str(stderr_path) if stderr_path else None,
            "stdout_tail": tail(self.stdout),
            "stderr_tail": tail(self.stderr),
            "token_usage": extract_usage(self.stdout + "\n" + self.stderr),
        }


@dataclass(frozen=True)
class DockerConfig:
    image: str
    env: tuple[str, ...]
    oauth_mounts: tuple[tuple[str, str], ...]
    oauth_setup_script: str | None


@dataclass(frozen=True)
class TestcaseConfig:
    id: str
    path: Path
    text: str
    agent_prompt: str
    agent_timeout_seconds: int
    docker_config: DockerConfig


def main() -> int:
    parser = argparse.ArgumentParser(description="Run agent skill testcase evaluations.")
    parser.add_argument(
        "--testcase",
        type=Path,
        action="append",
        dest="testcases",
        help="Path to testcase directory. Repeatable. Defaults to the first NVFlare testcase.",
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR, help="Directory for run artifacts.")
    parser.add_argument("--agents-file", type=Path, default=DEFAULT_AGENTS_FILE, help="YAML file with agents to run.")
    parser.add_argument("--agent", action="append", help="Agent id to run. Repeatable. Defaults to all.")
    parser.add_argument(
        "--runs-per-agent",
        type=int,
        default=5,
        help="Number of independent runs for each testcase/agent pair. Defaults to 5.",
    )
    parser.add_argument("--parallel", type=int, default=1, help="Number of testcase/agent runs to run concurrently.")
    parser.add_argument(
        "--docker-env",
        action="append",
        default=[],
        metavar="NAME[=VALUE]",
        help="Environment variable to pass into Docker. Repeatable. Use NAME to pass through from the host.",
    )
    parser.add_argument(
        "--docker-oauth",
        action="append",
        choices=["codex", "claude", "all"],
        default=[],
        help="Mount local OAuth auth state for selected agent CLI into the container. Repeatable.",
    )
    parser.add_argument(
        "--docker-claude-keychain",
        action="store_true",
        help="Read the Claude Code API key from macOS Keychain and pass it to Docker as ANTHROPIC_API_KEY.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print planned runs without invoking agents.")
    parser.add_argument("--list-agents", action="store_true", help="List configured agents and exit.")
    args = parser.parse_args()

    agents = select_agents(args.agent, load_agents(args.agents_file.resolve()))
    if args.list_agents:
        for agent in agents:
            print(f"{agent['id']}\t{agent['label']}")
        return 0

    if args.parallel < 1:
        raise SystemExit("--parallel must be >= 1")
    if args.runs_per_agent < 1:
        raise SystemExit("--runs-per-agent must be >= 1")

    configure_claude_keychain_env(args)
    testcases = load_testcases(args)
    if args.dry_run:
        for testcase in testcases:
            for agent in agents:
                for run_index in range(1, args.runs_per_agent + 1):
                    print(
                        f"would run {testcase.id}/{agent['id']}#{run_index} in {testcase.docker_config.image}: "
                        f"{' '.join(agent['command'])} "
                        f"(agent timeout {testcase.agent_timeout_seconds}s)"
                    )
        return 0

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    out_dir = (args.out_dir / run_id).resolve()
    out_dir.mkdir(parents=True, exist_ok=False)

    tasks = [
        (testcase, agent, run_index)
        for testcase in testcases
        for agent in agents
        for run_index in range(1, args.runs_per_agent + 1)
    ]
    summaries = run_eval_tasks(tasks, out_dir, args)

    if not args.dry_run:
        aggregate_rows = build_aggregate_rows(summaries)
        write_summary_csv(out_dir / "summary.csv", summaries)
        write_aggregate_csv(out_dir / "aggregate.csv", aggregate_rows)
        print(f"Wrote {out_dir / 'summary.csv'}")
        print(f"Wrote {out_dir / 'aggregate.csv'}")
        print_aggregate_table(aggregate_rows)
    return 0


def load_testcases(args: argparse.Namespace) -> list[TestcaseConfig]:
    testcase_paths = args.testcases or [DEFAULT_TESTCASE]
    configs = []
    used_ids: dict[str, int] = {}
    for testcase_path in testcase_paths:
        testcase_dir = testcase_path.resolve()
        testcase_md = testcase_dir / "testcase.md"
        initial_dir = testcase_dir / "initial"
        if not testcase_md.exists():
            raise SystemExit(f"Missing testcase.md: {testcase_md}")
        if not initial_dir.exists():
            raise SystemExit(f"Missing initial fixture directory: {initial_dir}")
        testcase_text = testcase_md.read_text()
        testcase_id = slugify(testcase_dir.name) or "testcase"
        if testcase_id in used_ids:
            used_ids[testcase_id] += 1
            testcase_id = f"{testcase_id}_{used_ids[testcase_id]}"
        else:
            used_ids[testcase_id] = 1
        configs.append(
            TestcaseConfig(
                id=testcase_id,
                path=testcase_dir,
                text=testcase_text,
                agent_prompt=build_agent_prompt(testcase_text),
                agent_timeout_seconds=extract_agent_timeout_seconds(testcase_text),
                docker_config=build_docker_config(args, testcase_text),
            )
        )
    return configs


def run_eval_tasks(
    tasks: list[tuple[TestcaseConfig, dict[str, Any], int]],
    out_dir: Path,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    if args.parallel == 1:
        return [
            run_agent_eval(
                agent=agent,
                testcase=testcase,
                run_index=run_index,
                out_dir=out_dir,
                agent_timeout=testcase.agent_timeout_seconds,
            )
            for testcase, agent, run_index in tasks
        ]

    summaries = []
    with ThreadPoolExecutor(max_workers=args.parallel) as executor:
        future_to_task = {
            executor.submit(
                run_agent_eval,
                agent=agent,
                testcase=testcase,
                run_index=run_index,
                out_dir=out_dir,
                agent_timeout=testcase.agent_timeout_seconds,
            ): (testcase, agent, run_index)
            for testcase, agent, run_index in tasks
        }
        for future in as_completed(future_to_task):
            testcase, agent, run_index = future_to_task[future]
            try:
                summaries.append(future.result())
            except Exception as e:  # noqa: BLE001
                summaries.append(harness_error_summary(testcase, agent, run_index, e))
    return sorted(summaries, key=lambda row: (row["testcase_id"], row["agent_id"], row["run_index"]))


def harness_error_summary(
    testcase: TestcaseConfig,
    agent: dict[str, Any],
    run_index: int,
    error: Exception,
) -> dict[str, Any]:
    return {
        "testcase_id": testcase.id,
        "agent_id": agent["id"],
        "agent_label": agent["label"],
        "run_index": run_index,
        "score": None,
        "score_before_caps": None,
        "agent_returncode": None,
        "agent_timed_out": None,
        "agent_duration_seconds": None,
        "total_duration_seconds": None,
        "agent_tokens": None,
        "grader_tokens": None,
        "agent_cost_usd": None,
        "grader_cost_usd": None,
        "summary": f"harness error: {error}",
        "process_observations": "[]",
        "skill_improvement_suggestions": "[]",
    }


def load_agents(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise SystemExit(f"Missing agents file: {path}")
    try:
        import yaml  # type: ignore[import-untyped]
    except ModuleNotFoundError:
        raise SystemExit("PyYAML is required to read agent_skill_eval/agents.yaml. Install it with `pip install PyYAML`.") from None
    data = yaml.safe_load(path.read_text())
    if not isinstance(data, dict) or not isinstance(data.get("agents"), list):
        raise SystemExit(f"Invalid agents file, expected top-level 'agents' list: {path}")
    return validate_agents(data["agents"], path)


def validate_agents(agents: list[Any], path: Path) -> list[dict[str, Any]]:
    validated = []
    seen = set()
    for index, agent in enumerate(agents, start=1):
        if not isinstance(agent, dict):
            raise SystemExit(f"Invalid agent #{index} in {path}: expected mapping")
        agent_id = agent.get("id")
        label = agent.get("label")
        command = agent.get("command")
        if not isinstance(agent_id, str) or not agent_id:
            raise SystemExit(f"Invalid agent #{index} in {path}: missing string id")
        if agent_id in seen:
            raise SystemExit(f"Duplicate agent id in {path}: {agent_id}")
        if not isinstance(label, str) or not label:
            raise SystemExit(f"Invalid agent {agent_id} in {path}: missing string label")
        if not isinstance(command, list) or not command or not all(isinstance(part, str) for part in command):
            raise SystemExit(f"Invalid agent {agent_id} in {path}: command must be a non-empty string list")
        seen.add(agent_id)
        validated.append({"id": agent_id, "label": label, "command": command})
    return validated


def select_agents(agent_ids: list[str] | None, agents: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not agent_ids:
        return agents
    by_id = {agent["id"]: agent for agent in agents}
    missing = [agent_id for agent_id in agent_ids if agent_id not in by_id]
    if missing:
        raise SystemExit(f"Unknown agent id(s): {', '.join(missing)}")
    return [by_id[agent_id] for agent_id in agent_ids]


def run_agent_eval(
    agent: dict[str, Any],
    testcase: TestcaseConfig,
    run_index: int,
    out_dir: Path,
    agent_timeout: int,
) -> dict[str, Any]:
    agent_dir = out_dir / testcase.id / agent["id"] / f"run_{run_index:02d}"
    workspace_dir = agent_dir / "workspace"
    baseline_workspace_dir = agent_dir / "baseline_workspace"
    logs_dir = agent_dir / "logs"
    logs_dir.mkdir(parents=True)
    shutil.copytree(testcase.path / "initial", workspace_dir)
    shutil.copytree(testcase.path / "initial", baseline_workspace_dir)
    workdir = infer_workdir(workspace_dir)
    baseline_workdir = infer_workdir(baseline_workspace_dir)
    env = build_run_env(agent_dir)
    baseline_evidence = collect_baseline_evidence(
        baseline_workdir, logs_dir, testcase.text, EVIDENCE_TIMEOUT_SECONDS, env
    )

    last_message_host = workdir / ".agent_eval_last_message.txt"
    placeholders = {
        "workdir": DOCKER_CONTAINER_WORKDIR,
        "last_message": f"{DOCKER_CONTAINER_WORKDIR}/.agent_eval_last_message.txt",
    }
    agent_command = expand_command(agent["command"], placeholders)
    agent_run_command = maybe_wrap_docker_command(
        command=agent_command,
        docker_config=testcase.docker_config,
        host_workdir=workdir,
        container_name=build_container_name(out_dir.name, testcase.id, agent["id"], run_index),
    )

    started = time.monotonic()
    agent_stdout = logs_dir / "agent_stdout.txt"
    agent_stderr = logs_dir / "agent_stderr.txt"
    agent_result = run_command(
        agent_run_command,
        cwd=REPO_ROOT,
        stdin=testcase.agent_prompt,
        timeout_seconds=agent_timeout,
        env=env,
        stdout_path=agent_stdout,
        stderr_path=agent_stderr,
    )
    agent_duration = time.monotonic() - started

    evidence = collect_evidence(workdir, logs_dir, testcase.text, EVIDENCE_TIMEOUT_SECONDS, env)
    grade = grade_with_codex(
        workdir=workdir,
        logs_dir=logs_dir,
        testcase_text=testcase.text,
        baseline_evidence=baseline_evidence,
        evidence=evidence,
        agent_public_context=build_agent_public_context(agent_result, last_message_host),
        env=env,
    )

    total_duration = time.monotonic() - started
    result = {
        "agent": {"id": agent["id"], "label": agent["label"], "command": agent_command},
        "agent_run_command": agent_run_command,
        "testcase_id": testcase.id,
        "run_index": run_index,
        "testcase": str(testcase.path),
        "workdir": str(workdir),
        "baseline_workdir": str(baseline_workdir),
        "duration_seconds": round(total_duration, 3),
        "agent_duration_seconds": round(agent_duration, 3),
        "agent_result": agent_result.to_record(agent_stdout, agent_stderr),
        "baseline_evidence": baseline_evidence,
        "evidence": evidence,
        "grade": grade,
        "token_usage": {
            "agent": agent_result.to_record()["token_usage"],
            "grader": grade.get("grader_result", {}).get("token_usage", {}),
        },
    }
    write_json(agent_dir / "result.json", result)
    return flatten_summary(result)


def infer_workdir(workspace_dir: Path) -> Path:
    children = [path for path in workspace_dir.iterdir() if path.is_dir()]
    if len(children) == 1:
        return children[0]
    return workspace_dir


def build_docker_config(args: argparse.Namespace, testcase_text: str) -> DockerConfig:
    image = extract_docker_image(testcase_text)
    if not image:
        raise SystemExit(
            "No Docker image configured. Add a line like this to testcase.md:\n"
            "  - Docker image: `nvflare-agent-eval:basic`"
        )
    warn_for_unset_docker_env(args.docker_env)
    oauth_mounts, oauth_setup_script = build_oauth_config(args.docker_oauth)
    return DockerConfig(
        image=image,
        env=tuple(args.docker_env),
        oauth_mounts=oauth_mounts,
        oauth_setup_script=oauth_setup_script,
    )


def build_oauth_config(requested: list[str]) -> tuple[tuple[tuple[str, str], ...], str | None]:
    providers = set(requested)
    if "all" in providers:
        providers.update(["codex", "claude"])

    mounts: list[tuple[str, str]] = []
    setup_lines = ["set -e"]
    home = Path.home()

    if "codex" in providers:
        codex_auth = home / ".codex/auth.json"
        if codex_auth.exists():
            mounts.append((str(codex_auth), "/tmp/agent_eval_oauth/codex/auth.json"))
            setup_lines.extend(
                [
                    'mkdir -p "$HOME/.codex"',
                    'cp /tmp/agent_eval_oauth/codex/auth.json "$HOME/.codex/auth.json"',
                ]
            )
        else:
            print(f"warning: Codex OAuth file not found: {codex_auth}", file=sys.stderr)

    if "claude" in providers:
        claude_credentials = home / ".claude/.credentials.json"
        claude_settings = home / ".claude/settings.json"
        claude_json = home / ".claude.json"
        if claude_credentials.exists():
            mounts.append((str(claude_credentials), "/tmp/agent_eval_oauth/claude/.credentials.json"))
            setup_lines.extend(
                [
                    'mkdir -p "$HOME/.claude"',
                    'cp /tmp/agent_eval_oauth/claude/.credentials.json "$HOME/.claude/.credentials.json"',
                ]
            )
        else:
            print(f"warning: Claude OAuth file not found: {claude_credentials}", file=sys.stderr)
        if claude_settings.exists():
            mounts.append((str(claude_settings), "/tmp/agent_eval_oauth/claude/settings.json"))
            setup_lines.extend(
                [
                    'mkdir -p "$HOME/.claude"',
                    'cp /tmp/agent_eval_oauth/claude/settings.json "$HOME/.claude/settings.json"',
                ]
            )
        if claude_json.exists():
            mounts.append((str(claude_json), "/tmp/agent_eval_oauth/claude/.claude.json"))
            setup_lines.append('cp /tmp/agent_eval_oauth/claude/.claude.json "$HOME/.claude.json"')

    if not mounts:
        return (), None
    setup_lines.append('exec "$@"')
    return tuple(mounts), "\n".join(setup_lines)


def configure_claude_keychain_env(args: argparse.Namespace) -> None:
    if not args.docker_claude_keychain:
        return

    if not is_macos():
        raise SystemExit("--docker-claude-keychain is only supported on macOS")

    completed = subprocess.run(
        ["security", "find-generic-password", "-s", CLAUDE_KEYCHAIN_SERVICE, "-w"],
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        stderr = completed.stderr.strip() or completed.stdout.strip()
        raise SystemExit(
            f"Could not read Claude keychain service {CLAUDE_KEYCHAIN_SERVICE!r}. "
            f"Run outside the command sandbox and approve the macOS prompt. {stderr}"
        )

    key = completed.stdout.strip()
    if not key:
        raise SystemExit(f"Claude keychain service {CLAUDE_KEYCHAIN_SERVICE!r} returned an empty key")

    os.environ["ANTHROPIC_API_KEY"] = key
    if "ANTHROPIC_API_KEY" not in [env_spec.split("=", 1)[0] for env_spec in args.docker_env]:
        args.docker_env.append("ANTHROPIC_API_KEY")


def is_macos() -> bool:
    return sys.platform == "darwin"


def warn_for_unset_docker_env(env_specs: list[str]) -> None:
    for env_spec in env_specs:
        if "=" not in env_spec and not os.environ.get(env_spec):
            print(f"warning: Docker env passthrough {env_spec} is not set on the host", file=sys.stderr)


def extract_docker_image(testcase_text: str) -> str | None:
    match = re.search(r"^\s*-\s*Docker image:\s*`([^`]+)`", testcase_text, flags=re.MULTILINE)
    if match:
        return match.group(1).strip()
    match = re.search(r"^\s*-\s*Docker image:\s*(\S+)", testcase_text, flags=re.MULTILINE)
    return match.group(1).strip().rstrip(".") if match else None


def extract_agent_timeout_seconds(testcase_text: str) -> int:
    match = re.search(r"^\s*-\s*Agent timeout:\s*([^\n]+)$", testcase_text, flags=re.MULTILINE)
    if not match:
        return 1800
    return parse_duration_seconds(match.group(1).strip().rstrip("."))


def parse_duration_seconds(text: str) -> int:
    match = re.fullmatch(r"(\d+)\s*(seconds?|secs?|s|minutes?|mins?|m)?", text, flags=re.IGNORECASE)
    if not match:
        raise SystemExit(f"Invalid agent timeout duration: {text}")
    value = int(match.group(1))
    unit = (match.group(2) or "seconds").lower()
    if unit in {"m", "min", "mins", "minute", "minutes"}:
        value *= 60
    if value < 1:
        raise SystemExit(f"Agent timeout must be positive: {text}")
    return value


def build_run_env(run_dir: Path) -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{REPO_ROOT}{os.pathsep}{env['PYTHONPATH']}" if env.get("PYTHONPATH") else str(REPO_ROOT)

    bin_dir = run_dir / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)
    for name in ["python", "python3"]:
        shim = bin_dir / name
        python_target = Path(sys.executable)
        shim.write_text(f'#!/bin/sh\nexec "{python_target}" "$@"\n')
        shim.chmod(0o755)
    env["PATH"] = f"{bin_dir}{os.pathsep}{env.get('PATH', '')}"
    return env


def build_agent_prompt(testcase_text: str) -> str:
    prompt = extract_section_code_block(testcase_text, "Prompt") or ""
    if not prompt:
        raise SystemExit("Could not find a prompt code block in testcase.md")
    return (
        "You are being evaluated on an agent skill testcase.\n"
        "Work only in the current project folder. Implement the user's request. "
        "Do not ask clarifying questions; make reasonable assumptions and finish the task.\n"
        "Before finishing, write AGENT_EVAL_NOTES.md with concise public notes on your approach, "
        "assumptions, blockers, and any skill improvements that would have helped. Do not include hidden "
        "chain-of-thought; write only a brief engineering summary.\n\n"
        f"User prompt:\n{prompt.strip()}\n"
    )


def extract_section_code_block(markdown: str, section: str) -> str | None:
    match = re.search(rf"^## {re.escape(section)}\s*$", markdown, flags=re.MULTILINE)
    if not match:
        return None
    next_section = re.search(r"^## .*$", markdown[match.end() :], flags=re.MULTILINE)
    end = match.end() + next_section.start() if next_section else len(markdown)
    section_text = markdown[match.end() : end]
    code = re.search(r"```(?:text|bash)?\n(.*?)```", section_text, flags=re.DOTALL)
    if code:
        return code.group(1).strip()
    return section_text.strip()


def collect_evidence(
    workdir: Path,
    logs_dir: Path,
    testcase_text: str,
    timeout_seconds: int,
    env: dict[str, str],
) -> list[dict[str, Any]]:
    evidence_dir = logs_dir / "evidence"
    evidence_dir.mkdir()
    commands = parse_evidence_commands(testcase_text, timeout_seconds)
    records = []
    for command_id, command, command_timeout, original in commands:
        if runs_job_py(command):
            records.append(
                collect_nvflare_version_evidence(
                    evidence_dir=evidence_dir,
                    command_id=f"{command_id}_nvflare_version",
                    job_command=command,
                    workdir=workdir,
                    timeout_seconds=min(command_timeout, 60),
                    env=env,
                )
            )
        result = run_command(
            command,
            cwd=workdir,
            stdin=None,
            timeout_seconds=command_timeout,
            env=env,
        )
        stdout_path = evidence_dir / f"{command_id}.stdout.txt"
        stderr_path = evidence_dir / f"{command_id}.stderr.txt"
        write_text(stdout_path, result.stdout)
        write_text(stderr_path, result.stderr)
        record = result.to_record(stdout_path, stderr_path)
        record["id"] = command_id
        record["original"] = original
        record["run_command"] = command
        records.append(record)
    return records


def collect_baseline_evidence(
    workdir: Path,
    logs_dir: Path,
    testcase_text: str,
    timeout_seconds: int,
    env: dict[str, str],
) -> list[dict[str, Any]]:
    evidence_dir = logs_dir / "baseline_evidence"
    commands = parse_evidence_commands(testcase_text, timeout_seconds, section="Baseline Evidence To Collect")
    if not commands:
        return []
    evidence_dir.mkdir()
    records = []
    for command_id, command, command_timeout, original in commands:
        result = run_command(
            command,
            cwd=workdir,
            stdin=None,
            timeout_seconds=command_timeout,
            env=env,
        )
        stdout_path = evidence_dir / f"{command_id}.stdout.txt"
        stderr_path = evidence_dir / f"{command_id}.stderr.txt"
        write_text(stdout_path, result.stdout)
        write_text(stderr_path, result.stderr)
        record = result.to_record(stdout_path, stderr_path)
        record["id"] = command_id
        record["original"] = original
        record["run_command"] = command
        records.append(record)
    return records


def runs_job_py(command: list[str]) -> bool:
    if not command:
        return False
    executable = Path(command[0]).name
    if executable == "job.py":
        return True
    return executable.startswith("python") and len(command) >= 2 and Path(command[1]).name == "job.py"


def collect_nvflare_version_evidence(
    evidence_dir: Path,
    command_id: str,
    job_command: list[str],
    workdir: Path,
    timeout_seconds: int,
    env: dict[str, str],
) -> dict[str, Any]:
    command = nvflare_version_command(job_command)
    result = run_command(
        command,
        cwd=workdir,
        stdin=None,
        timeout_seconds=timeout_seconds,
        env=env,
    )
    stdout_path = evidence_dir / f"{command_id}.stdout.txt"
    stderr_path = evidence_dir / f"{command_id}.stderr.txt"
    write_text(stdout_path, result.stdout)
    write_text(stderr_path, result.stderr)
    record = result.to_record(stdout_path, stderr_path)
    record["id"] = command_id
    record["original"] = f"nvflare version probe before: {' '.join(job_command)}"
    record["run_command"] = command
    record["nvflare_version_probe"] = parse_nvflare_version_probe(result.stdout)
    return record


def nvflare_version_command(job_command: list[str]) -> list[str]:
    python_executable = job_command[0] if job_command else "python"
    probe = (
        "import importlib.metadata as md\n"
        "import json\n"
        "import sys\n"
        "try:\n"
        "    import nvflare\n"
        "    try:\n"
        "        distribution_version = md.version('nvflare')\n"
        "    except md.PackageNotFoundError:\n"
        "        distribution_version = None\n"
        "    print(json.dumps({\n"
        "        'nvflare_version': getattr(nvflare, '__version__', None) or distribution_version,\n"
        "        'distribution_version': distribution_version,\n"
        "        'module_file': getattr(nvflare, '__file__', None),\n"
        "        'python': sys.executable,\n"
        "    }, sort_keys=True))\n"
        "except Exception as e:\n"
        "    print(json.dumps({\n"
        "        'error': f'{type(e).__name__}: {e}',\n"
        "        'python': sys.executable,\n"
        "    }, sort_keys=True))\n"
    )
    return [python_executable, "-c", probe]


def parse_nvflare_version_probe(stdout: str) -> dict[str, Any] | None:
    for line in stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return None


def maybe_wrap_docker_command(
    command: list[str],
    docker_config: DockerConfig,
    host_workdir: Path,
    container_name: str,
) -> list[str]:
    docker_command = [
        "docker",
        "run",
        "--rm",
        "-i",
        "--name",
        container_name,
        "--network",
        DOCKER_AGENT_NETWORK,
    ]
    for security_opt in DOCKER_SECURITY_OPTS:
        docker_command.extend(["--security-opt", security_opt])
    docker_command.extend(
        [
            "--mount",
            f"type=bind,src={host_workdir},dst={DOCKER_CONTAINER_WORKDIR}",
            "-w",
            DOCKER_CONTAINER_WORKDIR,
        ]
    )
    for env_spec in docker_config.env:
        docker_command.extend(["--env", env_spec])
    for source, target in docker_config.oauth_mounts:
        docker_command.extend(["--mount", f"type=bind,src={source},dst={target},readonly"])
    if docker_config.oauth_setup_script:
        return docker_command + [
            docker_config.image,
            "sh",
            "-lc",
            docker_config.oauth_setup_script,
            "agent-eval-command",
        ] + command
    return docker_command + [docker_config.image] + command


def build_container_name(run_id: str, testcase_id: str, agent_id: str, run_index: int) -> str:
    raw = f"agent-eval-{run_id}-{testcase_id}-{agent_id}-run-{run_index:02d}"
    name = re.sub(r"[^A-Za-z0-9_.-]+", "-", raw).strip("-")
    if not name or not re.match(r"^[A-Za-z0-9]", name):
        name = f"agent-eval-{name}"
    return name[:128]


def docker_container_name_from_command(command: list[str]) -> str | None:
    if len(command) < 4 or command[0:2] != ["docker", "run"]:
        return None
    for index, part in enumerate(command):
        if part == "--name" and index + 1 < len(command):
            return command[index + 1]
        if part.startswith("--name="):
            return part.split("=", 1)[1]
    return None


def parse_evidence_commands(
    testcase_text: str,
    default_timeout: int,
    section: str = "Evidence To Collect",
) -> list[tuple[str, list[str], int, str]]:
    block = extract_section_code_block(testcase_text, section) or ""
    commands = []
    for index, raw_line in enumerate(block.splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            parts = shlex.split(line)
        except ValueError as e:
            raise SystemExit(f"Could not parse evidence command {index}: {line}\n{e}") from e
        if not parts:
            continue

        timeout_seconds = default_timeout
        if parts[0] == "timeout" and len(parts) >= 3:
            try:
                timeout_seconds = int(parts[1])
            except ValueError as e:
                raise SystemExit(f"Invalid timeout in evidence command {index}: {line}") from e
            parts = parts[2:]

        command_id = slugify(" ".join(parts[:3])) or f"evidence_{index}"
        commands.append((f"{index:02d}_{command_id}", parts, timeout_seconds, line))
    return commands


def slugify(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", text).strip("_").lower()


def grade_with_codex(
    workdir: Path,
    logs_dir: Path,
    testcase_text: str,
    baseline_evidence: list[dict[str, Any]],
    evidence: list[dict[str, Any]],
    agent_public_context: dict[str, str | None],
    env: dict[str, str],
) -> dict[str, Any]:
    grade_file = logs_dir / "grade.json"
    grader_stdout = logs_dir / "grader_stdout.txt"
    grader_stderr = logs_dir / "grader_stderr.txt"
    prompt = build_grader_prompt(testcase_text, baseline_evidence, evidence, agent_public_context, str(workdir))
    command = expand_command(
        GRADER_COMMAND,
        {
            "workdir": str(workdir),
            "schema_file": str(GRADER_OUTPUT_SCHEMA),
            "grade_file": str(grade_file),
        },
    )
    result = run_command(
        command,
        cwd=workdir,
        stdin=prompt,
        timeout_seconds=GRADER_TIMEOUT_SECONDS,
        env=env,
        stdout_path=grader_stdout,
        stderr_path=grader_stderr,
    )

    parsed_grade = parse_grade_file(grade_file)
    return {
        "parsed": parsed_grade,
        "grade_file": str(grade_file) if grade_file.exists() else None,
        "grader_result": result.to_record(grader_stdout, grader_stderr),
        "grader_run_command": command,
    }


def build_agent_public_context(agent_result: CommandResult, last_message: Path) -> dict[str, str | None]:
    return {
        "agent_stdout_tail": tail(agent_result.stdout),
        "agent_stderr_tail": tail(agent_result.stderr),
        "agent_last_message": last_message.read_text(errors="replace") if last_message.exists() else None,
    }


def build_grader_prompt(
    testcase_text: str,
    baseline_evidence: list[dict[str, Any]],
    evidence: list[dict[str, Any]],
    agent_public_context: dict[str, str | None],
    workdir: str,
) -> str:
    baseline_evidence_text = json.dumps(baseline_evidence, indent=2)
    evidence_text = json.dumps(evidence, indent=2)
    public_context_text = json.dumps(agent_public_context, indent=2)
    return (
        "Grade this agent skill evaluation run. Use the rubric and score caps in the testcase. "
        "Return JSON only, matching the provided schema. Score must be an integer from 0 to 100.\n"
        f"You are running with read-only access to the final workspace at: {workdir}\n"
        "Inspect the final workspace files directly before assigning partial credit. "
        "The harness has already executed the evidence commands; use those command results for runtime behavior.\n"
        "For process notes, use only public artifacts such as AGENT_EVAL_NOTES.md, the agent final message, "
        "and stdout/stderr. Do not request or infer hidden chain-of-thought.\n\n"
        "Testcase:\n"
        f"{testcase_text}\n\n"
        "Baseline evidence from the unmodified starting workspace:\n"
        f"{baseline_evidence_text}\n\n"
        "Collected evidence:\n"
        f"{evidence_text}\n\n"
        "Agent public output:\n"
        f"{public_context_text}\n"
    )


def run_command(
    command: list[str],
    cwd: Path,
    stdin: str | None,
    timeout_seconds: int,
    env: dict[str, str],
    stdout_path: Path | None = None,
    stderr_path: Path | None = None,
) -> CommandResult:
    if stdout_path or stderr_path:
        return run_command_streaming(command, cwd, stdin, timeout_seconds, env, stdout_path, stderr_path)

    started = time.monotonic()
    try:
        completed = subprocess.run(
            command,
            input=stdin,
            text=True,
            cwd=str(cwd),
            env=env,
            capture_output=True,
            timeout=timeout_seconds,
            check=False,
        )
        return CommandResult(
            command=command,
            returncode=completed.returncode,
            duration_seconds=time.monotonic() - started,
            stdout=completed.stdout,
            stderr=completed.stderr,
        )
    except subprocess.TimeoutExpired as exc:
        return CommandResult(
            command=command,
            returncode=None,
            duration_seconds=time.monotonic() - started,
            stdout=ensure_text(exc.stdout),
            stderr=ensure_text(exc.stderr),
            timed_out=True,
        )


def run_command_streaming(
    command: list[str],
    cwd: Path,
    stdin: str | None,
    timeout_seconds: int,
    env: dict[str, str],
    stdout_path: Path | None,
    stderr_path: Path | None,
) -> CommandResult:
    started = time.monotonic()
    stdout_chunks: list[str] = []
    stderr_chunks: list[str] = []

    if stdout_path:
        stdout_path.parent.mkdir(parents=True, exist_ok=True)
    if stderr_path:
        stderr_path.parent.mkdir(parents=True, exist_ok=True)

    stdout_file = stdout_path.open("w") if stdout_path else None
    stderr_file = stderr_path.open("w") if stderr_path else None
    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE if stdin is not None else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=str(cwd),
        env=env,
        bufsize=1,
    )

    def reader(stream: Any, chunks: list[str], output_file: Any) -> None:
        try:
            for line in iter(stream.readline, ""):
                chunks.append(line)
                if output_file:
                    output_file.write(line)
                    output_file.flush()
        finally:
            stream.close()

    stdout_thread = threading.Thread(target=reader, args=(process.stdout, stdout_chunks, stdout_file), daemon=True)
    stderr_thread = threading.Thread(target=reader, args=(process.stderr, stderr_chunks, stderr_file), daemon=True)
    stdout_thread.start()
    stderr_thread.start()

    if stdin is not None and process.stdin is not None:
        try:
            process.stdin.write(stdin)
            process.stdin.close()
        except BrokenPipeError:
            pass

    timed_out = False
    try:
        returncode = process.wait(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        timed_out = True
        stop_docker_container(docker_container_name_from_command(command), env)
        process.kill()
        returncode = None
        process.wait()

    stdout_thread.join(timeout=5)
    stderr_thread.join(timeout=5)
    if stdout_file:
        stdout_file.close()
    if stderr_file:
        stderr_file.close()

    return CommandResult(
        command=command,
        returncode=returncode,
        duration_seconds=time.monotonic() - started,
        stdout="".join(stdout_chunks),
        stderr="".join(stderr_chunks),
        timed_out=timed_out,
    )


def stop_docker_container(container_name: str | None, env: dict[str, str]) -> None:
    if not container_name:
        return
    try:
        subprocess.run(
            ["docker", "stop", container_name],
            text=True,
            cwd=str(REPO_ROOT),
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=20,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        pass


def expand_command(command: list[str], placeholders: dict[str, str]) -> list[str]:
    return [part.format(**placeholders) for part in command]


def parse_grade_file(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    text = path.read_text(errors="replace").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            return None
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return None


def iter_json_values(text: str) -> list[Any]:
    values = []
    stripped = text.strip()
    if stripped:
        try:
            values.append(json.loads(stripped))
        except json.JSONDecodeError:
            pass
    for line in text.splitlines():
        line = line.strip()
        if not line or line[0] not in "[{":
            continue
        try:
            values.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return values


def extract_usage(text: str) -> dict[str, Any]:
    metrics: dict[str, float] = {}
    for value in iter_json_values(text):
        collect_usage_metrics(value, metrics)
    result: dict[str, Any] = {key: int(value) if float(value).is_integer() else value for key, value in metrics.items()}
    if "total_tokens" not in result:
        token_total = 0
        for key in [
            "input_tokens",
            "output_tokens",
            "prompt_tokens",
            "completion_tokens",
            "cache_creation_input_tokens",
            "cache_read_input_tokens",
            "cached_tokens",
            "reasoning_tokens",
        ]:
            value = result.get(key)
            if isinstance(value, (int, float)):
                token_total += value
        if token_total:
            result["total_tokens_estimate"] = int(token_total)
    return result


def collect_usage_metrics(value: Any, metrics: dict[str, float]) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if key in TOKEN_KEYS | COST_KEYS and isinstance(item, (int, float)):
                metrics[key] = max(metrics.get(key, 0), float(item))
            collect_usage_metrics(item, metrics)
    elif isinstance(value, list):
        for item in value:
            collect_usage_metrics(item, metrics)


def flatten_summary(result: dict[str, Any]) -> dict[str, Any]:
    grade = result.get("grade", {}).get("parsed") or {}
    agent_usage = result.get("token_usage", {}).get("agent", {})
    grader_usage = result.get("token_usage", {}).get("grader", {})
    return {
        "testcase_id": result["testcase_id"],
        "agent_id": result["agent"]["id"],
        "agent_label": result["agent"]["label"],
        "run_index": result["run_index"],
        "score": grade.get("score"),
        "score_before_caps": grade.get("score_before_caps"),
        "agent_returncode": result["agent_result"]["returncode"],
        "agent_timed_out": result["agent_result"]["timed_out"],
        "agent_duration_seconds": result["agent_duration_seconds"],
        "total_duration_seconds": result["duration_seconds"],
        "agent_tokens": token_total(agent_usage),
        "grader_tokens": token_total(grader_usage),
        "agent_cost_usd": cost_value(agent_usage),
        "grader_cost_usd": cost_value(grader_usage),
        "summary": grade.get("summary"),
        "process_observations": json.dumps(grade.get("process_observations", [])),
        "skill_improvement_suggestions": json.dumps(grade.get("skill_improvement_suggestions", [])),
    }


def token_total(usage: dict[str, Any]) -> int | None:
    for key in ["total_tokens", "total_tokens_estimate"]:
        value = usage.get(key)
        if isinstance(value, int):
            return value
    return None


def cost_value(usage: dict[str, Any]) -> float | None:
    for key in ["total_cost_usd", "cost_usd"]:
        value = usage.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    return None


def write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "testcase_id",
        "agent_id",
        "agent_label",
        "run_index",
        "score",
        "score_before_caps",
        "agent_returncode",
        "agent_timed_out",
        "agent_duration_seconds",
        "total_duration_seconds",
        "agent_tokens",
        "grader_tokens",
        "agent_cost_usd",
        "grader_cost_usd",
        "summary",
        "process_observations",
        "skill_improvement_suggestions",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def build_aggregate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = (row["testcase_id"], row["agent_id"], row["agent_label"])
        grouped.setdefault(key, []).append(row)

    aggregate_rows = []
    for (testcase_id, agent_id, agent_label), group_rows in sorted(grouped.items()):
        row: dict[str, Any] = {
            "testcase_id": testcase_id,
            "agent_id": agent_id,
            "agent_label": agent_label,
            "runs": len(group_rows),
            "scored_runs": len(numeric_values(group_rows, "score")),
            "agent_failed_runs": sum(
                1
                for item in group_rows
                if item.get("score") is None
                or item.get("agent_timed_out")
                or item.get("agent_returncode") not in (0, None)
            ),
        }
        for field in [
            "score",
            "score_before_caps",
            "agent_duration_seconds",
            "total_duration_seconds",
            "agent_tokens",
            "grader_tokens",
            "agent_cost_usd",
            "grader_cost_usd",
        ]:
            add_aggregate_stats(row, field, numeric_values(group_rows, field))
        aggregate_rows.append(row)
    return aggregate_rows


def add_aggregate_stats(row: dict[str, Any], field: str, values: list[float]) -> None:
    if not values:
        row[f"{field}_avg"] = None
        row[f"{field}_min"] = None
        row[f"{field}_max"] = None
        return
    row[f"{field}_avg"] = round(sum(values) / len(values), 3)
    row[f"{field}_min"] = round(min(values), 3)
    row[f"{field}_max"] = round(max(values), 3)


def numeric_values(rows: list[dict[str, Any]], field: str) -> list[float]:
    values = []
    for row in rows:
        value = row.get(field)
        if isinstance(value, bool) or value is None or value == "":
            continue
        if isinstance(value, (int, float)):
            values.append(float(value))
            continue
        try:
            values.append(float(value))
        except (TypeError, ValueError):
            continue
    return values


def write_aggregate_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "testcase_id",
        "agent_id",
        "agent_label",
        "runs",
        "scored_runs",
        "agent_failed_runs",
        "score_avg",
        "score_min",
        "score_max",
        "score_before_caps_avg",
        "score_before_caps_min",
        "score_before_caps_max",
        "agent_duration_seconds_avg",
        "agent_duration_seconds_min",
        "agent_duration_seconds_max",
        "total_duration_seconds_avg",
        "total_duration_seconds_min",
        "total_duration_seconds_max",
        "agent_tokens_avg",
        "agent_tokens_min",
        "agent_tokens_max",
        "grader_tokens_avg",
        "grader_tokens_min",
        "grader_tokens_max",
        "agent_cost_usd_avg",
        "agent_cost_usd_min",
        "agent_cost_usd_max",
        "grader_cost_usd_avg",
        "grader_cost_usd_min",
        "grader_cost_usd_max",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def print_aggregate_table(rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    headers = [
        "testcase",
        "agent",
        "runs",
        "score avg/min/max",
        "agent sec avg/min/max",
        "tokens avg/min/max",
    ]
    table = [
        [
            row["testcase_id"],
            row["agent_id"],
            str(row["runs"]),
            format_triplet(row, "score"),
            format_triplet(row, "agent_duration_seconds"),
            format_triplet(row, "agent_tokens"),
        ]
        for row in rows
    ]
    widths = [
        max(len(headers[index]), *(len(item[index]) for item in table))
        for index in range(len(headers))
    ]
    print("\nAggregate results:")
    print("  ".join(headers[index].ljust(widths[index]) for index in range(len(headers))))
    print("  ".join("-" * width for width in widths))
    for item in table:
        print("  ".join(item[index].ljust(widths[index]) for index in range(len(item))))


def format_triplet(row: dict[str, Any], field: str) -> str:
    values = [row.get(f"{field}_avg"), row.get(f"{field}_min"), row.get(f"{field}_max")]
    if all(value is None for value in values):
        return ""
    return "/".join("" if value is None else format_number(value) for value in values)


def format_number(value: Any) -> str:
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True))


def write_text(path: Path, text: str) -> None:
    path.write_text(text or "")


def ensure_text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode(errors="replace")
    return value


def tail(text: str, max_chars: int = 4000) -> str:
    if len(text) <= max_chars:
        return text
    return text[-max_chars:]


if __name__ == "__main__":
    raise SystemExit(main())
