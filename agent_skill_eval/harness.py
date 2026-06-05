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
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TESTCASE = REPO_ROOT / "agent_skill_eval/testcases/nvflare_basic_pytorch_to_sim"
DEFAULT_OUT_DIR = REPO_ROOT / "agent_skill_eval/runs"
DEFAULT_AGENTS_FILE = REPO_ROOT / "agent_skill_eval/agents.yaml"
DEFAULT_MODEL_COSTS_FILE = REPO_ROOT / "agent_skill_eval/model_costs.yaml"
GRADER_OUTPUT_SCHEMA = REPO_ROOT / "agent_skill_eval/grader_output_schema.json"
ANALYSIS_OUTPUT_SCHEMA = REPO_ROOT / "agent_skill_eval/analysis_output_schema.json"
EVIDENCE_TIMEOUT_SECONDS = 600
GRADER_TIMEOUT_SECONDS = 600
DOCKER_AGENT_NETWORK = "bridge"
DOCKER_SECURITY_OPTS = ("seccomp=unconfined",)
DOCKER_CONTAINER_WORKDIR = "/workspace"
CLAUDE_KEYCHAIN_SERVICE = "Claude Code"
HERMES_NVIDIA_KEYCHAIN_SERVICE = "nvidia-inference-hub-api-key"

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
    "cached_input_tokens",
    "cached_tokens",
    "reasoning_output_tokens",
    "reasoning_tokens",
}
COST_KEYS = {"cost_usd", "total_cost_usd"}
HERMES_SESSION_USAGE_FIELDS = {
    "input_tokens",
    "output_tokens",
    "cache_read_tokens",
    "cache_write_tokens",
    "reasoning_tokens",
    "estimated_cost_usd",
    "api_call_count",
}
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
    parser.add_argument(
        "--model-costs-file",
        type=Path,
        default=DEFAULT_MODEL_COSTS_FILE,
        help="YAML file with per-model token prices. Use /dev/null to disable configured cost estimates.",
    )
    parser.add_argument("--agent", action="append", help="Agent id to run. Repeatable. Defaults to all.")
    parser.add_argument(
        "--runs-per-agent",
        type=int,
        default=3,
        help="Number of independent runs for each testcase/agent pair. Defaults to 3.",
    )
    parser.add_argument("--parallel", type=int, default=4, help="Number of testcase/agent runs to run concurrently.")
    parser.add_argument(
        "--docker-env",
        action="append",
        default=[],
        metavar="NAME[=VALUE]",
        help="Environment variable to pass into Docker. Repeatable. Use NAME to pass through from the host.",
    )
    parser.add_argument(
        "--docker-image",
        help="Override the Docker image declared in testcase.md. Useful for running one testcase across versions.",
    )
    parser.add_argument(
        "--docker-oauth",
        action="append",
        choices=["codex", "claude", "hermes", "all"],
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

    args.model_costs = load_model_costs(args.model_costs_file.resolve())
    configure_keychain_env(args)
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
                model_costs=args.model_costs,
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
                model_costs=args.model_costs,
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
        "agent_cost_source": None,
        "grader_cost_source": None,
        "summary": f"harness error: {error}",
        "flare_version_used": None,
        "achieved_accuracy": None,
        "run_summary_bullets": "[]",
        "testcase_improvement_recommendations": "[]",
        "interesting_observations": "[]",
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


def load_model_costs(path: Path) -> dict[str, Any]:
    if str(path) == "/dev/null":
        return {}
    if not path.exists():
        raise SystemExit(f"Missing model costs file: {path}")
    try:
        import yaml  # type: ignore[import-untyped]
    except ModuleNotFoundError:
        raise SystemExit(
            "PyYAML is required to read agent_skill_eval/model_costs.yaml. Install it with `pip install PyYAML`."
        ) from None
    data = yaml.safe_load(path.read_text()) or {}
    if not isinstance(data, dict):
        raise SystemExit(f"Invalid model costs file, expected mapping: {path}")
    models = data.get("models", {})
    if models is None:
        data["models"] = {}
    elif not isinstance(models, dict):
        raise SystemExit(f"Invalid model costs file, expected top-level 'models' mapping: {path}")
    return data


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
        agent_type = agent.get("agent_type") or infer_agent_type(agent_id, command)
        if not isinstance(agent_type, str) or not agent_type:
            raise SystemExit(f"Invalid agent {agent_id} in {path}: missing string agent_type")
        seen.add(agent_id)
        validated.append({"id": agent_id, "label": label, "agent_type": agent_type, "command": command})
    return validated


def infer_agent_type(agent_id: str, command: list[str]) -> str:
    head = Path(command[0]).name if command else ""
    if head in {"codex", "claude"}:
        return head
    if agent_id.startswith("codex-"):
        return "codex"
    if agent_id.startswith("claude-"):
        return "claude"
    return head or agent_id


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
    model_costs: dict[str, Any],
) -> dict[str, Any]:
    agent_dir = out_dir / testcase.id / agent["id"] / f"run_{run_index:02d}"
    workspace_dir = agent_dir / "workspace"
    logs_dir = agent_dir / "logs"
    logs_dir.mkdir(parents=True)
    shutil.copytree(testcase.path / "initial", workspace_dir)
    workdir = infer_workdir(workspace_dir)
    env = build_run_env(agent_dir)

    last_message_host = workdir / ".agent_eval_last_message.txt"
    placeholders = {
        "workdir": DOCKER_CONTAINER_WORKDIR,
        "last_message": f"{DOCKER_CONTAINER_WORKDIR}/.agent_eval_last_message.txt",
    }
    agent_command = expand_command(agent["command"], placeholders)
    agent_container_name = build_container_name(out_dir.name, testcase.id, agent["id"], run_index)
    container_start_stdout = logs_dir / "container_start_stdout.txt"
    container_start_stderr = logs_dir / "container_start_stderr.txt"
    container_prepare_stdout = logs_dir / "container_prepare_stdout.txt"
    container_prepare_stderr = logs_dir / "container_prepare_stderr.txt"
    container_start_result = start_agent_container(
        workdir=workdir,
        docker_config=testcase.docker_config,
        container_name=agent_container_name,
        env=env,
    )
    write_text(container_start_stdout, container_start_result.stdout)
    write_text(container_start_stderr, container_start_result.stderr)
    container_prepare_result = prepare_agent_container(
        container_name=agent_container_name,
        docker_config=testcase.docker_config,
        env=env,
    )
    write_text(container_prepare_stdout, container_prepare_result.stdout)
    write_text(container_prepare_stderr, container_prepare_result.stderr)
    agent_run_command = docker_exec_command(agent_container_name, agent_command)
    agent_prompt = build_agent_prompt(testcase.text, agent)

    started = time.monotonic()
    agent_stdout = logs_dir / "agent_stdout.txt"
    agent_stderr = logs_dir / "agent_stderr.txt"
    if container_start_result.returncode != 0 or container_prepare_result.returncode != 0:
        agent_result = failed_container_agent_result(
            agent_run_command,
            container_start_result,
            container_prepare_result,
            agent_stdout,
            agent_stderr,
        )
    else:
        agent_result = run_command(
            agent_run_command,
            cwd=REPO_ROOT,
            stdin=agent_prompt,
            timeout_seconds=agent_timeout,
            env=env,
            stdout_path=agent_stdout,
            stderr_path=agent_stderr,
        )
    agent_duration = time.monotonic() - started
    agent_token_usage = agent_result.to_record()["token_usage"]
    hermes_usage_collection = collect_hermes_usage(agent, agent_container_name, logs_dir, env, agent_result)
    if hermes_usage_collection.get("usage"):
        agent_token_usage = merge_usage(agent_token_usage, hermes_usage_collection["usage"])

    agent_public_context = build_agent_public_context(agent_result, last_message_host)
    if agent_result.timed_out:
        evidence = []
        container_stop_result = stop_recoverable_container(agent_container_name, logs_dir, env)
        grade = timed_out_grade(logs_dir)
        analysis = timed_out_analysis(logs_dir)
    else:
        evidence = collect_evidence(
            logs_dir=logs_dir,
            testcase_text=testcase.text,
            timeout_seconds=EVIDENCE_TIMEOUT_SECONDS,
            env=env,
            container_name=agent_container_name,
        )
        container_stop_result = stop_recoverable_container(agent_container_name, logs_dir, env)
        grade = run_codex_json_eval(
            name="grader",
            workdir=workdir,
            logs_dir=logs_dir,
            testcase_text=testcase.text,
            evidence=evidence,
            agent_public_context=agent_public_context,
            env=env,
            schema_path=GRADER_OUTPUT_SCHEMA,
            output_file_name="grade.json",
            prompt_builder=build_grader_prompt,
            file_result_key="grade_file",
            command_result_key="grader_result",
            run_command_key="grader_run_command",
        )
        analysis = run_codex_json_eval(
            name="analyzer",
            workdir=workdir,
            logs_dir=logs_dir,
            testcase_text=testcase.text,
            evidence=evidence,
            agent_public_context=agent_public_context,
            env=env,
            schema_path=ANALYSIS_OUTPUT_SCHEMA,
            output_file_name="analysis.json",
            prompt_builder=build_analysis_prompt,
            file_result_key="analysis_file",
            command_result_key="analysis_result",
            run_command_key="analysis_run_command",
        )

    container_remove_result = remove_container(agent_container_name, logs_dir, env)
    archive_path = agent_dir.with_suffix(".zip")
    total_duration = time.monotonic() - started
    result = {
        "agent": {
            "id": agent["id"],
            "label": agent["label"],
            "agent_type": agent["agent_type"],
            "command": agent_command,
        },
        "agent_run_command": agent_run_command,
        "agent_container": {
            "name": agent_container_name,
            "kept": False,
            "recoverable": False,
            "stopped_after_run": container_stop_result.returncode == 0,
            "stopped_after_timeout": agent_result.timed_out,
            "removed_after_run": container_remove_result.returncode == 0,
            "start": container_start_result.to_record(container_start_stdout, container_start_stderr),
            "prepare": container_prepare_result.to_record(container_prepare_stdout, container_prepare_stderr),
            "stop": container_stop_result.to_record(
                logs_dir / "container_stop_stdout.txt",
                logs_dir / "container_stop_stderr.txt",
            ),
            "remove": container_remove_result.to_record(
                logs_dir / "container_remove_stdout.txt",
                logs_dir / "container_remove_stderr.txt",
            ),
        },
        "artifact_archive": {
            "path": str(archive_path),
            "created": True,
            "contains": "run directory files, including logs, result metadata, and final workspace output",
        },
        "testcase_id": testcase.id,
        "run_index": run_index,
        "testcase": str(testcase.path),
        "workdir": str(workdir),
        "duration_seconds": round(total_duration, 3),
        "agent_duration_seconds": round(agent_duration, 3),
        "agent_result": agent_result.to_record(agent_stdout, agent_stderr),
        "agent_usage_collection": hermes_usage_collection,
        "evidence": evidence,
        "grade": grade,
        "analysis": analysis,
        "token_usage": {
            "agent": agent_token_usage,
            "grader": grade.get("grader_result", {}).get("token_usage", {}),
            "analysis": analysis.get("analysis_result", {}).get("token_usage", {}),
        },
    }
    apply_cost_estimates(result, model_costs)
    write_json(agent_dir / "result.json", result)
    archive_result = create_run_archive(agent_dir, archive_path)
    result["artifact_archive"].update(archive_result)
    write_json(agent_dir / "result.json", result)
    return flatten_summary(result)


def infer_workdir(workspace_dir: Path) -> Path:
    children = [path for path in workspace_dir.iterdir() if path.is_dir()]
    if len(children) == 1:
        return children[0]
    return workspace_dir


def build_docker_config(args: argparse.Namespace, testcase_text: str) -> DockerConfig:
    image = args.docker_image or extract_docker_image(testcase_text)
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
    return tuple(mounts), "\n".join(setup_lines)


def configure_keychain_env(args: argparse.Namespace) -> None:
    requested = set(args.docker_oauth)
    if "all" in requested:
        requested.update(["codex", "claude"])

    if args.docker_claude_keychain or "claude" in requested:
        configure_secret_from_keychain(
            service=CLAUDE_KEYCHAIN_SERVICE,
            env_name="ANTHROPIC_API_KEY",
            docker_env=args.docker_env,
            required=args.docker_claude_keychain,
        )

    if "hermes" in requested:
        configure_secret_from_keychain(
            service=HERMES_NVIDIA_KEYCHAIN_SERVICE,
            env_name="NVIDIA_API_KEY",
            docker_env=args.docker_env,
            required=True,
        )


def configure_secret_from_keychain(
    service: str,
    env_name: str,
    docker_env: list[str],
    required: bool,
) -> None:
    if os.environ.get(env_name):
        append_docker_env_once(docker_env, env_name)
        return

    if not is_macos():
        if required:
            raise SystemExit(f"Reading {service!r} from Keychain is only supported on macOS")
        return

    completed = subprocess.run(
        ["security", "find-generic-password", "-s", service, "-w"],
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        if not required:
            return
        stderr = completed.stderr.strip() or completed.stdout.strip()
        raise SystemExit(
            f"Could not read keychain service {service!r}. "
            f"Run outside the command sandbox and approve the macOS prompt. {stderr}"
        )

    key = completed.stdout.strip()
    if not key:
        if required:
            raise SystemExit(f"Keychain service {service!r} returned an empty key")
        return

    os.environ[env_name] = key
    append_docker_env_once(docker_env, env_name)


def append_docker_env_once(docker_env: list[str], env_name: str) -> None:
    if env_name not in [env_spec.split("=", 1)[0] for env_spec in docker_env]:
        docker_env.append(env_name)


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


def build_agent_prompt(testcase_text: str, agent: dict[str, Any]) -> str:
    prompt = extract_section_code_block(testcase_text, "Prompt") or ""
    if not prompt:
        raise SystemExit("Could not find a prompt code block in testcase.md")
    prompt = render_prompt_template(prompt, agent)
    return (
        "You are being evaluated on an agent skill testcase.\n"
        "Work only in the current project folder. Implement the user's request. "
        "Do not ask clarifying questions; make reasonable assumptions and finish the task.\n"
        "Before finishing, write AGENT_EVAL_NOTES.md with concise public notes on your approach, "
        "assumptions, blockers, and any skill improvements that would have helped. Do not include hidden "
        "chain-of-thought; write only a brief engineering summary.\n\n"
        f"User prompt:\n{prompt.strip()}\n"
    )


def render_prompt_template(prompt: str, agent: dict[str, Any]) -> str:
    values = {
        "agent_id": agent["id"],
        "agent_label": agent["label"],
        "agent_type": agent["agent_type"],
    }
    for key, value in values.items():
        prompt = prompt.replace(f"{{{key}}}", str(value))
    return prompt


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
    logs_dir: Path,
    testcase_text: str,
    timeout_seconds: int,
    env: dict[str, str],
    container_name: str,
    evidence_name: str = "evidence",
) -> list[dict[str, Any]]:
    evidence_dir = logs_dir / evidence_name
    evidence_dir.mkdir(exist_ok=True)
    commands = parse_evidence_commands(testcase_text, timeout_seconds)
    return run_evidence_commands_in_container(
        commands=commands,
        evidence_dir=evidence_dir,
        env=env,
        container_name=container_name,
    )


def run_evidence_commands_in_container(
    commands: list[tuple[str, list[str], int, str]],
    evidence_dir: Path,
    env: dict[str, str],
    container_name: str,
) -> list[dict[str, Any]]:
    records = []
    for command_id, command, command_timeout, original in commands:
        if runs_job_py(command):
            records.append(
                collect_nvflare_version_evidence(
                    evidence_dir=evidence_dir,
                    command_id=f"{command_id}_nvflare_version",
                    job_command=command,
                    container_name=container_name,
                    timeout_seconds=min(command_timeout, 60),
                    env=env,
                )
            )
        docker_exec = docker_exec_command(container_name, command)
        result = run_command(
            docker_exec,
            cwd=REPO_ROOT,
            stdin=None,
            timeout_seconds=command_timeout,
            env=env,
        )
        records.append(command_record(command_id, result, evidence_dir, original))
    return records


def start_agent_container(
    workdir: Path,
    docker_config: DockerConfig,
    container_name: str,
    env: dict[str, str],
) -> CommandResult:
    return start_container(
        workdir=workdir,
        docker_config=docker_config,
        container_name=container_name,
        env=env,
        remove_on_stop=False,
        include_oauth=True,
    )


def start_container(
    workdir: Path,
    docker_config: DockerConfig,
    container_name: str,
    env: dict[str, str],
    remove_on_stop: bool,
    include_oauth: bool,
) -> CommandResult:
    command = [
        "docker",
        "run",
        "-d",
        "--name",
        container_name,
        "--network",
        DOCKER_AGENT_NETWORK,
    ]
    for security_opt in DOCKER_SECURITY_OPTS:
        command.extend(["--security-opt", security_opt])
    if remove_on_stop:
        command.insert(2, "--rm")
    command.extend(
        [
            "--mount",
            f"type=bind,src={workdir},dst={DOCKER_CONTAINER_WORKDIR}",
            "-w",
            DOCKER_CONTAINER_WORKDIR,
        ]
    )
    for env_spec in docker_config.env:
        command.extend(["--env", env_spec])
    if include_oauth:
        for source, target in docker_config.oauth_mounts:
            command.extend(["--mount", f"type=bind,src={source},dst={target},readonly"])
    command.extend([docker_config.image, "sleep", "infinity"])
    return run_command(command, cwd=REPO_ROOT, stdin=None, timeout_seconds=60, env=env)


def prepare_agent_container(
    container_name: str,
    docker_config: DockerConfig,
    env: dict[str, str],
) -> CommandResult:
    if not docker_config.oauth_setup_script:
        return CommandResult(
            command=[],
            returncode=0,
            duration_seconds=0,
            stdout="",
            stderr="",
        )
    command = ["docker", "exec", "-i", container_name, "sh", "-lc", docker_config.oauth_setup_script]
    return run_command(command, cwd=REPO_ROOT, stdin=None, timeout_seconds=60, env=env)


def docker_exec_command(container_name: str, command: list[str]) -> list[str]:
    return ["docker", "exec", "-i", "-w", DOCKER_CONTAINER_WORKDIR, container_name] + command


def failed_container_agent_result(
    command: list[str],
    start_result: CommandResult,
    prepare_result: CommandResult,
    stdout_path: Path,
    stderr_path: Path,
) -> CommandResult:
    stdout = (
        "Could not run agent because the Docker container did not start or prepare successfully.\n"
        f"container_start_stdout:\n{start_result.stdout}\n"
        f"container_prepare_stdout:\n{prepare_result.stdout}\n"
    )
    stderr = (
        f"container_start_stderr:\n{start_result.stderr}\n"
        f"container_prepare_stderr:\n{prepare_result.stderr}\n"
    )
    write_text(stdout_path, stdout)
    write_text(stderr_path, stderr)
    return CommandResult(
        command=command,
        returncode=start_result.returncode if start_result.returncode != 0 else prepare_result.returncode,
        duration_seconds=start_result.duration_seconds + prepare_result.duration_seconds,
        stdout=stdout,
        stderr=stderr,
    )


def collect_hermes_usage(
    agent: dict[str, Any],
    container_name: str,
    logs_dir: Path,
    env: dict[str, str],
    agent_result: CommandResult,
) -> dict[str, Any]:
    if agent.get("agent_type") != "hermes":
        return {}
    if agent_result.timed_out:
        return {"source": "hermes_sessions_export", "skipped": "agent_timed_out", "usage": {}}
    if agent_result.returncode is not None and agent_result.returncode < 0:
        return {"source": "hermes_sessions_export", "skipped": "agent_not_run", "usage": {}}
    if docker_container_is_running(container_name, env) is False:
        return {"source": "hermes_sessions_export", "skipped": "container_not_running", "usage": {}}

    stdout_path = logs_dir / "hermes_sessions.jsonl"
    stderr_path = logs_dir / "hermes_sessions_stderr.txt"
    command = docker_exec_command(container_name, ["hermes", "sessions", "export", "-", "--source", "cli"])
    result = run_command(command, cwd=REPO_ROOT, stdin=None, timeout_seconds=120, env=env)
    write_text(stdout_path, result.stdout)
    write_text(stderr_path, result.stderr)

    usage = parse_hermes_session_usage(result.stdout) if result.returncode == 0 else {}
    return {
        "source": "hermes_sessions_export",
        "session_export_path": str(stdout_path),
        "usage": usage,
        "export_result": result.to_record(stdout_path, stderr_path),
    }


def parse_hermes_session_usage(jsonl_text: str) -> dict[str, Any]:
    totals = {field: 0.0 for field in HERMES_SESSION_USAGE_FIELDS}
    row_count = 0
    for line in jsonl_text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        rows = value if isinstance(value, list) else [value]
        for row in rows:
            if not isinstance(row, dict):
                continue
            row_count += 1
            for field in HERMES_SESSION_USAGE_FIELDS:
                item = hermes_session_metric(row, field)
                if isinstance(item, bool) or not isinstance(item, (int, float)):
                    continue
                totals[field] += float(item)

    usage: dict[str, Any] = {}
    field_map = {
        "input_tokens": "input_tokens",
        "output_tokens": "output_tokens",
        "cache_read_tokens": "cache_read_input_tokens",
        "cache_write_tokens": "cache_creation_input_tokens",
        "reasoning_tokens": "reasoning_tokens",
        "api_call_count": "api_call_count",
    }
    for hermes_field, normalized_field in field_map.items():
        value = totals[hermes_field]
        if value:
            usage[normalized_field] = int(value) if value.is_integer() else value

    total_tokens = (
        totals["input_tokens"]
        + totals["output_tokens"]
        + totals["cache_read_tokens"]
        + totals["cache_write_tokens"]
    )
    if total_tokens:
        usage["total_tokens_estimate"] = int(total_tokens) if total_tokens.is_integer() else total_tokens

    estimated_cost = totals["estimated_cost_usd"]
    if estimated_cost:
        usage["total_cost_usd"] = round(estimated_cost, 8)
        usage["cost_source"] = "reported"

    if row_count:
        usage["session_rows"] = row_count
        usage["usage_source"] = "hermes_sessions_export"
    return usage


def hermes_session_metric(row: dict[str, Any], field: str) -> Any:
    for container_key in [None, "usage", "metrics", "totals", "token_usage"]:
        source = row if container_key is None else row.get(container_key)
        if isinstance(source, dict) and field in source:
            return source[field]
    return None


def merge_usage(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if value is not None:
            merged[key] = value
    return merged


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
    container_name: str,
    timeout_seconds: int,
    env: dict[str, str],
) -> dict[str, Any]:
    command = docker_exec_command(container_name, nvflare_version_command(job_command))
    result = run_command(
        command,
        cwd=REPO_ROOT,
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
    probe = parse_nvflare_version_probe(result.stdout) or {}
    record["nvflare_version_probe"] = probe
    record["nvflare_version"] = probe.get("distribution_version") or probe.get("nvflare_version")
    record["nvflare_version_error"] = probe.get("error")
    return record


def command_record(command_id: str, result: CommandResult, evidence_dir: Path, original: str) -> dict[str, Any]:
    stdout_path = evidence_dir / f"{command_id}.stdout.txt"
    stderr_path = evidence_dir / f"{command_id}.stderr.txt"
    write_text(stdout_path, result.stdout)
    write_text(stderr_path, result.stderr)
    record = result.to_record(stdout_path, stderr_path)
    record["id"] = command_id
    record["original"] = original
    record["run_command"] = result.command
    return record


def nvflare_version_command(job_command: list[str]) -> list[str]:
    python_executable = job_command[0] if job_command and Path(job_command[0]).name.startswith("python") else "python"
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
        "        'nvflare_version': None,\n"
        "        'distribution_version': None,\n"
        "        'module_file': None,\n"
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


def build_container_name(run_id: str, testcase_id: str, agent_id: str, run_index: int) -> str:
    raw = f"agent-eval-{run_id}-{testcase_id}-{agent_id}-run-{run_index:02d}"
    name = re.sub(r"[^A-Za-z0-9_.-]+", "-", raw).strip("-")
    if not name or not re.match(r"^[A-Za-z0-9]", name):
        name = f"agent-eval-{name}"
    return name[:128]


def docker_container_name_from_command(command: list[str]) -> str | None:
    if len(command) < 3 or command[0] != "docker":
        return None
    if command[1] == "run":
        for index, part in enumerate(command):
            if part == "--name" and index + 1 < len(command):
                return command[index + 1]
            if part.startswith("--name="):
                return part.split("=", 1)[1]
    if command[1] == "exec":
        index = 2
        options_with_values = {"-e", "--env", "-u", "--user", "-w", "--workdir"}
        while index < len(command):
            part = command[index]
            if part == "--":
                return command[index + 1] if index + 1 < len(command) else None
            if not part.startswith("-"):
                return part
            index += 2 if part in options_with_values else 1
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


def timed_out_grade(logs_dir: Path) -> dict[str, Any]:
    grade_file = logs_dir / "grade.json"
    parsed = {
        "score": 0,
        "score_before_caps": 0,
        "caps_applied": ["Agent run timed out; harness assigns score 0 without grader evaluation."],
        "rubric_breakdown": [
            {
                "category": "Timeout",
                "points_awarded": 0,
                "points_possible": 100,
                "evidence": "The agent command exceeded the testcase timeout.",
            }
        ],
        "summary": "Agent run timed out; score forced to 0.",
        "process_observations": ["The agent did not complete within the testcase timeout."],
        "skill_improvement_suggestions": [],
        "failures": ["agent_timeout"],
    }
    write_json(grade_file, parsed)
    return {
        "parsed": parsed,
        "grade_file": str(grade_file),
        "grader_result": {},
        "grader_run_command": None,
    }


def timed_out_analysis(logs_dir: Path) -> dict[str, Any]:
    analysis_file = logs_dir / "analysis.json"
    parsed = {
        "flare_version_used": "unknown",
        "achieved_accuracy": "none",
        "run_summary_bullets": [
            "The agent command exceeded the testcase timeout.",
            "The harness did not run final evidence after timeout.",
            "The grader was skipped because timeout runs receive score 0.",
            "No reliable NVFlare runtime version was collected for this run.",
            "No federated accuracy evidence was collected for this run.",
        ],
        "testcase_improvement_recommendations": [],
        "interesting_observations": ["Timeout runs are scored deterministically by the harness."],
    }
    write_json(analysis_file, parsed)
    return {
        "parsed": parsed,
        "analysis_file": str(analysis_file),
        "analysis_result": {},
        "analysis_run_command": None,
    }


def run_codex_json_eval(
    name: str,
    workdir: Path,
    logs_dir: Path,
    testcase_text: str,
    evidence: list[dict[str, Any]],
    agent_public_context: dict[str, str | None],
    env: dict[str, str],
    schema_path: Path,
    output_file_name: str,
    prompt_builder: Any,
    file_result_key: str,
    command_result_key: str,
    run_command_key: str,
) -> dict[str, Any]:
    output_file = logs_dir / output_file_name
    stdout_path = logs_dir / f"{name}_stdout.txt"
    stderr_path = logs_dir / f"{name}_stderr.txt"
    prompt = prompt_builder(testcase_text, evidence, agent_public_context, str(workdir))
    command = expand_command(
        GRADER_COMMAND,
        {
            "workdir": str(workdir),
            "schema_file": str(schema_path),
            "grade_file": str(output_file),
        },
    )
    result = run_command(
        command,
        cwd=workdir,
        stdin=prompt,
        timeout_seconds=GRADER_TIMEOUT_SECONDS,
        env=env,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
    )

    return {
        "parsed": parse_grade_file(output_file),
        file_result_key: str(output_file) if output_file.exists() else None,
        command_result_key: result.to_record(stdout_path, stderr_path),
        run_command_key: command,
    }


def build_agent_public_context(agent_result: CommandResult, last_message: Path) -> dict[str, str | None]:
    return {
        "agent_stdout_tail": tail(agent_result.stdout),
        "agent_stderr_tail": tail(agent_result.stderr),
        "agent_last_message": last_message.read_text(errors="replace") if last_message.exists() else None,
    }


def build_grader_prompt(
    testcase_text: str,
    evidence: list[dict[str, Any]],
    agent_public_context: dict[str, str | None],
    workdir: str,
) -> str:
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
        "Collected evidence:\n"
        f"{evidence_text}\n\n"
        "Agent public output:\n"
        f"{public_context_text}\n"
    )


def build_analysis_prompt(
    testcase_text: str,
    evidence: list[dict[str, Any]],
    agent_public_context: dict[str, str | None],
    workdir: str,
) -> str:
    evidence_text = json.dumps(evidence, indent=2)
    public_context_text = json.dumps(agent_public_context, indent=2)
    return (
        "Analyze this agent skill evaluation run independently from the grader. Do not assign a score. "
        "Return JSON only, matching the provided schema.\n"
        f"You are running with read-only access to the final workspace at: {workdir}\n"
        "Inspect the final workspace files directly. Summarize what the agent did in exactly five concise "
        "bullet strings. Use only public artifacts such as files, AGENT_EVAL_NOTES.md, the agent final message, "
        "and stdout/stderr; do not request or infer hidden chain-of-thought.\n"
        "For flare_version_used, use only the nvflare_version_probe evidence collected from the run container; "
        "if the probe failed or is absent, report 'unknown' and mention the error in interesting_observations. "
        "For achieved_accuracy, report the best final-round or final validation accuracy "
        "from evidence, or 'none' if the simulation did not run or did not report accuracy.\n\n"
        "Testcase:\n"
        f"{testcase_text}\n\n"
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


def stop_recoverable_container(container_name: str, logs_dir: Path, env: dict[str, str]) -> CommandResult:
    stdout_path = logs_dir / "container_stop_stdout.txt"
    stderr_path = logs_dir / "container_stop_stderr.txt"
    result = stop_docker_container_result(container_name, env)
    write_text(stdout_path, result.stdout)
    write_text(stderr_path, result.stderr)
    return result


def remove_container(container_name: str, logs_dir: Path, env: dict[str, str]) -> CommandResult:
    stdout_path = logs_dir / "container_remove_stdout.txt"
    stderr_path = logs_dir / "container_remove_stderr.txt"
    result = remove_docker_container_result(container_name, env)
    write_text(stdout_path, result.stdout)
    write_text(stderr_path, result.stderr)
    return result


def stop_docker_container(container_name: str | None, env: dict[str, str]) -> None:
    stop_docker_container_result(container_name, env)


def stop_docker_container_result(container_name: str | None, env: dict[str, str]) -> CommandResult:
    command = ["docker", "stop", container_name or ""]
    if not container_name:
        return CommandResult(command=command, returncode=0, duration_seconds=0, stdout="", stderr="")

    if docker_container_is_running(container_name, env) is False:
        return CommandResult(
            command=command,
            returncode=0,
            duration_seconds=0,
            stdout=f"{container_name} already stopped\n",
            stderr="",
        )
    return run_subprocess_command(command, env, timeout_seconds=20)


def remove_docker_container_result(container_name: str | None, env: dict[str, str]) -> CommandResult:
    command = ["docker", "rm", "-f", container_name or ""]
    if not container_name:
        return CommandResult(command=command, returncode=0, duration_seconds=0, stdout="", stderr="")
    return run_subprocess_command(command, env, timeout_seconds=30)


def run_subprocess_command(command: list[str], env: dict[str, str], timeout_seconds: int) -> CommandResult:
    started = time.monotonic()
    try:
        completed = subprocess.run(
            command,
            text=True,
            cwd=str(REPO_ROOT),
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
            stdout=exc.stdout or "",
            stderr=exc.stderr or "",
            timed_out=True,
        )
    except OSError:
        return CommandResult(
            command=command,
            returncode=None,
            duration_seconds=time.monotonic() - started,
            stdout="",
            stderr="Could not run command.\n",
        )


def docker_container_is_running(container_name: str, env: dict[str, str]) -> bool | None:
    try:
        completed = subprocess.run(
            ["docker", "inspect", "-f", "{{.State.Running}}", container_name],
            text=True,
            cwd=str(REPO_ROOT),
            env=env,
            capture_output=True,
            timeout=20,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip().lower() == "true"


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
        token_total = estimated_total_tokens(result)
        if token_total:
            result["total_tokens_estimate"] = int(token_total)
    return result


def estimated_total_tokens(usage: dict[str, Any]) -> int:
    input_tokens = numeric_usage_value(usage, "input_tokens", "prompt_tokens") or 0
    output_tokens = numeric_usage_value(usage, "output_tokens", "completion_tokens") or 0
    cache_creation_tokens = numeric_usage_value(usage, "cache_creation_input_tokens") or 0
    cache_read_tokens = numeric_usage_value(usage, "cache_read_input_tokens") or 0
    cached_input_tokens = numeric_usage_value(usage, "cached_input_tokens", "cached_tokens") or 0

    total = input_tokens + output_tokens + cache_creation_tokens + cache_read_tokens
    if not total:
        total += cached_input_tokens

    # Some CLIs report reasoning as a detail of output tokens. Only add it when
    # no output token count exists.
    if not output_tokens:
        total += numeric_usage_value(usage, "reasoning_output_tokens", "reasoning_tokens") or 0
    return int(total)


def numeric_usage_value(usage: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        value = usage.get(key)
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            return float(value)
    return None


def collect_usage_metrics(value: Any, metrics: dict[str, float]) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if key in TOKEN_KEYS | COST_KEYS and isinstance(item, (int, float)):
                metrics[key] = max(metrics.get(key, 0), float(item))
            collect_usage_metrics(item, metrics)
    elif isinstance(value, list):
        for item in value:
            collect_usage_metrics(item, metrics)


def apply_cost_estimates(result: dict[str, Any], model_costs: dict[str, Any]) -> None:
    usage_by_role = result.get("token_usage", {})
    if not isinstance(usage_by_role, dict):
        return

    agent = result.get("agent", {})
    if isinstance(agent, dict):
        enrich_usage_cost(
            usage_by_role.get("agent"),
            model_name=model_from_command(agent.get("command", [])),
            agent_id=agent.get("id"),
            model_costs=model_costs,
        )

    grade = result.get("grade", {})
    if isinstance(grade, dict):
        enrich_usage_cost(
            usage_by_role.get("grader"),
            model_name=model_from_command(grade.get("grader_run_command", [])),
            agent_id="grader",
            model_costs=model_costs,
        )

    analysis = result.get("analysis", {})
    if isinstance(analysis, dict):
        enrich_usage_cost(
            usage_by_role.get("analysis"),
            model_name=model_from_command(analysis.get("analysis_run_command", [])),
            agent_id="analysis",
            model_costs=model_costs,
        )


def enrich_usage_cost(
    usage: Any,
    model_name: str | None,
    agent_id: str | None,
    model_costs: dict[str, Any],
) -> None:
    if not isinstance(usage, dict) or not usage:
        return
    if model_name:
        usage.setdefault("cost_model", model_name)

    prefer_reported = bool(model_costs.get("prefer_reported_cost", True))
    if prefer_reported and reported_cost_value(usage) is not None:
        usage["cost_source"] = "reported"
        return

    pricing = find_model_pricing(model_costs, model_name, agent_id)
    estimate = calculate_configured_cost(usage, pricing)
    if estimate is None:
        usage.setdefault("cost_source", "unpriced")
        return

    usage["total_cost_usd"] = round(estimate, 8)
    usage["cost_source"] = "model_costs"


def model_from_command(command: Any) -> str | None:
    if not isinstance(command, list):
        return None
    for index, part in enumerate(command):
        if part == "--model" and index + 1 < len(command):
            return command[index + 1]
        if isinstance(part, str) and part.startswith("--model="):
            return part.split("=", 1)[1]
    return None


def find_model_pricing(model_costs: dict[str, Any], model_name: str | None, agent_id: str | None) -> dict[str, Any] | None:
    models = model_costs.get("models")
    if not isinstance(models, dict):
        return None
    if model_name in models and isinstance(models[model_name], dict):
        return models[model_name]
    for name, config in models.items():
        if not isinstance(config, dict):
            continue
        aliases = config.get("aliases", [])
        if isinstance(aliases, list) and (model_name in aliases or agent_id in aliases):
            config = dict(config)
            config.setdefault("model", name)
            return config
    return None


def calculate_configured_cost(usage: dict[str, Any], pricing: dict[str, Any] | None) -> float | None:
    if not pricing:
        return None
    rates = pricing.get("rates")
    if not isinstance(rates, dict):
        return None

    input_tokens = numeric_usage_value(usage, "input_tokens", "prompt_tokens") or 0
    output_tokens = numeric_usage_value(usage, "output_tokens", "completion_tokens") or 0
    cached_tokens = numeric_usage_value(usage, "cached_input_tokens", "cached_tokens") or 0
    cache_creation_tokens = numeric_usage_value(usage, "cache_creation_input_tokens") or 0
    cache_read_tokens = numeric_usage_value(usage, "cache_read_input_tokens") or 0

    cost = 0.0
    priced = False

    cached_input_rate = rate_value(rates, "cached_input_tokens")
    input_rate = rate_value(rates, "input_tokens", "prompt_tokens")
    if cached_input_rate is not None and cached_tokens:
        uncached_input_tokens = max(input_tokens - cached_tokens, 0)
        if input_rate is not None:
            cost += uncached_input_tokens * input_rate / 1_000_000
            priced = True
        cost += cached_tokens * cached_input_rate / 1_000_000
        priced = True
    elif input_rate is not None and input_tokens:
        cost += input_tokens * input_rate / 1_000_000
        priced = True

    for token_count, *rate_keys in [
        (output_tokens, "output_tokens", "completion_tokens"),
        (cache_creation_tokens, "cache_creation_input_tokens"),
        (cache_read_tokens, "cache_read_input_tokens"),
    ]:
        rate = rate_value(rates, *rate_keys)
        if rate is not None and token_count:
            cost += token_count * rate / 1_000_000
            priced = True

    return cost if priced else None


def rate_value(rates: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        value = rates.get(key)
        if isinstance(value, bool) or value is None:
            continue
        if isinstance(value, (int, float)):
            return float(value)
    return None


def flatten_summary(result: dict[str, Any]) -> dict[str, Any]:
    grade = result.get("grade", {}).get("parsed") or {}
    analysis = result.get("analysis", {}).get("parsed") or {}
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
        "agent_cost_source": agent_usage.get("cost_source"),
        "grader_cost_source": grader_usage.get("cost_source"),
        "summary": grade.get("summary"),
        "flare_version_used": analysis.get("flare_version_used"),
        "achieved_accuracy": analysis.get("achieved_accuracy"),
        "run_summary_bullets": json.dumps(analysis.get("run_summary_bullets", [])),
        "testcase_improvement_recommendations": json.dumps(
            analysis.get("testcase_improvement_recommendations", [])
        ),
        "interesting_observations": json.dumps(analysis.get("interesting_observations", [])),
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
    for key in ["total_cost_usd", "cost_usd", "cost_usd_estimate"]:
        value = usage.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    return None


def reported_cost_value(usage: dict[str, Any]) -> float | None:
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
        "agent_cost_source",
        "grader_cost_source",
        "summary",
        "flare_version_used",
        "achieved_accuracy",
        "run_summary_bullets",
        "testcase_improvement_recommendations",
        "interesting_observations",
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


def create_run_archive(run_dir: Path, archive_path: Path) -> dict[str, Any]:
    started = time.monotonic()
    temp_path = archive_path.with_name(f"{archive_path.name}.tmp")
    file_count = 0
    try:
        if temp_path.exists():
            temp_path.unlink()
        with zipfile.ZipFile(temp_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(run_dir.rglob("*")):
                if not path.is_file():
                    continue
                archive.write(path, Path(run_dir.name) / path.relative_to(run_dir))
                file_count += 1
        temp_path.replace(archive_path)
        return {
            "path": str(archive_path),
            "created": True,
            "file_count": file_count,
            "size_bytes": archive_path.stat().st_size,
            "duration_seconds": round(time.monotonic() - started, 3),
        }
    except Exception as e:  # noqa: BLE001
        if temp_path.exists():
            temp_path.unlink()
        return {
            "path": str(archive_path),
            "created": False,
            "file_count": file_count,
            "duration_seconds": round(time.monotonic() - started, 3),
            "error": f"{type(e).__name__}: {e}",
        }


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
