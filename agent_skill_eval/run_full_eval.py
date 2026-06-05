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

"""One-command runner for the current agent skill eval matrix.

This script is intended for the remote eval host. It bootstraps a small local
venv for harness/report dependencies, builds the Docker images, runs the current
testcase against the selected NVFlare images, and generates one combined report.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = REPO_ROOT / "agent_skill_eval/runs"
VENV_DIR = REPO_ROOT / ".agent-skill-eval-venv"
BOOTSTRAP_ENV = "AGENT_SKILL_EVAL_BOOTSTRAPPED"
IMAGE_BUILDS = {
    "basic": ("agent_skill_eval/docker/basic/Dockerfile", "nvflare-agent-eval:basic"),
    "2.8": ("agent_skill_eval/docker/nvflare-2.8/Dockerfile", "nvflare-agent-eval:2.8"),
    "2.9-skills": (
        "agent_skill_eval/docker/nvflare-2.9-skills/Dockerfile",
        "nvflare-agent-eval:2.9-skills",
    ),
}
IMAGE_ALIASES = {
    "basic": "nvflare-agent-eval:basic",
    "2.8": "nvflare-agent-eval:2.8",
    "2.9": "nvflare-agent-eval:2.9-skills",
    "2.9-skills": "nvflare-agent-eval:2.9-skills",
}


def main() -> int:
    args = parse_args()
    if args.dry_run:
        print_plan(args)
        return 0

    if not args.no_bootstrap:
        ensure_bootstrap()

    preflight(args)
    if not args.skip_build:
        build_images(args.build_image or ["basic", "2.8", "2.9-skills"])

    run_dirs = []
    for image in [normalize_image_name(image) for image in args.image]:
        run_dirs.append(run_harness(args, image))

    if not args.skip_report and run_dirs:
        report_path = RUNS_DIR / f"combined_report_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.html"
        run([sys.executable, "agent_skill_eval/report.py", *map(str, run_dirs), "--output", str(report_path)])
        print(f"Combined report: {report_path}")

    print("Run directories:")
    for run_dir in run_dirs:
        print(f"  {run_dir}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the full agent skill eval matrix on this host.")
    parser.add_argument(
        "--image",
        action="append",
        default=[],
        help="Docker image or alias to evaluate. Repeatable. Defaults to 2.8 and 2.9-skills.",
    )
    parser.add_argument(
        "--build-image",
        action="append",
        choices=sorted(IMAGE_BUILDS),
        help="Docker image to build. Repeatable. Defaults to all images.",
    )
    parser.add_argument("--runs-per-agent", type=int, default=3, help="Independent runs per testcase/agent pair.")
    parser.add_argument("--parallel", type=int, default=6, help="Concurrent testcase/agent runs.")
    parser.add_argument("--agent", action="append", help="Agent id to run. Repeatable. Defaults to all agents.")
    parser.add_argument("--testcase", action="append", type=Path, help="Testcase directory. Repeatable.")
    parser.add_argument("--skip-build", action="store_true", help="Skip Docker image builds.")
    parser.add_argument("--skip-report", action="store_true", help="Skip combined report generation.")
    parser.add_argument("--no-bootstrap", action="store_true", help="Do not create/use the local eval venv.")
    parser.add_argument("--dry-run", action="store_true", help="Print build/harness commands without running them.")
    parser.add_argument(
        "harness_args",
        nargs=argparse.REMAINDER,
        help="Arguments after -- are passed through to harness.py.",
    )
    args = parser.parse_args()
    if args.harness_args[:1] == ["--"]:
        args.harness_args = args.harness_args[1:]
    if not args.image:
        args.image = ["2.8", "2.9-skills"]
    return args


def ensure_bootstrap() -> None:
    if os.environ.get(BOOTSTRAP_ENV):
        return

    python = VENV_DIR / "bin/python"
    if not python.exists():
        run([sys.executable, "-m", "venv", str(VENV_DIR)])

    probe = subprocess.run(
        [str(python), "-c", "import yaml, matplotlib"],
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if probe.returncode != 0:
        run([str(python), "-m", "pip", "install", "--upgrade", "pip"])
        run([str(python), "-m", "pip", "install", "PyYAML", "matplotlib"])

    env = os.environ.copy()
    env[BOOTSTRAP_ENV] = "1"
    os.execve(str(python), [str(python), str(Path(__file__).resolve()), *sys.argv[1:]], env)


def preflight(args: argparse.Namespace) -> None:
    missing = [name for name in ["docker", "git", "codex"] if shutil.which(name) is None]
    if missing:
        raise SystemExit(f"Missing required command(s) on this host: {', '.join(missing)}")

    if not (Path.home() / ".codex/auth.json").exists():
        print("warning: ~/.codex/auth.json not found; Codex agent/grader auth may fail", file=sys.stderr)
    if not os.environ.get("ANTHROPIC_API_KEY") and not (Path.home() / ".claude/.credentials.json").exists():
        print("warning: no ANTHROPIC_API_KEY or ~/.claude/.credentials.json found; Claude runs may fail", file=sys.stderr)
    if not os.environ.get("NVIDIA_API_KEY"):
        print("warning: NVIDIA_API_KEY is not set; Hermes runs may fail", file=sys.stderr)

    if args.runs_per_agent < 1:
        raise SystemExit("--runs-per-agent must be >= 1")
    if args.parallel < 1:
        raise SystemExit("--parallel must be >= 1")


def build_images(build_keys: list[str]) -> None:
    if any(key in build_keys for key in ["2.8", "2.9-skills"]) and "basic" not in build_keys:
        build_keys = ["basic", *build_keys]
    for key in build_keys:
        dockerfile, tag = IMAGE_BUILDS[key]
        run(["docker", "build", "-f", dockerfile, "-t", tag, "."])


def run_harness(args: argparse.Namespace, image: str) -> Path:
    before = current_run_dirs()
    command = build_harness_command(args, image)
    run(command)
    after = current_run_dirs()
    new_dirs = sorted(after - before, key=lambda path: path.stat().st_mtime)
    if not new_dirs:
        raise SystemExit(f"Harness completed for {image}, but no new run directory was found")
    return new_dirs[-1]


def build_harness_command(args: argparse.Namespace, image: str) -> list[str]:
    command = [
        sys.executable,
        "agent_skill_eval/harness.py",
        "--docker-image",
        image,
        "--runs-per-agent",
        str(args.runs_per_agent),
        "--parallel",
        str(args.parallel),
        "--docker-oauth",
        "codex",
        "--docker-oauth",
        "claude",
    ]
    if os.environ.get("ANTHROPIC_API_KEY"):
        command.extend(["--docker-env", "ANTHROPIC_API_KEY"])
    if os.environ.get("NVIDIA_API_KEY"):
        command.extend(["--docker-env", "NVIDIA_API_KEY"])
    for agent in args.agent or []:
        command.extend(["--agent", agent])
    for testcase in args.testcase or []:
        command.extend(["--testcase", str(testcase)])
    command.extend(args.harness_args)
    return command


def print_plan(args: argparse.Namespace) -> None:
    if not args.skip_build:
        build_keys = args.build_image or ["basic", "2.8", "2.9-skills"]
        if any(key in build_keys for key in ["2.8", "2.9-skills"]) and "basic" not in build_keys:
            build_keys = ["basic", *build_keys]
        for key in build_keys:
            dockerfile, tag = IMAGE_BUILDS[key]
            print("+ " + " ".join(["docker", "build", "-f", dockerfile, "-t", tag, "."]))

    for image in [normalize_image_name(image) for image in args.image]:
        print("+ " + " ".join(build_harness_command(args, image)))


def current_run_dirs() -> set[Path]:
    if not RUNS_DIR.exists():
        return set()
    return {path.resolve() for path in RUNS_DIR.glob("20*Z") if path.is_dir()}


def normalize_image_name(image: str) -> str:
    return IMAGE_ALIASES.get(image, image)


def run(command: list[str]) -> None:
    print("+ " + " ".join(str(part) for part in command), flush=True)
    subprocess.run(command, cwd=REPO_ROOT, check=True)


if __name__ == "__main__":
    raise SystemExit(main())
