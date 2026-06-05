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

"""Run the agent skill eval harness on a remote SSH host.

The remote host is expected to have git, Docker, and Python 3. The script uses
the pushed git branch for reproducibility, builds the requested Docker images on
the remote host, and passes remaining arguments through to harness.py.
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_HOST = "pcnudde@rajivc.nvidia.com"
DEFAULT_REMOTE_DIR = "~/agent_skill_eval/NVFlare"
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
    parser = argparse.ArgumentParser(description="Run agent_skill_eval on a remote SSH host.")
    parser.add_argument("--host", default=DEFAULT_HOST, help=f"SSH target. Defaults to {DEFAULT_HOST}.")
    parser.add_argument("--remote-dir", default=DEFAULT_REMOTE_DIR, help="Remote checkout directory.")
    parser.add_argument("--repo-url", default=git_output("config", "--get", "remote.origin.url"), help="Git repo URL.")
    parser.add_argument("--branch", default=git_output("rev-parse", "--abbrev-ref", "HEAD"), help="Git branch to run.")
    parser.add_argument("--python", default="python3", help="Remote Python executable used to create the venv.")
    parser.add_argument(
        "--build-image",
        action="append",
        choices=sorted(IMAGE_BUILDS),
        help="Docker image to build on the remote. Repeatable. Defaults to all images.",
    )
    parser.add_argument("--skip-build", action="store_true", help="Do not build Docker images on the remote.")
    parser.add_argument(
        "--run-image",
        action="append",
        help="Run the harness once for this Docker image or alias. Repeatable for image comparisons.",
    )
    parser.add_argument("--skip-report", action="store_true", help="Do not generate report.html/combined_report.html.")
    parser.add_argument("--allow-dirty", action="store_true", help="Allow running even if the local branch has changes.")
    parser.add_argument("--dry-run", action="store_true", help="Print the remote shell script instead of executing it.")
    parser.add_argument(
        "harness_args",
        nargs=argparse.REMAINDER,
        help="Arguments after -- are passed to agent_skill_eval/harness.py.",
    )
    args = parser.parse_args()

    if args.harness_args[:1] == ["--"]:
        args.harness_args = args.harness_args[1:]

    if not args.allow_dirty:
        dirty = git_output("status", "--porcelain")
        if dirty:
            raise SystemExit("Local working tree has changes. Commit/push them first or use --allow-dirty.")

    script = build_remote_script(args)
    if args.dry_run:
        print(script)
        return 0

    completed = subprocess.run(["ssh", args.host, "bash -s"], input=script, text=True, cwd=REPO_ROOT, check=False)
    return completed.returncode


def build_remote_script(args: argparse.Namespace) -> str:
    build_keys = args.build_image or ["basic", "2.8", "2.9-skills"]
    if any(key in build_keys for key in ["2.8", "2.9-skills"]) and "basic" not in build_keys:
        build_keys = ["basic", *build_keys]

    harness_args = " ".join(shlex.quote(arg) for arg in args.harness_args)
    run_images = [normalize_image_name(image) for image in (args.run_image or [])]

    lines = [
        "set -euo pipefail",
        f"REMOTE_DIR_TEXT={shlex.quote(args.remote_dir)}",
        'case "$REMOTE_DIR_TEXT" in "~/"*) REMOTE_DIR="$HOME/${REMOTE_DIR_TEXT#~/}" ;; *) REMOTE_DIR="$REMOTE_DIR_TEXT" ;; esac',
        f"REPO_URL={shlex.quote(args.repo_url)}",
        f"BRANCH={shlex.quote(args.branch)}",
        f"PYTHON_BIN={shlex.quote(args.python)}",
        'mkdir -p "$(dirname "$REMOTE_DIR")"',
        'if [ ! -d "$REMOTE_DIR/.git" ]; then git clone "$REPO_URL" "$REMOTE_DIR"; fi',
        'cd "$REMOTE_DIR"',
        'git fetch origin "$BRANCH"',
        'git checkout -B "$BRANCH" "origin/$BRANCH"',
        'if [ ! -d .agent-skill-eval-venv ]; then "$PYTHON_BIN" -m venv .agent-skill-eval-venv; fi',
        ". .agent-skill-eval-venv/bin/activate",
        "python -m pip install --upgrade pip",
        "python -m pip install PyYAML matplotlib",
    ]

    if not args.skip_build:
        for key in build_keys:
            dockerfile, tag = IMAGE_BUILDS[key]
            lines.append(f"docker build -f {shlex.quote(dockerfile)} -t {shlex.quote(tag)} .")

    lines.extend(
        [
            "RUN_DIRS=",
            "run_harness() {",
            "  IMAGE_ARG=\"$1\"",
            "  if [ -n \"$IMAGE_ARG\" ]; then",
            f"    python agent_skill_eval/harness.py --docker-image \"$IMAGE_ARG\" {harness_args}",
            "  else",
            f"    python agent_skill_eval/harness.py {harness_args}",
            "  fi",
            "  LATEST_RUN=$(python - <<'PY'",
            "from pathlib import Path",
            "runs = sorted((Path('agent_skill_eval/runs')).glob('20*Z'), key=lambda p: p.stat().st_mtime)",
            "print(runs[-1] if runs else '')",
            "PY",
            "  )",
            "  if [ -n \"$LATEST_RUN\" ]; then RUN_DIRS=\"$RUN_DIRS $LATEST_RUN\"; fi",
            "}",
        ]
    )

    if run_images:
        for image in run_images:
            lines.append(f"run_harness {shlex.quote(image)}")
    else:
        lines.append("run_harness ''")

    if not args.skip_report:
        lines.extend(
            [
                'if [ -n "$RUN_DIRS" ]; then',
                "  python agent_skill_eval/report.py $RUN_DIRS",
                "fi",
            ]
        )

    lines.extend(
        [
            'echo "Remote checkout: $REMOTE_DIR"',
            'echo "Run directories:$RUN_DIRS"',
        ]
    )
    return "\n".join(lines) + "\n"


def normalize_image_name(image: str) -> str:
    return IMAGE_ALIASES.get(image, image)


def git_output(*args: str) -> str:
    completed = subprocess.run(["git", *args], cwd=REPO_ROOT, text=True, capture_output=True, check=True)
    return completed.stdout.strip()


if __name__ == "__main__":
    raise SystemExit(main())
