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

"""Small shared helpers for comparing Python state graphs with TLC output."""

import argparse
import json
import re
import subprocess
import tempfile
from collections.abc import Callable, Mapping
from pathlib import Path

SUCCESS = "Model checking completed. No error has been found."
EDGE = re.compile(r'^([^ ]+) -> ([^ ]+) \[label="([^"]+)"')


def canonical(snapshot: dict) -> str:
    return json.dumps(snapshot, sort_keys=True, separators=(",", ":"))


def parse_value(value: str):
    value = value.replace(r"\"", '"')
    if value in {"TRUE", "FALSE"}:
        return value == "TRUE"
    if value.startswith('"') and value.endswith('"'):
        return value[1:-1]
    if value.isdigit():
        return int(value)
    raise ValueError(f"cannot parse TLC value {value}")


def parse_set(value: str) -> list[str]:
    members = value.removeprefix("{").removesuffix("}").strip()
    return sorted(member.strip() for member in members.split(",") if member.strip())


def parse_graph(
    path: Path, field_parsers: dict[str, Callable[[str], object]]
) -> tuple[set[str], set[tuple[str, str, str]]]:
    nodes = {}
    raw_edges = []
    with path.open(encoding="utf-8") as file:
        for line in file:
            if match := EDGE.match(line):
                raw_edges.append(match.groups())
            elif ' [label="' in line and " -> " not in line:
                node, label = line.split(' [label="', 1)
                snapshot = {}
                for clause in label[: label.rfind('"')].split(r"\n"):
                    field, value = clause.removeprefix(r"/\\ ").split(" = ", 1)
                    if parser := field_parsers.get(field):
                        snapshot[field] = parser(value)
                nodes[node] = canonical(snapshot)
    return set(nodes.values()), {(nodes[a], action, nodes[b]) for a, b, action in raw_edges}


def run_tlc(
    java: str,
    jar: Path,
    model_dir: Path,
    module: str,
    config: str,
    *extra: str,
) -> subprocess.CompletedProcess[str]:
    with tempfile.TemporaryDirectory(prefix=f"nvflare_{module.lower()}_tlc_") as state_dir:
        return subprocess.run(
            [
                java,
                "-XX:+UseParallelGC",
                "-cp",
                str(jar),
                "tlc2.TLC",
                "-cleanup",
                "-workers",
                "1",
                "-metadir",
                state_dir,
                "-config",
                config,
                *extra,
                module,
            ],
            cwd=model_dir,
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )


def checked_output(result: subprocess.CompletedProcess[str], purpose: str) -> str:
    output = f"{result.stdout}\n{result.stderr}"
    if result.returncode or SUCCESS not in output:
        raise RuntimeError(f"{purpose} failed:\n{output}")
    return output


def verify_model(
    java: str,
    jar: Path,
    model_dir: Path,
    module: str,
    safe_config: str,
    liveness_config: str | None,
    expected_mutations: Mapping[str, str],
    *safe_extra: str,
) -> subprocess.CompletedProcess[str]:
    """Run a safe model, optional liveness model, and discriminating mutations."""
    safe = run_tlc(java, jar, model_dir, module, safe_config, *safe_extra)
    checked_output(safe, "safe TLA+ model")

    if liveness_config:
        liveness = run_tlc(java, jar, model_dir, module, liveness_config)
        checked_output(liveness, "safe liveness model")

    for config, expected in expected_mutations.items():
        mutation = run_tlc(java, jar, model_dir, module, config)
        output = f"{mutation.stdout}\n{mutation.stderr}"
        if mutation.returncode == 0 or expected not in output:
            raise RuntimeError(f"{config} did not expose the expected bug ({expected}):\n{output}")
    return safe


def run_cli(check: Callable[[str, Path], None]) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--java", default="java")
    parser.add_argument("--tla2tools", required=True, type=Path)
    args = parser.parse_args()
    check(args.java, args.tla2tools.resolve())
