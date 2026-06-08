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

from pathlib import Path
from types import SimpleNamespace

from tests.agent_skill_eval import harness, report, run_full_eval


def test_eval_paths_are_under_tests() -> None:
    assert harness.EVAL_ROOT == Path(__file__).resolve().parents[1] / "agent_skill_eval"
    assert harness.REPO_ROOT == Path(__file__).resolve().parents[2]
    assert harness.DEFAULT_TESTCASE == harness.EVAL_ROOT / "testcases/nvflare_basic_pytorch_to_sim"
    assert harness.DEFAULT_OUT_DIR == harness.EVAL_ROOT / "runs"
    assert harness.DEFAULT_TESTCASE.exists()


def test_run_full_eval_harness_command_uses_tests_path(tmp_path: Path) -> None:
    args = SimpleNamespace(
        runs_per_agent=1,
        parallel=8,
        agent=["codex-5.5-xhigh"],
        testcase=[Path("tests/agent_skill_eval/testcases/nvflare_basic_pytorch_to_sim")],
        harness_args=["--docker-oauth", "codex"],
    )

    command = run_full_eval.build_harness_command(
        args,
        "nvflare-agent-eval:2.9-skills",
        tmp_path / "runs",
    )

    assert command[:3] == [
        run_full_eval.sys.executable,
        "tests/agent_skill_eval/harness.py",
        "--docker-image",
    ]
    assert "--parallel" in command
    assert command[command.index("--parallel") + 1] == "8"
    assert "tests/agent_skill_eval/testcases/nvflare_basic_pytorch_to_sim" in command


def test_report_cohort_label_reads_docker_image() -> None:
    result = {
        "agent_container": {
            "start": {
                "command": [
                    "docker",
                    "run",
                    "--name",
                    "agent-eval",
                    "nvflare-agent-eval:2.8",
                    "sleep",
                    "infinity",
                ]
            }
        }
    }

    assert report.cohort_label(Path("ignored-run-dir"), result) == "nvflare-agent-eval:2.8"


def test_agent_prompt_is_exact_testcase_prompt() -> None:
    testcase_text = """
## Prompt

```text
Convert this project for {agent_type}.
```
"""

    prompt = harness.build_agent_prompt(
        testcase_text, {"id": "codex-test", "label": "Codex Test", "agent_type": "codex"}
    )

    assert prompt == "Convert this project for codex."
    assert "being evaluated" not in prompt
    assert "AGENT_EVAL_NOTES" not in prompt


def test_configured_cost_uses_public_gpt_pricing() -> None:
    pricing = {
        "rates": {
            "input_tokens": 5.00,
            "output_tokens": 30.00,
            "cached_input_tokens": 0.50,
        }
    }
    usage = {
        "input_tokens": 1000,
        "cached_input_tokens": 400,
        "output_tokens": 100,
    }

    cost = report.calculate_configured_cost(usage, pricing)

    assert round(cost or 0, 8) == 0.0062
