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

import ast
from pathlib import Path

SERVER_DIR = Path(__file__).resolve().parents[5] / "nvflare" / "private" / "fed" / "server"


class _StatusWrites(ast.NodeVisitor):
    def __init__(self, path):
        self.path = path
        self.function = None
        self.writes = []

    def visit_FunctionDef(self, node):
        previous = self.function
        self.function = node.name
        self.generic_visit(node)
        self.function = previous

    def visit_Call(self, node):
        if isinstance(node.func, ast.Attribute) and node.func.attr == "set_status" and len(node.args) >= 2:
            self.writes.append((self.path.name, self.function, ast.unparse(node.args[1])))
        self.generic_visit(node)


def test_server_status_write_routes_are_explicit():
    writes = []
    for path in SERVER_DIR.glob("*.py"):
        visitor = _StatusWrites(path)
        visitor.visit(ast.parse(path.read_text(encoding="utf-8")))
        writes.extend(visitor.writes)

    assert set(writes) == {
        ("job_cmds.py", "abort_job", "RunStatus.FINISHED_ABORTED"),  # pre-start cancellation
        ("job_runner.py", "publish_completion_status", "status"),  # modeled live completion
        ("job_runner.py", "run", "RunStatus.DISPATCHED"),
        ("job_runner.py", "run", "RunStatus.RUNNING"),
        ("job_runner.py", "run", "RunStatus.FAILED_TO_RUN"),
        ("job_runner.py", "update_abnormal_finished_jobs", "RunStatus.FINISHED_ABNORMAL"),  # startup recovery
        ("job_runner.py", "update_unfinished_jobs", "RunStatus.ABANDONED"),  # shutdown recovery
    }
