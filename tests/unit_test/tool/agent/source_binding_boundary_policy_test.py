# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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
from unittest.mock import Mock

import pytest

from nvflare.tool.agent.inspection import files, source


@pytest.mark.parametrize(
    "body,owners,unresolved",
    [
        ("import torch\noptimizer = torch.optim.SGD([])\noptimizer += []\noptimizer.step()\n", [], []),
        (
            'import torch\noptimizer = torch.optim.SGD([])\nmatch data:\n case [*optimizer] if optimizer.step():\n  pass\n case {"a": item, **optimizer}:\n  optimizer.step()\n',
            [],
            ["pytorch", "pytorch"],
        ),
        (
            "from transformers import Trainer\ndef train(*args):\n trainer = Trainer()\n assert ready, trainer.train()\n",
            ["huggingface"],
            [],
        ),
        (
            "import torch\ndef outer():\n torch = None\n def inner():\n  optimizer = torch.optim.SGD([])\n  optimizer.step()\n inner()\n",
            [],
            [],
        ),
    ],
)
def test_static_analysis_tracks_rebinding_match_captures_and_assert_calls(body, owners, unresolved):
    facts = source.analyze_tree(ast.parse(body), "train.py", is_job_py=False, findings=[])
    assert facts.complete
    assert [framework for framework, _, _ in facts.owners] == owners
    assert [framework for framework, _, _ in facts.unresolved] == unresolved


def test_nested_binding_scope_cannot_recover_explicitly_shadowed_import():
    bindings = source.Bindings(outer_symbols={"torch": "torch", "other": "safe"})
    bindings.rebind("torch")
    child = bindings.nested_scope()
    assert "torch" not in child.outer_symbols
    assert child.outer_symbols == {"other": "safe"}
    assert bindings.outer_symbols["torch"] == "torch"


def test_source_scanner_skips_secret_files_without_opening_them(tmp_path):
    secret = tmp_path / "private.key"
    secret.write_text("do not inspect")
    scan = files.scan_sources(secret, redact=True, max_files=1, max_file_bytes=100)
    assert scan.files_read == 0 and scan.facts == {}
    assert scan.findings[0]["code"] == "SENSITIVE_FILE_SKIPPED"
    directory_scan = files.scan_sources(tmp_path, redact=True, max_files=1, max_file_bytes=100)
    assert directory_scan.files_read == 0
    assert directory_scan.findings[0]["code"] == "SENSITIVE_FILE_SKIPPED"


@pytest.mark.parametrize("boundary", ["directory", "file"])
def test_source_scanner_reports_unreadable_boundaries_without_partial_analysis(tmp_path, monkeypatch, boundary):
    path = tmp_path / "train.py"
    path.write_text("print('not read')")
    if boundary == "directory":
        monkeypatch.setattr(Path, "iterdir", Mock(side_effect=PermissionError("denied")))
        target = tmp_path
    else:
        monkeypatch.setattr(Path, "open", Mock(side_effect=PermissionError("denied")))
        target = path
    scan = files.scan_sources(target, redact=True, max_files=1, max_file_bytes=100)
    assert not scan.complete and scan.files_read == 0 and scan.facts == {}
    assert scan.findings[0]["code"] == ("UNREADABLE_DIRECTORY" if boundary == "directory" else "UNREADABLE_FILE")
