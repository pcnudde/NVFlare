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

"""Hard kills before publication, mid-epoch, at epoch end and before result send."""

import signal
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from tests.pt_checkpoint_test_utils import assert_identical


@pytest.mark.timeout(180)
def test_repeated_sigkill_preserves_exact_training_result(tmp_path):
    repo = Path(__file__).resolve().parents[3]

    def run(path, step=-1, phase="none"):
        # subprocess.run kills and reaps its child on timeout, including failures.
        result = subprocess.run(
            [sys.executable, "-m", "tests.pt_checkpoint_test_utils", str(path), str(step), phase],
            cwd=repo,
            capture_output=True,
            text=True,
            timeout=25,
        )
        assert result.returncode == (0 if step < 0 else -signal.SIGKILL), result.stdout + result.stderr

    baseline, resumed = tmp_path / "baseline.pt", tmp_path / "resumed.pt"
    run(baseline)
    for step, phase, committed_step in [
        (2, "after_save", 2),
        (4, "during_save", 2),  # partial .tmp must not replace the committed state
        (4, "after_save", 4),  # exactly at an epoch boundary
        (10, "after_save", 10),
        (12, "after_save", 12),  # finished training, result not yet sent
    ]:
        run(resumed, step, phase)
        assert torch.load(resumed, weights_only=True)["progress"]["step"] == committed_step
        if phase == "during_save":
            assert resumed.with_suffix(".tmp").stat().st_size > 0
    run(resumed)
    assert_identical(torch.load(baseline, weights_only=True), torch.load(resumed, weights_only=True))
