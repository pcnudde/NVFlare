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

CLIENT_DIR = Path(__file__).resolve().parents[5] / "nvflare" / "private" / "fed" / "client"


def test_job_executor_has_no_shadow_process_state_machine():
    source = (CLIENT_DIR / "client_executor.py").read_text(encoding="utf-8")

    assert "RunProcessKey" not in source
    assert "_PendingJobHandle" not in source
    assert "transition(" not in source
