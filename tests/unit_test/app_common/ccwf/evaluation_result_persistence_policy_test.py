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

from nvflare.apis.dxo import DXO, DataKind, from_file
from nvflare.app_common.ccwf.val_result_manager import EvalResultManager


def test_evaluation_result_persistence_tracks_both_parties_and_returns_independent_snapshot(tmp_path):
    folder = tmp_path / "evaluation"
    manager = EvalResultManager(str(folder))
    path = manager.add_result("evaluator", "owner", DXO(DataKind.METRICS, {"accuracy": 0.75}))
    assert from_file(path).data == {"accuracy": 0.75}
    assert path == str(folder / "evaluator-owner")
    snapshot = manager.get_results()
    snapshot["evaluator"]["owner"] = "mutated"
    snapshot["other"] = {}
    assert manager.get_results() == {"evaluator": {"owner": path}}
    replacement = manager.add_result("evaluator", "owner", DXO(DataKind.METRICS, {"accuracy": 0.8}))
    assert replacement == path and from_file(path).data == {"accuracy": 0.8}
