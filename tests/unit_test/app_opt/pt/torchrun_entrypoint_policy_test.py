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

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from nvflare.app_opt.pt import torchrun_node as module


def test_entrypoint_executes_same_interpreter_with_validated_node_group_arguments(monkeypatch):
    execute = Mock()
    monkeypatch.setattr(
        module,
        "sys",
        SimpleNamespace(
            argv=["torchrun_node", "--nproc-per-node=2", "train.py", "--epochs", "3"], executable="/python", stderr=None
        ),
    )
    monkeypatch.setattr(module, "os", SimpleNamespace(environ={}, execv=execute))
    module.main()
    execute.assert_called_once_with(
        "/python",
        [
            "/python",
            "-u",
            "-m",
            "torch.distributed.run",
            "--nproc_per_node=2",
            "--standalone",
            "train.py",
            "--epochs",
            "3",
        ],
    )


def test_entrypoint_reports_topology_error_and_exits_before_process_replacement(monkeypatch, capsys):
    import sys

    execute = Mock()
    monkeypatch.setattr(
        module, "sys", SimpleNamespace(argv=["torchrun_node", "train.py"], executable="/python", stderr=sys.stderr)
    )
    monkeypatch.setattr(module, "os", SimpleNamespace(environ={"NVFL_NNODES": "0"}, execv=execute))
    with pytest.raises(SystemExit) as error:
        module.main()
    assert error.value.code == 2 and isinstance(error.value.__cause__, ValueError)
    assert "torchrun_node: invalid node-group topology" in capsys.readouterr().err
    execute.assert_not_called()
