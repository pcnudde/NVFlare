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

import errno
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from nvflare.tool.poc import poc_commands as subject
from tests.timing_utils import isolate_time


@pytest.mark.parametrize("returncode", [0, 7])
def test_synchronous_service_launch_reports_child_exit_without_real_process(monkeypatch, returncode):
    run = Mock(return_value=SimpleNamespace(returncode=returncode))
    monkeypatch.setattr(subject.subprocess, "run", run)
    monkeypatch.setattr(subject, "_env_with_cli_python_path", Mock(return_value={"NVFL_TEST": "1"}))
    if returncode:
        with pytest.raises(subject.PocServiceStartError, match="exited with code 7"):
            subject.sync_process("site", "python train.py")
    else:
        subject.sync_process("site", "python train.py")
    run.assert_called_once_with(["python", "train.py"], env={"NVFL_TEST": "1"})


@pytest.mark.parametrize("failure", ["timeout", "exit"])
def test_stop_command_reports_timeout_or_child_stderr(monkeypatch, failure):
    run = (
        Mock(side_effect=subject.subprocess.TimeoutExpired("stop", 2))
        if failure == "timeout"
        else Mock(return_value=SimpleNamespace(returncode=3, stderr=" refused ", stdout=""))
    )
    monkeypatch.setattr(subject.subprocess, "run", run)
    monkeypatch.setattr(subject, "_docker_cli_env", Mock(return_value={}))
    with pytest.raises((TimeoutError, RuntimeError), match="timeout|refused"):
        subject._run_stop_command("site", "bash stop.sh", 2)
    assert run.call_args.kwargs["timeout"] == 2


@pytest.mark.parametrize("failure", ["success", "timeout"])
def test_service_stop_wait_is_bounded_by_deadline(monkeypatch, failure):
    live = Mock(side_effect=[True] if failure == "timeout" else [True, False, False])
    monkeypatch.setattr(subject, "_is_live_pid_file", live)
    clock = isolate_time(monkeypatch, subject)
    clock.monotonic = Mock(side_effect=[2] if failure == "timeout" else [0])
    clock.sleep = Mock(side_effect=[None])
    if failure == "timeout":
        with pytest.raises(TimeoutError, match="site"):
            subject._wait_for_poc_services_stopped("/kits", ["site"], 1)
        clock.sleep.assert_not_called()
    else:
        subject._wait_for_poc_services_stopped("/kits", ["site"], 1)
        clock.sleep.assert_called_once_with(1)


@pytest.mark.parametrize("content,permission", [("invalid pid", False), ("17", True), ("17", False)])
def test_pid_file_probe_distinguishes_invalid_pid_permission_and_absent_process(
    tmp_path, monkeypatch, content, permission
):
    path = tmp_path / "pid.fl"
    path.write_text(content)
    kill = Mock(side_effect=OSError(errno.EPERM if permission else errno.ESRCH, "probe"))
    monkeypatch.setattr(subject.os, "kill", kill)
    assert subject._is_live_pid_file(str(path)) is permission
    assert kill.call_count == int(content == "17")
