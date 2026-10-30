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

import io
import signal
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from nvflare.utils import process_utils as subject


def test_pipe_drain_handles_partial_lines_crlf_invalid_utf8_and_final_unterminated_line(capsys):
    logger = Mock()
    stream = Mock()
    stream.read1.side_effect = [b"first\r", b"\nsecond\n\n\xffthird\r", b"2026-10-30 12:01:02 INFO child\nlast", b""]
    subject.log_subprocess_output(SimpleNamespace(stdout=stream), logger)
    assert [call.args[0] for call in logger.info.call_args_list] == ["first", "second", "\ufffdthird", "last"]
    assert capsys.readouterr().out == "2026-10-30 12:01:02 INFO child\n"


def test_pipe_drain_continues_if_parent_logger_fails():
    logger = Mock()
    logger.info.side_effect = [RuntimeError("logger failure"), None]
    stream = io.BytesIO(b"first\nsecond\n")
    subject.log_subprocess_output(SimpleNamespace(stdout=stream), logger)
    assert [call.args[0] for call in logger.info.call_args_list] == ["first", "second"]
    assert stream.read() == b""


@pytest.mark.parametrize(
    "argv",
    [
        ["env", "-i", "X=1", "--", "bash", "--rcfile", "rc", "-lc", "echo ${secret:TOKEN}"],
        ["env", "--ignore-environment", "env", "X=1", "python3.12", "-I", "-cprint('${secret:TOKEN}')"],
        ["env", "--unknown-option", "python", "train.py", "${secret:TOKEN}"],
        ["busybox", "ash", "-c", "echo ${secret:TOKEN}"],
    ],
)
def test_secret_references_in_wrapped_code_strings_are_rejected_before_resolution(argv, monkeypatch):
    resolve = Mock()
    monkeypatch.setattr(subject, "resolve_secret_refs", resolve)
    with pytest.raises(ValueError, match="nested interpreter"):
        subject.prepare_subprocess_command(argv)
    resolve.assert_not_called()


@pytest.mark.parametrize(
    "argv",
    [
        ["env", "-i", "X=1", "python", "train.py", "${secret:TOKEN}"],
        ["bash", "--", "script.sh", "${secret:TOKEN}"],
        ["bash", "-o", "errexit", "script.sh", "${secret:TOKEN}"],
        ["python", "--", "train.py", "${secret:TOKEN}"],
    ],
)
def test_script_arguments_preserve_secret_as_one_argv_element(argv, monkeypatch):
    monkeypatch.setenv("TOKEN", "spaces ; and symbols")
    result = subject.prepare_subprocess_command(argv)
    assert result[:-1] == argv[:-1]
    assert result[-1] == "spaces ; and symbols"


@pytest.mark.parametrize(
    "lookup_error, kill_error",
    [(ProcessLookupError(), None), (PermissionError(), ProcessLookupError()), (None, PermissionError())],
)
def test_process_group_cleanup_handles_disappeared_or_inaccessible_processes(monkeypatch, lookup_error, kill_error):
    lookup = Mock(return_value=4321, side_effect=lookup_error)
    kill = Mock(side_effect=kill_error)
    monkeypatch.setattr(subject.os, "getpgid", lookup)
    monkeypatch.setattr(subject.os, "killpg", kill)
    adapter = subject.ProcessAdapter(pid=1234)
    adapter.terminate()
    if isinstance(lookup_error, ProcessLookupError):
        kill.assert_not_called()
    else:
        kill.assert_called_once_with(1234 if lookup_error else 4321, signal.SIGKILL)


def test_wait_reaps_pid_once_and_delegates_popen_wait(monkeypatch):
    waitpid = Mock(return_value=(1234, 5 << 8))
    monkeypatch.setattr(subject.os, "waitpid", waitpid)
    adapter = subject.ProcessAdapter(pid=1234)
    adapter.wait()
    adapter.wait()
    assert adapter.poll() == 5
    waitpid.assert_called_once_with(1234, 0)
    process = Mock(pid=4321)
    subject.ProcessAdapter(process=process).wait()
    process.wait.assert_called_once_with()


def test_reaped_child_wait_is_tolerated_and_poll_marks_unknown_exit(monkeypatch):
    monkeypatch.setattr(subject.os, "waitpid", Mock(side_effect=ChildProcessError()))
    adapter = subject.ProcessAdapter(pid=1234)
    adapter.wait()
    assert adapter.poll() == -1
