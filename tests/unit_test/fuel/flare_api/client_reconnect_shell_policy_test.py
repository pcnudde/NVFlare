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

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from nvflare.fuel.flare_api import flare_api as subject
from nvflare.fuel.hci.proto import ProtoKey
from tests.timing_utils import isolate_time


@pytest.fixture
def session():
    value = object.__new__(subject.Session)
    value._do_command = Mock(return_value={ProtoKey.DATA: [{ProtoKey.TYPE: ProtoKey.STRING, ProtoKey.DATA: "output"}]})
    return value


@pytest.mark.parametrize(
    "method,args,command",
    [
        ("tail_target_log", ("server", "-n 3"), "tail server -n 3 log.txt"),
        ("head_target", ("server", "-n 2", "log.txt"), "head server -n 2 log.txt"),
        ("head_target_log", ("server", "-n 2"), "head server -n 2 log.txt"),
        ("get_working_directory", ("server",), "pwd server"),
    ],
)
def test_remote_shell_wrappers_preserve_command_options_and_file_tokens(session, method, args, command):
    assert getattr(session, method)(*args) == "output"
    session._do_command.assert_called_once_with(command, enforce_meta=False)


@pytest.mark.parametrize(
    "values,error,fragment",
    [
        ({"pattern_required": True}, SyntaxError, "pattern"),
        ({"pattern_required": True, "pattern": 3}, ValueError, "pattern"),
        ({"fp_required": True}, SyntaxError, "path"),
    ],
)
def test_remote_shell_missing_arguments_are_rejected_before_command(session, values, error, fragment):
    with pytest.raises(error, match=fragment):
        session._shell_command_on_target("grep", "server", None, None, **values)
    session._do_command.assert_not_called()


def test_remote_shell_directory_path_is_validated_before_command(session):
    with pytest.raises(SyntaxError, match="unsupported characters"):
        session._shell_command_on_target("ls", "server", None, "a b")
    session._do_command.assert_not_called()
    assert session._shell_command_on_target("ls", "server", None, "safe/path") == "output"
    session._do_command.assert_called_once_with("ls server safe/path", enforce_meta=False)


@pytest.mark.parametrize("state", ["same", "missing", "reconnected"])
def test_restart_wait_checks_each_previously_connected_client_before_completion(session, monkeypatch, state):
    clients = (
        []
        if state == "missing"
        else [SimpleNamespace(name="site", last_connect_time="new" if state == "reconnected" else "old")]
    )
    session.get_system_info = Mock(side_effect=[SimpleNamespace(client_info=clients)])
    clock = isolate_time(monkeypatch, subject)
    now, reads = [0], []

    def time():
        reads.append(True)
        assert len(reads) <= 8, "client restart monitor exhausted clock budget"
        return now[0]

    def sleep(delay):
        assert now[0] == 0, "client restart monitor exceeded single-poll budget"
        now[0] = 1

    clock.time, clock.sleep = time, Mock(side_effect=sleep)
    if state == "reconnected":
        session._wait_for_clients_restart({"site": "old"}, 1)
        clock.sleep.assert_not_called()
    else:
        with pytest.raises(TimeoutError, match="site"):
            session._wait_for_clients_restart({"site": "old"}, 1)
        clock.sleep.assert_called_once()
    session.get_system_info.assert_called_once()


def test_empty_restart_targets_never_probe_server(session):
    session.get_system_info = Mock()
    session._wait_for_clients_restart({}, 1)
    session.get_system_info.assert_not_called()
    session.api = Mock()
    session.set_timeout(3)
    session.unset_timeout()
    session.api.set_command_timeout.assert_called_once_with(3)
    session.api.unset_command_timeout.assert_called_once_with()
