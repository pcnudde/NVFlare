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

import os
import threading
from argparse import Namespace
from unittest.mock import Mock

import pytest

from nvflare.private.fed.app.simulator import simulator_runner as module


@pytest.mark.parametrize("mode", ["first-poll", "second-poll", "send-poll", "recv-error", "next-client", "same-client"])
def test_fake_simulator_worker_health_checks_bound_dispatch_and_always_cleanup(monkeypatch, tmp_path, mode):
    subject = module.SimulatorClientRunner.__new__(module.SimulatorClientRunner)
    subject.logger = Mock()
    subject.args = Namespace(workspace=str(tmp_path), log_config=None)
    subject.simulator_root = str(tmp_path)
    custom = tmp_path / "server-custom"
    custom.mkdir()
    subject.server_custom_folder = str(custom)
    subject.build_ctx, subject.client_config = {}, {"setting": True}
    subject.deploy_args = Namespace(workspace=str(tmp_path))
    client = Mock()
    client.client_name, client.token = "site", "token"
    subject.federated_clients = [client]
    next_client = Mock(client_name="next")
    subject.get_next_run_client = Mock(return_value=next_client)
    process = Mock(returncode=9)
    process.poll.side_effect = {
        "first-poll": [9],
        "second-poll": [None, 9],
        "send-poll": [None, 9],
        "recv-error": [None],
        "next-client": [None, None],
        "same-client": [None] * 5,
    }[mode]
    launch = Mock(return_value=process)
    monkeypatch.setattr(module, "popen_in_new_session", launch)
    monkeypatch.setattr(module, "get_open_ports", lambda count: [12345])
    workspace = Mock()
    workspace.get_app_custom_dir.return_value = str(tmp_path / "client-custom")
    monkeypatch.setattr(module, "Workspace", Mock(return_value=workspace))
    monkeypatch.setattr(module, "ConfigService", Mock())
    monkeypatch.setattr(module, "add_custom_dir_to_path", Mock())
    monkeypatch.setenv(module.SystemVarName.PYTHONPATH, os.pathsep.join([str(custom), "retained"]))
    connection = Mock()
    connection.recv.side_effect = {
        "first-poll": [],
        "second-poll": [True],
        "send-poll": [False],
        "recv-error": [EOFError("closed")],
        "next-client": [False],
        "same-client": [False, True, "site"],
    }[mode]
    subject._create_connection = Mock(return_value=connection)
    subject._cleanup_process = Mock()
    result = subject.do_one_task(client, 2 if mode == "next-client" else 1, "0", threading.Lock(), timeout=3)
    subject._create_connection.assert_called_once_with(12345, timeout=3)
    subject._cleanup_process.assert_called_once_with(process)
    assert launch.call_args.kwargs["shell"] is False
    assert launch.call_args.kwargs["env"][module.SystemVarName.PYTHONPATH] == "retained"
    assert launch.call_args.args[0][-2:] == ["--gpu", "0"]
    assert subject.deploy_args.workspace == str(tmp_path)
    if mode == "next-client":
        assert result == (False, next_client, None)
        subject.get_next_run_client.assert_called_once_with("0")
    elif mode == "same-client":
        assert result == (True, client, "site")
        assert [call.args[0] for call in connection.send.call_args_list[1:]] == [True, False]
    else:
        assert result == (True, None, "site")
        assert (
            "Subprocess died" in subject.logger.error.call_args.args[0]
            or "Communication error" in subject.logger.error.call_args.args[0]
        )
