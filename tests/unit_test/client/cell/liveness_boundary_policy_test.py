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
from types import SimpleNamespace
from unittest.mock import MagicMock, call

import numpy as np
import pytest

from nvflare.client.cell import api as module
from nvflare.client.cell.bootstrap import (
    BOOTSTRAP_SCHEMA_VERSION,
    EXTERNAL_PROCESS_EXECUTION_MODE,
    BootstrapKey,
    write_bootstrap_config,
)
from nvflare.client.cell.defs import MsgKey, Topic


@pytest.fixture
def client(tmp_path):
    path = tmp_path / "bootstrap.json"
    write_bootstrap_config(
        str(path),
        {
            BootstrapKey.SCHEMA_VERSION: BOOTSTRAP_SCHEMA_VERSION,
            BootstrapKey.EXECUTION_MODE: EXTERNAL_PROCESS_EXECUTION_MODE,
            BootstrapKey.CONNECT_URL: "tcp://127.0.0.1:12345",
            BootstrapKey.CJ_FQCN: "site-1.job",
            BootstrapKey.TRAINER_FQCN: "site-1.job.trainer",
            BootstrapKey.LAUNCH_TOKEN: "token",
            BootstrapKey.JOB_ID: "job",
            BootstrapKey.SITE_NAME: "site-1",
        },
    )
    return module.CellClientAPI(str(path))


@pytest.mark.parametrize("error,expected", [(None, True), (ProcessLookupError(), False), (PermissionError(), True)])
def test_owner_liveness_probe_treats_permission_denial_as_existing_process(monkeypatch, client, error, expected):
    local_os = SimpleNamespace(**vars(os))
    local_os.kill = MagicMock(side_effect=error)
    monkeypatch.setattr(module, "os", local_os)
    client._cj_pid = 123
    assert client._owner_process_alive() is expected
    local_os.kill.assert_called_once_with(123, 0)


@pytest.mark.parametrize("mode", ["non_posix", "term_missing", "term_error", "kill_missing", "kill_success"])
def test_orphan_cleanup_escalation_uses_only_owned_process_group_boundary(monkeypatch, client, mode):
    local_os = SimpleNamespace(**vars(os))
    local_os.name = "nt" if mode == "non_posix" else "posix"
    local_os.getpgrp = MagicMock(return_value=123)
    local_os._exit = MagicMock(side_effect=SystemExit(1))
    failure = {"term_missing": ProcessLookupError(), "term_error": OSError("denied")}.get(mode)
    local_os.killpg = MagicMock(
        side_effect=failure if failure else [None, ProcessLookupError()] if mode == "kill_missing" else None
    )
    monkeypatch.setattr(module, "os", local_os)
    sleep = MagicMock()
    monkeypatch.setattr(module, "time", SimpleNamespace(sleep=sleep))
    exits = mode in ("non_posix", "term_error", "kill_success")
    if exits:
        with pytest.raises(SystemExit) as caught:
            client._terminate_orphaned_process_group()
        assert caught.value.code == 1
    else:
        client._terminate_orphaned_process_group()
    if mode == "non_posix":
        local_os.killpg.assert_not_called()
        local_os.getpgrp.assert_not_called()
    else:
        assert local_os.killpg.call_args_list[0] == call(123, module.process_signal.SIGTERM)
        if mode in ("kill_missing", "kill_success"):
            assert local_os.killpg.call_args_list[1] == call(123, module.process_signal.SIGKILL)
            sleep.assert_called_once_with(module._OWNER_TERM_GRACE)
        else:
            sleep.assert_not_called()
    assert local_os._exit.call_count == int(exits)


def test_owner_watchdog_observes_loss_after_first_alive_probe_without_real_thread(client):
    stop = MagicMock()
    stop.wait.side_effect = [False, False]
    client._owner_watchdog_stop = stop
    client._owner_process_alive = MagicMock(side_effect=[True, False])
    client._terminate_orphaned_process_group = MagicMock()
    client._owner_watchdog_loop()
    client._terminate_orphaned_process_group.assert_called_once_with()
    assert stop.wait.call_args_list == [call(module._OWNER_WATCHDOG_INTERVAL), call(module._OWNER_WATCHDOG_INTERVAL)]


@pytest.mark.parametrize(
    "attach,stopped,terminated", [(True, False, False), (False, True, False), (False, False, True)]
)
def test_cj_timeout_escalates_only_for_live_external_process_owner(client, attach, stopped, terminated):
    client._is_attach = attach
    client._owner_watchdog_stop = MagicMock()
    client._owner_watchdog_stop.wait.return_value = stopped
    client._terminate_orphaned_process_group = MagicMock()
    client._terminate_after_cj_timeout()
    assert client._terminate_orphaned_process_group.call_count == int(terminated)
    assert client._owner_watchdog_stop.wait.call_count == int(not attach)


def test_cleanup_joins_owned_watchdog_and_heartbeat_and_cancels_control_wait(client):
    owner, heartbeat = MagicMock(), MagicMock()
    owner.is_alive.return_value = heartbeat.is_alive.return_value = True
    client._owner_watchdog_thread = owner
    client._heartbeat_thread = heartbeat
    client._stop_owner_watchdog()
    client._stop_heartbeat()
    owner.join.assert_called_once_with(timeout=module._HEARTBEAT_JOIN_TIMEOUT)
    heartbeat.join.assert_called_once_with(timeout=module._HEARTBEAT_JOIN_TIMEOUT)
    assert client._owner_watchdog_stop.is_set() and client._heartbeat_stop.is_set()
    assert client._heartbeat_cancel.triggered


def test_heartbeat_transport_failure_still_triggers_expiry_policy(client):
    stop = MagicMock()
    stop.wait.return_value = False
    client._heartbeat_stop = stop
    client._heartbeat_interval = 5
    client._heartbeat_timeout = 20
    client._session_id = "session"
    client._result_send_active = True
    client._cell = MagicMock()
    client._cell.send_request.side_effect = OSError("transport failed")
    client._abort_if_cj_timed_out = MagicMock(return_value=True)
    client._terminate_after_cj_timeout = MagicMock()
    client._heartbeat_loop()
    kwargs = client._cell.send_request.call_args.kwargs
    assert kwargs["topic"] == Topic.HEARTBEAT and kwargs["timeout"] == 5
    assert kwargs["request"].payload == {MsgKey.SESSION_ID: "session", MsgKey.RESULT_SOURCE_LIVE: True}
    client._terminate_after_cj_timeout.assert_called_once_with()
    assert stop.wait.call_count == 1


def test_cell_cleanup_suppresses_stop_failure_and_scalar_conversion_is_shape_sensitive(client):
    client._cell = MagicMock()
    client._cell.stop.side_effect = ValueError("already stopped")
    client._stop_cell()
    client._cell.stop.assert_called_once_with()
    scalar = module._to_python_scalar(np.array(2.5))
    assert type(scalar) is float and scalar == 2.5
    array = np.array([2.5])
    assert module._to_python_scalar(array) is array
