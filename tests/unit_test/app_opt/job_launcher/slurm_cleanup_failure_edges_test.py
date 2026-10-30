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

import errno
import os
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from nvflare.apis.fl_exception import UnsafeComponentError
from nvflare.apis.job_launcher_spec import JobReturnCode
from nvflare.app_opt.job_launcher.slurm import manager as module
from nvflare.app_opt.job_launcher.slurm.config import (
    LookupStatus,
    QueryResult,
    SlurmLauncherError,
    SlurmProtocolError,
    SlurmRecord,
)


@pytest.fixture
def manager(tmp_path):
    config = SimpleNamespace(workspace_path=str(tmp_path / "missing"), query_timeout=1, cancel_timeout=1)
    return module.SlurmJobManager(config, Mock(), Mock(), monotonic_clock=lambda: 10.0)


def handle(manager, tmp_path):
    return module.SlurmJobHandle(manager, "nv-job", "job-name", str(tmp_path / "job"), "42", 3, 0.01)


def test_exclusive_file_failure_closes_descriptor_and_removes_partial_secret(tmp_path, monkeypatch):
    path = tmp_path / "secret"
    local_os = SimpleNamespace(**vars(os))
    descriptors = []

    def fail_permission(fd, mode):
        descriptors.append(fd)
        raise OSError("permission update failed")

    local_os.fchmod = fail_permission
    monkeypatch.setattr(module, "os", local_os)
    with pytest.raises(OSError, match="permission update"):
        module._write_exclusive(str(path), b"credential", 0o600)
    assert not path.exists()
    with pytest.raises(OSError) as error:
        os.fstat(descriptors[0])
    assert error.value.errno == errno.EBADF


def test_control_path_rejects_files_and_missing_workspace_without_scheduler_calls(manager, tmp_path):
    occupied = tmp_path / "file"
    occupied.write_text("preserve")
    with pytest.raises(UnsafeComponentError, match="control directory"):
        module._ensure_dir(str(occupied))
    assert occupied.read_text() == "preserve"
    module._cleanup_job_dir(str(tmp_path / "already-gone"))
    with pytest.raises(UnsafeComponentError, match="cannot inspect workspace_path"):
        manager._validate_workspace()
    with pytest.raises(SlurmLauncherError, match="SYSTEM_BOOTSTRAP"):
        manager._require_initialized()
    manager.shutdown()
    assert not manager.adapter.mock_calls


def test_failed_sandbox_directory_creation_rolls_back_only_its_new_job_directory(manager, tmp_path, monkeypatch):
    manager.jobs_dir = str(tmp_path)
    local_os = SimpleNamespace(**vars(os))

    def mkdir(path, mode):
        if os.path.basename(path) == module.SANDBOX_ROOT:
            raise OSError("sandbox unavailable")
        return os.mkdir(path, mode)

    local_os.mkdir = mkdir
    monkeypatch.setattr(module, "os", local_os)
    with pytest.raises(OSError, match="sandbox unavailable"):
        manager._prepare_job_dir("nv-job", True)
    assert list(tmp_path.iterdir()) == []


def test_terminal_cleanup_failure_still_releases_owned_handle_and_caches_result(manager, tmp_path, monkeypatch):
    item = handle(manager, tmp_path)
    manager._handles[item.nvflare_job_id] = item
    cleanup = Mock(side_effect=OSError("read-only artifacts"))
    monkeypatch.setattr(module, "_cleanup_job_dir", cleanup)
    result = manager._result_for(item, SlurmRecord("42", "COMPLETED", exit_status=777))
    assert result == JobReturnCode.EXECUTION_ERROR
    assert manager._finish(item, result) == result
    assert item.terminal_result == result and not manager._handles
    manager.logger.exception.assert_called_once()
    assert manager._finish(item, JobReturnCode.SUCCESS) == result
    cleanup.assert_called_once_with(item.job_dir)
    manager._initialized = True
    assert manager._poll_handle(item) == result
    manager.adapter.active_by_id.assert_not_called()


def test_accounting_backoff_and_running_record_preserve_unknown_without_false_exit(manager, tmp_path):
    item = handle(manager, tmp_path)
    item.accounting_last_query = 9
    item.accounting_misses = 2
    assert manager._poll_accounting(item) == JobReturnCode.UNKNOWN
    manager.adapter.accounting_by_id.assert_not_called()
    item.accounting_last_query = 0
    manager.adapter.accounting_by_id.return_value = QueryResult(LookupStatus.FOUND, (SlurmRecord("42", "RUNNING"),))
    assert manager._poll_accounting(item) == JobReturnCode.UNKNOWN
    assert item.accounting_misses == 0 and not item.exit_confirmed and item.terminal_result is None
    manager.adapter.accounting_by_id.assert_called_once_with("42", "job-name", timeout=1)


def test_cancellation_keeps_user_intent_when_scheduler_protocol_is_invalid(manager, tmp_path, monkeypatch):
    item = handle(manager, tmp_path)
    poll = Mock(side_effect=SlurmProtocolError("invalid state response"))
    monkeypatch.setattr(manager, "_poll_handle", poll)
    manager._abort_handle(item)
    assert item.cancel_requested and item.user_abort
    assert item.terminal_result is None
    manager.logger.exception.assert_called_once()
    poll.assert_called_once_with(item)
