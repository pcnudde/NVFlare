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

from unittest.mock import Mock

import pytest

from nvflare.fuel.flare_api import flare_api as subject
from nvflare.fuel.hci.proto import MetaKey
from nvflare.fuel.utils import deprecated
from tests.timing_utils import isolate_time


@pytest.fixture
def session(monkeypatch):
    monkeypatch.setattr(deprecated, "_WARNED_DEPRECATION_MESSAGES", set(deprecated._WARNED_DEPRECATION_MESSAGES))
    result = object.__new__(subject.Session)
    result._do_command = Mock(return_value={subject.ResultKey.META: {}})
    return result


@pytest.mark.parametrize("path", [None, "", 7, "absent-job"])
def test_job_submission_rejects_missing_or_invalid_folder_before_server_command(session, tmp_path, path):
    session.upload_dir = str(tmp_path)
    with pytest.raises(subject.InvalidJobDefinition):
        session.submit_job(path)
    session._do_command.assert_not_called()


def test_relative_submission_resolves_upload_folder_and_requires_server_job_id(session, tmp_path):
    folder = tmp_path / "job"
    folder.mkdir()
    session.upload_dir = str(tmp_path)
    with pytest.raises(subject.InternalError, match="return job id"):
        session.submit_job("job")
    assert str(folder) in session._do_command.call_args.args[0]
    session._do_command.return_value = {subject.ResultKey.META: {MetaKey.JOB_ID: "accepted"}}
    assert session.submit_job("job") == "accepted"


@pytest.mark.parametrize("method", ["clone_job", "get_job_meta"])
def test_server_job_reply_must_contain_required_identity_or_metadata(session, method):
    with pytest.raises(subject.InternalError, match="return job"):
        getattr(session, method)("job")
    with pytest.raises(subject.JobNotFound, match="required"):
        getattr(session, method)("")


def test_download_without_destination_keeps_server_location(session):
    session._do_command.return_value = {subject.ResultKey.META: {MetaKey.LOCATION: "/download/job"}}
    assert session.download_job_result("job") == "/download/job"


@pytest.mark.parametrize(
    "kwargs", [{"detailed": 1}, {"reverse": 1}, {"limit": "1"}, {"id_prefix": 1}, {"name_prefix": 1}]
)
def test_job_list_argument_types_are_validated_before_command(session, kwargs):
    with pytest.raises(ValueError):
        session.list_jobs(**kwargs)
    session._do_command.assert_not_called()


@pytest.mark.parametrize("mode", ["callback", "missing-status", "timeout", "running-then-done"])
def test_job_monitor_is_bounded_and_preserves_terminal_metadata(session, monkeypatch, mode):
    meta = {subject.JobMetaKey.STATUS.value: "FINISHED:COMPLETED"}
    pending = {subject.JobMetaKey.STATUS.value: "RUNNING"}
    session.get_job_meta = Mock(side_effect=[pending, meta] if mode == "running-then-done" else [{}])
    clock = isolate_time(monkeypatch, subject)
    clock.time = Mock(side_effect=[0, 2] if mode == "timeout" else [0, 0.1, 0.2])
    clock.sleep = Mock(side_effect=[None])
    callback = Mock(return_value=False) if mode == "callback" else None
    if mode == "missing-status":
        with pytest.raises(subject.InternalError, match="missing status"):
            session.monitor_job_and_return_job_meta("job", timeout=1)
    else:
        result, metadata = session.monitor_job_and_return_job_meta("job", timeout=1, cb=callback)
        expected = {
            "timeout": subject.MonitorReturnCode.TIMEOUT,
            "callback": subject.MonitorReturnCode.ENDED_BY_CB,
            "running-then-done": subject.MonitorReturnCode.JOB_FINISHED,
        }[mode]
        assert result == expected
        assert metadata is (meta if mode == "running-then-done" else None)
    assert session.get_job_meta.call_count == (0 if mode == "timeout" else 2 if mode == "running-then-done" else 1)
    assert clock.sleep.call_count == int(mode == "running-then-done")


@pytest.mark.parametrize("method", ["check_status", "report_resources", "report_version"])
def test_system_reporting_refuses_unknown_target_type(session, method):
    with pytest.raises(ValueError, match="invalid target_type"):
        getattr(session, method)("invalid")
    session._do_command.assert_not_called()
