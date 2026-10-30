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
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock
from zipfile import ZipFile

import pytest

from nvflare.apis.fl_context import FLContext
from nvflare.apis.job_def import JobMetaKey, RunStatus
from nvflare.apis.job_def_manager_spec import JobDefManagerSpec
from nvflare.fuel.hci.server.constants import ConnProps
from nvflare.private.fed.server import job_cmds as subject
from nvflare.private.fed.server.server_engine import ServerEngine


@pytest.fixture
def env(monkeypatch):
    module = subject.JobCommandModule()
    props = {
        module.JOB_ID: "job",
        ConnProps.USER_NAME: "alice",
        ConnProps.USER_ORG: "org",
        ConnProps.USER_ROLE: "member",
    }
    conn = Mock()
    conn.get_prop.side_effect = lambda key, default=None: props.get(key, default)
    engine = Mock(spec=ServerEngine)
    engine.new_context.side_effect = lambda: nullcontext(FLContext())
    manager = Mock(spec=JobDefManagerSpec)
    engine.job_def_manager = manager
    conn.app_ctx = engine
    monkeypatch.setattr(subject, "_active_study_from_conn", lambda connection: subject.DEFAULT_STUDY)
    return SimpleNamespace(module=module, conn=conn, engine=engine, manager=manager, props=props)


@pytest.mark.parametrize("reverse", [False, True])
def test_job_listing_sorts_before_limiting_latest_results(env, monkeypatch, reverse):
    jobs = [SimpleNamespace(meta={JobMetaKey.JOB_ID.value: str(i), JobMetaKey.SUBMIT_TIME.value: i}) for i in (3, 1, 2)]
    env.manager.get_all_jobs.return_value = jobs
    send = Mock()
    monkeypatch.setattr(env.module, "_send_summary_list", send)
    args = ["list_jobs", "-m", "2"] + (["-r"] if reverse else [])
    env.module.list_jobs(env.conn, args)
    selected = send.call_args.args[1]
    assert [job.meta[JobMetaKey.SUBMIT_TIME.value] for job in selected] == ([3, 2] if reverse else [2, 3])
    env.conn.append_success.assert_called_once_with("")


@pytest.mark.parametrize("case", ["no-jobs", "no-match", "storage-error", "bad-arguments"])
def test_listing_handles_absent_filtered_and_failed_job_storage(env, case):
    env.manager.get_all_jobs.return_value = (
        []
        if case == "no-jobs"
        else [SimpleNamespace(meta={JobMetaKey.JOB_ID.value: "job", JobMetaKey.JOB_NAME.value: "training"})]
    )
    if case == "storage-error":
        env.manager.get_all_jobs.side_effect = OSError("storage offline")
    args = ["list_jobs", "--unknown"] if case == "bad-arguments" else ["list_jobs", "-n", "unmatched"]
    env.module.list_jobs(env.conn, args)
    if case in ("storage-error", "bad-arguments"):
        env.conn.append_error.assert_called_once()
    else:
        assert ("No jobs found" if case == "no-jobs" else "No jobs matching") in env.conn.append_string.call_args.args[
            0
        ]
    assert env.conn.append_success.call_count == int(case == "no-jobs")


@pytest.mark.parametrize("failure", ["missing", "running", "storage", "success"])
def test_delete_job_checks_state_and_tombstones_submit_records(env, failure):
    job = SimpleNamespace(
        meta={
            JobMetaKey.STATUS: RunStatus.RUNNING.value if failure == "running" else RunStatus.FINISHED_COMPLETED.value
        }
    )
    if failure != "missing":
        env.props[env.module.JOB] = job
    if failure == "storage":
        env.manager.delete.side_effect = OSError("storage offline")
    env.manager.mark_submit_records_job_deleted.return_value = [{"record": "a"}]
    env.module.delete_job(env.conn, ["delete_job", "job"])
    assert env.manager.delete.call_count == int(failure in ("storage", "success"))
    assert env.manager.mark_submit_records_job_deleted.call_count == int(failure == "success")
    assert env.conn.append_error.call_count == int(failure != "success")
    assert env.conn.append_success.call_count == int(failure == "success")


@pytest.mark.parametrize("failure", [False, True])
def test_clone_copies_safe_metadata_and_replaces_submitter(env, failure):
    env.props[env.module.JOB] = SimpleNamespace(
        meta={
            JobMetaKey.JOB_NAME.value: "model",
            JobMetaKey.SUBMITTER_NAME.value: "original",
            JobMetaKey.STATUS.value: RunStatus.FINISHED_COMPLETED.value,
        }
    )
    env.manager.clone.return_value = {JobMetaKey.JOB_ID: "clone"}
    if failure:
        env.manager.clone.side_effect = OSError("storage offline")
    env.module.clone_job(env.conn, ["clone", "job"])
    meta = env.manager.clone.call_args.kwargs["meta"]
    assert meta[JobMetaKey.SUBMITTER_NAME.value] == "alice"
    assert meta[JobMetaKey.SUBMITTER_ORG.value] == "org"
    assert meta[JobMetaKey.CLONED_FROM.value] == "job"
    assert JobMetaKey.STATUS.value not in meta
    assert env.conn.append_error.call_count == int(failure)
    assert env.conn.append_success.call_count == int(not failure)


@pytest.mark.parametrize("method", ["_send_detail_list", "_send_summary_list"])
def test_list_rendering_survives_invalid_duration_metadata(env, monkeypatch, method):
    job = SimpleNamespace(meta={JobMetaKey.JOB_ID.value: "job", JobMetaKey.JOB_NAME.value: "model"})
    monkeypatch.setattr(subject.JobCommandModule, "_set_duration", Mock(side_effect=ValueError("bad timestamp")))
    getattr(env.module, method)(env.conn, [job])
    if method == "_send_detail_list":
        assert '"job_id": "job"' in env.conn.append_string.call_args_list[0].args[0]
        assert env.conn.append_string.call_count == 2
    else:
        row = env.conn.append_table.return_value.add_row.call_args.args[0]
        assert row[:2] == ["job", "model"]
        assert row[-1] == "N/A"


def test_client_log_discovery_prefers_exact_paths_and_excludes_server_logs(env):
    paths = ["run/site-a/log.txt", "site-a/log.txt", "server/log.txt", "run/site-b/log.txt", "site-c/", "log.txt"]
    assert env.module._find_workspace_client_log_members(paths) == {
        "site-a": "site-a/log.txt",
        "site-b": "run/site-b/log.txt",
    }
    assert env.module._find_workspace_client_log_sites(paths) == {"site-a", "site-b"}
    assert env.module._find_client_log_member(["run/site-a/log.txt"], "site-a") == "run/site-a/log.txt"
    assert env.module._find_client_log_member(paths, "unknown") is None


@pytest.mark.parametrize("payload", [None, b"broken zip", 123])
def test_corrupt_or_missing_archives_produce_no_job_logs(env, payload):
    assert env.module._extract_server_log_from_workspace_zip(payload) is None
    assert env.module._extract_client_log_from_workspace_zip(payload, "site-a") is None
    assert env.module._extract_client_logs_from_workspace_zip(payload) == {}


def test_workspace_archive_reads_multiple_clients_and_ignores_missing_members(env):
    output = io.BytesIO()
    with ZipFile(output, "w") as archive:
        archive.writestr("run/site-a/log.txt", "client A\n")
        archive.writestr("site-b/log.txt", "client B\n")
        archive.writestr("server/log.txt", "server\n")
    data = output.getvalue()
    assert env.module._extract_client_logs_from_workspace_zip(data) == {"site-a": "client A\n", "site-b": "client B\n"}
    assert env.module._extract_client_log_from_workspace_zip(data, "missing") is None
    assert env.module._extract_server_log_from_workspace_zip(data) == "server\n"


def test_job_log_text_limits_keep_tail_and_decode_incomplete_utf8_safely(env, monkeypatch):
    monkeypatch.setattr(env.module, "MAX_RETURNED_JOB_LOG_BYTES", 0)
    assert env.module._decode_job_log_data("hello") == "... output truncated to last 0 bytes ...\n"
    assert env.module._decode_job_log_data(None) is None
    assert env.module._tail_log_text("abcdef", 3) == "..."
    text = "old line\n" + "x" * 100 + "\nnew line\n"
    shortened = env.module._tail_log_text(text, 40)
    assert shortened.endswith("new line\n")
    assert len(shortened.encode()) <= 40
