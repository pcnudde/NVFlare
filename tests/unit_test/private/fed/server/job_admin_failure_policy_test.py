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

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from nvflare.apis.fl_constant import ReturnCode, ServerCommandKey
from nvflare.apis.fl_context import FLContext
from nvflare.apis.job_def import JobMetaKey, RunStatus, SubmitRecordKey, SubmitRecordState
from nvflare.apis.job_def_manager_spec import JobDefManagerSpec
from nvflare.apis.shareable import Shareable
from nvflare.fuel.hci.server.authz import PreAuthzReturnCode
from nvflare.fuel.hci.server.constants import ConnProps
from nvflare.private.admin_defs import Message
from nvflare.private.fed.server import job_cmds as subject
from nvflare.private.fed.server.server_engine import ServerEngine

JOB_ID = "01234567-89ab-4def-8123-456789abcdef"


@pytest.fixture
def env(monkeypatch):
    module = subject.JobCommandModule()
    props = {ConnProps.CMD_ENTRY: SimpleNamespace(usage="command job_id"), module.JOB_ID: JOB_ID}
    connection = Mock()
    connection.get_prop.side_effect = lambda key, default=None: props.get(key, default)
    connection.set_prop.side_effect = lambda key, value: props.__setitem__(key, value)
    engine = Mock(spec=ServerEngine)
    engine.run_processes = {}
    manager = Mock(spec=JobDefManagerSpec)
    engine.job_def_manager = manager
    engine.new_context.side_effect = lambda: nullcontext(FLContext())
    connection.app_ctx = engine
    monkeypatch.setattr(module, "_apply_study_role_for_authz", Mock(return_value=True))
    monkeypatch.setattr(subject, "secure_log_traceback", Mock())
    monkeypatch.setattr(subject, "new_message", lambda conn, topic, body, require_authz: Message(topic, body))
    return SimpleNamespace(module=module, conn=connection, engine=engine, manager=manager, props=props)


@pytest.mark.parametrize("failure", ["missing", "invalid", "absent", "study"])
def test_job_authorization_hides_absent_and_cross_study_jobs(env, monkeypatch, failure):
    job = SimpleNamespace(meta={JobMetaKey.STUDY: "study-a"})
    env.manager.get_job.return_value = None if failure == "absent" else job
    monkeypatch.setattr(subject, "_active_study_from_conn", lambda conn: "study-b" if failure == "study" else None)
    args = ["command"] if failure == "missing" else ["command", "bad" if failure == "invalid" else JOB_ID]
    assert env.module.authorize_job_id(env.conn, args) == PreAuthzReturnCode.ERROR
    env.conn.append_error.assert_called_once()
    assert env.props.get(env.module.JOB) is None


def test_job_authorization_normalizes_uuid_and_sets_submitter_for_policy(env, monkeypatch):
    job = SimpleNamespace(
        meta={
            JobMetaKey.SUBMITTER_NAME: "alice",
            JobMetaKey.SUBMITTER_ORG: "org",
            JobMetaKey.SUBMITTER_ROLE: "researcher",
        }
    )
    env.manager.get_job.return_value = job
    monkeypatch.setattr(subject, "_active_study_from_conn", lambda conn: None)
    assert (
        env.module.authorize_job_file(env.conn, ["download", "tx", JOB_ID.upper(), "meta.json"])
        == PreAuthzReturnCode.REQUIRE_AUTHZ
    )
    assert env.props[env.module.JOB_ID] == JOB_ID
    assert env.props[env.module.JOB] is job
    assert env.props[ConnProps.SUBMITTER_NAME] == "alice"
    assert env.props[ConnProps.SUBMITTER_ORG] == "org"
    assert env.props[ConnProps.SUBMITTER_ROLE] == "researcher"


@pytest.mark.parametrize(
    "method,args",
    [
        ("authorize_job_file", ["cmd"]),
        ("authorize_configure_job_log", ["cmd", "id"]),
        ("configure_job_log", ["cmd", "id"]),
        ("do_app_command", ["cmd"]),
    ],
)
def test_malformed_admin_commands_report_syntax_without_engine_calls(env, method, args):
    result = getattr(env.module, method)(env.conn, args)
    env.conn.append_error.assert_called_once()
    assert not env.engine.mock_calls
    if method.startswith("authorize"):
        assert result == PreAuthzReturnCode.ERROR


@pytest.mark.parametrize("failure", ["running", "server-delete", "success"])
def test_run_directory_deletion_rejects_running_jobs_and_propagates_server_failure(env, monkeypatch, failure):
    env.engine.run_processes = {JOB_ID: {}} if failure == "running" else {}
    env.engine.delete_job_id.return_value = "server deletion failed" if failure == "server-delete" else ""
    env.engine.get_clients.return_value = [SimpleNamespace(token="token")]
    send = Mock(return_value=[])
    monkeypatch.setattr(env.module, "send_request_to_clients", send)
    monkeypatch.setattr(env.module, "process_replies_to_table", Mock())
    env.module.delete_job_id(env.conn, ["delete", JOB_ID])
    if failure == "success":
        assert env.props[env.module.TARGET_CLIENT_TOKENS] == ["token"]
        message = send.call_args.args[1]
        assert message.get_header(subject.RequestHeader.JOB_ID) == JOB_ID
        env.conn.append_success.assert_called_once_with("")
    else:
        env.conn.append_error.assert_called_once()
        send.assert_not_called()
        env.conn.append_success.assert_not_called()


@pytest.mark.parametrize("failure", ["not-running", "missing", "wrong-type", "remote-error", "success"])
def test_app_command_validates_running_job_result_and_return_code(env, failure):
    env.engine.run_processes = {} if failure == "not-running" else {JOB_ID: {}}
    result = Shareable()
    result[ServerCommandKey.DATA] = {"round": 4}
    if failure == "remote-error":
        result.set_return_code(ReturnCode.EXECUTION_EXCEPTION)
    env.engine.send_app_command.return_value = (
        None if failure == "missing" else "bad" if failure == "wrong-type" else result
    )
    env.props[ConnProps.CMD_PROPS] = {"get": "round"}
    env.module.do_app_command(env.conn, ["app", JOB_ID, "query"])
    if failure == "success":
        env.conn.append_dict.assert_called_once_with({"round": 4}, meta=subject.make_meta(subject.MetaStatusValue.OK))
    else:
        env.conn.append_error.assert_called_once()
    if failure == "not-running":
        env.engine.send_app_command.assert_not_called()
    else:
        env.engine.send_app_command.assert_called_once_with(JOB_ID, "query", {"get": "round"}, 5.0)


@pytest.mark.parametrize("failure", ["missing-job", "running", "no-components", "storage", "success"])
def test_finished_job_download_checks_state_and_cleans_partial_transfer(env, monkeypatch, tmp_path, failure):
    status = RunStatus.RUNNING.value if failure == "running" else RunStatus.FINISHED_COMPLETED.value
    env.manager.get_job.return_value = (
        None if failure == "missing-job" else SimpleNamespace(meta={JobMetaKey.STATUS: status})
    )
    env.manager.get_storage_for_download.side_effect = OSError("download failed") if failure == "storage" else None
    monkeypatch.setattr(env.module, "tx_path", Mock(return_value=str(tmp_path / "transfer")))
    download = Mock()
    cleanup = Mock()
    monkeypatch.setattr(env.module, "download_folder", download)
    monkeypatch.setattr(env.module, "_clean_up_download", cleanup)
    components = Mock(return_value=[] if failure == "no-components" else [("DATA", "job.zip")])
    env.module._download_job_comps(env.conn, ["download", JOB_ID], components, require_finished=True)
    if failure == "success":
        env.manager.get_storage_for_download.assert_called_once()
        assert download.call_args.kwargs["folder_name"] == JOB_ID
        env.conn.append_error.assert_not_called()
    elif failure == "no-components":
        assert env.conn.append_string.call_args.args[0] == "No components to download"
        download.assert_not_called()
    else:
        env.conn.append_error.assert_called_once()
        download.assert_not_called()
    assert cleanup.call_count == int(failure == "storage")


def test_extra_job_component_filter_excludes_control_and_invalid_names(env):
    env.manager.list_components.return_value = [
        subject.META,
        subject.DATA,
        subject.WORKSPACE,
        "scheduled",
        "ERRORLOG_site-a",
        "../outside",
    ]
    assert env.module._get_extra_job_components(env.manager, JOB_ID, FLContext()) == [
        ("ERRORLOG_site-a", "ERRORLOG_site-a")
    ]
    env.manager.list_components.return_value = []
    assert env.module._get_extra_job_components(env.manager, JOB_ID, FLContext()) is None


@pytest.mark.parametrize("failure", ["created", "exception", "not-created", "conflict", "deleted"])
def test_submit_token_storage_races_recover_existing_jobs_without_duplicate_creation(env, monkeypatch, failure):
    record = {
        SubmitRecordKey.JOB_ID.value: JOB_ID,
        SubmitRecordKey.JOB_CONTENT_HASH.value: "hash",
        SubmitRecordKey.STATE.value: SubmitRecordState.CREATING.value,
    }
    existing = dict(record)
    if failure == "conflict":
        existing[SubmitRecordKey.JOB_CONTENT_HASH.value] = "different"
    if failure == "deleted":
        existing[SubmitRecordKey.STATE.value] = SubmitRecordState.JOB_DELETED.value
    monkeypatch.setattr(env.module, "_get_submit_record", Mock(side_effect=[None, existing]))
    monkeypatch.setattr(env.module, "_new_submit_record", Mock(return_value=record))
    create = Mock(
        side_effect=OSError("raced") if failure in ("exception", "conflict", "deleted") else None,
        return_value=failure == "created",
    )
    monkeypatch.setattr(env.module, "_create_submit_record", create)
    monkeypatch.setattr(env.module, "_job_for_submit_record", Mock(return_value=SimpleNamespace(job_id=JOB_ID)))
    update = Mock()
    monkeypatch.setattr(env.module, "_update_submit_record", update)
    env.manager.create.return_value = {JobMetaKey.JOB_ID.value: JOB_ID}
    result = env.module._handle_submit_token_record_locked(
        env.conn,
        env.manager,
        study="study",
        submitter={"name": "alice"},
        submit_token="token",
        job_content_hash="hash",
        meta={},
        folder_name="job",
        zip_file_name="job.zip",
        fl_ctx=FLContext(),
    )
    assert result == (None if failure in ("conflict", "deleted") else JOB_ID)
    assert env.manager.create.call_count == int(failure == "created")
    assert update.call_count == int(failure == "created")
    assert env.conn.append_error.call_count == int(failure in ("conflict", "deleted"))
