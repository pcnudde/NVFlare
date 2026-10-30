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

from nvflare.apis.fl_constant import FLContextKey
from nvflare.apis.fl_context import FLContext
from nvflare.apis.job_def import JobMetaKey, RunStatus, SubmitRecordKey, SubmitRecordState
from nvflare.apis.job_def_manager_spec import JobDefManagerSpec
from nvflare.fuel.hci.server.authz import PreAuthzReturnCode
from nvflare.fuel.hci.server.constants import ConnProps
from nvflare.private.fed.server import job_cmds as module
from nvflare.private.fed.server.server_engine import ServerEngine

JOB_ID = "01234567-89ab-4def-8123-456789abcdef"


@pytest.fixture
def env(monkeypatch):
    subject = module.JobCommandModule()
    subject.logger = Mock()
    props = {subject.JOB_ID: JOB_ID}
    connection = Mock()
    connection.get_prop.side_effect = lambda key, default=None: props.get(key, default)
    connection.set_prop.side_effect = lambda key, value: props.__setitem__(key, value)
    engine = Mock(spec=ServerEngine)
    manager = Mock(spec=JobDefManagerSpec)
    engine.job_def_manager = manager
    context = FLContext()
    engine.new_context.side_effect = lambda: nullcontext(context)
    connection.app_ctx = engine
    monkeypatch.setattr(module, "_active_study_from_conn", lambda conn: "study")
    monkeypatch.setattr(module.StudyRegistryService, "get_registry", lambda: None)
    return SimpleNamespace(subject=subject, conn=connection, props=props, engine=engine, manager=manager, ctx=context)


@pytest.mark.parametrize("failure", ["policy", "authorization", "target"])
def test_authorization_propagates_policy_and_invalid_target_rejection(env, monkeypatch, failure):
    env.manager.get_job.return_value = SimpleNamespace(meta={JobMetaKey.STUDY: "study"})
    monkeypatch.setattr(env.subject, "_apply_study_role_for_authz", lambda conn: failure != "policy")
    monkeypatch.setattr(env.subject, "validate_command_targets", Mock(return_value="unknown site"))
    args = ["command", "bad" if failure == "authorization" else JOB_ID, "client", "missing"]
    assert env.subject.authorize_job(env.conn, args) == PreAuthzReturnCode.ERROR
    if failure == "target":
        assert "unknown site" in env.conn.append_error.call_args.args[0]
    env.conn.append_success.assert_not_called()


@pytest.mark.parametrize("failure", ["stopped", "storage", "server", "target"])
def test_configure_logging_stops_on_job_or_server_error_and_rejects_invalid_target(env, failure):
    env.manager.get_job.return_value = SimpleNamespace(meta={JobMetaKey.STATUS: RunStatus.RUNNING.value})
    env.engine.configure_job_log.return_value = "cannot configure" if failure == "server" else ""
    if failure == "stopped":
        env.manager.get_job.return_value.meta[JobMetaKey.STATUS] = "FINISHED:COMPLETED"
    elif failure == "storage":
        env.manager.get_job.side_effect = OSError("storage down")
    env.subject.configure_job_log(env.conn, ["configure", JOB_ID, "invalid" if failure == "target" else "server", "{}"])
    env.conn.append_error.assert_called_once()
    assert env.engine.configure_job_log.call_count == int(failure == "server")
    assert not env.conn.append_success.called


@pytest.mark.parametrize(
    "method,args",
    [
        ("delete_job_id", ["delete", JOB_ID]),
        ("configure_job_log", ["configure", JOB_ID, "server", "{}"]),
        ("do_app_command", ["app", JOB_ID, "topic"]),
    ],
)
def test_commands_reject_non_server_engine_before_runtime_calls(env, method, args):
    env.conn.app_ctx = object()
    with pytest.raises(TypeError, match="engine must be"):
        getattr(env.subject, method)(env.conn, args)
    assert not env.manager.mock_calls


@pytest.mark.parametrize(
    "method,args", [("get_job_meta", ["meta", JOB_ID]), ("list_job_components", ["components", JOB_ID])]
)
def test_metadata_commands_reject_invalid_job_manager(env, method, args):
    env.engine.job_def_manager = object()
    with pytest.raises(TypeError, match="job_def_manager"):
        getattr(env.subject, method)(env.conn, args)
    env.conn.append_success.assert_not_called()


@pytest.mark.parametrize("failure", ["engine", "manager"])
def test_clone_reports_invalid_engine_or_storage_as_internal_error(env, failure):
    if failure == "engine":
        env.conn.app_ctx = object()
    else:
        env.engine.job_def_manager = object()
    env.subject.clone_job(env.conn, ["clone", JOB_ID])
    assert "Exception occurred trying to clone" in env.conn.append_error.call_args.args[0]
    env.conn.append_success.assert_not_called()


def test_abort_storage_exception_reports_failure_and_components_list_hides_system_entries(env):
    env.engine.job_runner = Mock()
    env.engine.job_runner.abort_before_start.side_effect = OSError("unavailable")
    env.subject.abort_job(env.conn, ["abort", JOB_ID])
    assert "unavailable" in env.conn.append_error.call_args.args[0]
    env.manager.list_components.return_value = ["workspace", "meta", "result", "data", "metrics"]
    env.subject.list_job_components(env.conn, ["components", JOB_ID])
    env.conn.append_string.assert_called_once_with("result, metrics")
    assert env.conn.append_success.call_args.kwargs["meta"][module.MetaKey.JOB_COMPONENTS] == ["result", "metrics"]


def test_missing_components_command_is_syntax_error_without_storage_access(env):
    env.subject.list_job_components(env.conn, ["components"])
    assert "Usage" in env.conn.append_error.call_args.args[0]
    assert not env.manager.mock_calls


@pytest.mark.parametrize("path", ["workspace", "client", "clients"])
def test_missing_stored_logs_are_unavailable_instead_of_internal_error(env, path):
    env.manager.get_storage_component.side_effect = module.StorageException("gone")
    if path == "workspace":
        result = env.subject._read_stored_workspace_zip(env.conn, JOB_ID)
    elif path == "client":
        result = env.subject._read_stored_client_job_log(env.conn, JOB_ID, "site")
    else:
        result = env.subject._read_workspace_client_job_logs(env.manager, JOB_ID, env.ctx)
    assert result == ({} if path == "clients" else None)
    env.conn.append_error.assert_not_called()


def test_client_logs_tolerate_invalid_manager_and_rotation_after_exists_check(env, monkeypatch, tmp_path):
    env.engine.job_def_manager = object()
    assert env.subject._read_stored_client_job_log(env.conn, JOB_ID, "site") is None
    root = tmp_path / "site"
    root.mkdir()
    (root / "fl.log").write_text("before rotation")
    env.engine.get_workspace.return_value.get_log_root.return_value = str(tmp_path)
    monkeypatch.setattr(env.subject, "_collect_job_log_lines", Mock(side_effect=FileNotFoundError("rotated")))
    assert env.subject._read_live_client_job_log(env.engine, JOB_ID, "site") is None
    assert env.subject._get_job_client_targets(None) == set()
    assert env.subject._get_job_client_targets(SimpleNamespace(meta={JobMetaKey.DEPLOY_MAP.value: []})) == set()
    assert env.subject._tail_log_text("short", 100) == "short"


@pytest.mark.parametrize("failure", ["missing-upload", "invalid", "manager", "blocked", "storage", "success"])
def test_submit_failure_paths_do_not_create_jobs_and_success_preserves_custom_properties(env, monkeypatch, failure):
    env.props[ConnProps.FILE_LOCATION] = None if failure == "missing-upload" else "upload.zip"
    env.props[ConnProps.CUSTOM_PROPS] = {"color": "blue"}
    meta = {JobMetaKey.JOB_NAME.value: "training"}
    validator = Mock()
    validator.validate.return_value = (failure != "invalid", "invalid definition", meta)
    monkeypatch.setattr(module, "JobMetaValidator", lambda: validator)
    if failure == "manager":
        env.engine.job_def_manager = object()
    if failure == "blocked":
        env.ctx.set_prop(FLContextKey.JOB_BLOCK_REASON, "policy denied")
    if failure == "storage":
        env.manager.create.side_effect = OSError("create failed")
    env.manager.create.return_value = {JobMetaKey.JOB_ID: JOB_ID}
    env.subject.submit_job(env.conn, ["submit", "training"])
    assert env.manager.create.call_count == int(failure in ["storage", "success"])
    if failure == "success":
        assert env.manager.create.call_args.args[0][JobMetaKey.CUSTOM_PROPS.value] == {"color": "blue"}
        env.conn.append_success.assert_called_once()
        env.conn.append_error.assert_not_called()
    else:
        env.conn.append_error.assert_called_once()
        env.conn.append_success.assert_not_called()


@pytest.mark.parametrize(
    "created,existing,error",
    [
        (False, None, "no existing"),
        (False, {SubmitRecordKey.STATE.value: SubmitRecordState.JOB_DELETED.value}, None),
        (False, {SubmitRecordKey.JOB_CONTENT_HASH.value: "other"}, None),
        ("exception", None, "storage failed"),
        ("exception", {SubmitRecordKey.JOB_CONTENT_HASH.value: "hash"}, None),
    ],
)
def test_submit_record_creation_races_recover_or_fail_without_duplicate_job_creation(
    env, monkeypatch, created, existing, error
):
    monkeypatch.setattr(env.subject, "_get_submit_record", Mock(side_effect=[None, existing]))
    monkeypatch.setattr(env.subject, "_new_submit_record", Mock(return_value={SubmitRecordKey.JOB_ID.value: JOB_ID}))
    creator = Mock(side_effect=OSError("storage failed") if created == "exception" else None, return_value=created)
    monkeypatch.setattr(env.subject, "_create_submit_record", creator)
    monkeypatch.setattr(env.subject, "_job_for_submit_record", Mock(return_value=None))
    repair = Mock(return_value=JOB_ID)
    monkeypatch.setattr(env.subject, "_ensure_submit_record_job_id", repair)
    recover = Mock(return_value="recovered")
    monkeypatch.setattr(env.subject, "_create_job_for_submit_record", recover)
    kwargs = dict(
        study="study",
        submitter={},
        submit_token="token",
        job_content_hash="hash",
        meta={},
        folder_name="training",
        zip_file_name="upload.zip",
        fl_ctx=env.ctx,
    )
    if error:
        with pytest.raises((RuntimeError, OSError), match=error):
            env.subject._handle_submit_token_record_locked(env.conn, env.manager, **kwargs)
    else:
        result = env.subject._handle_submit_token_record_locked(env.conn, env.manager, **kwargs)
        assert result == ("recovered" if created == "exception" else None)
    assert repair.call_count == recover.call_count == int(created == "exception" and existing is not None)
    env.manager.create.assert_not_called()


def test_legacy_submit_token_fallback_uses_record_and_rejects_malformed_storage_contract(env, monkeypatch):
    manager = SimpleNamespace(get_job=Mock(return_value="job"))
    monkeypatch.setattr(env.subject, "_get_submit_record", Mock(return_value={SubmitRecordKey.JOB_ID.value: JOB_ID}))
    assert env.subject._list_jobs_by_submit_token(manager, "study", {}, "token", env.ctx) == ["job"]
    manager.get_job.assert_called_once_with(JOB_ID, env.ctx)
    assert env.subject._job_for_submit_record(manager, None, env.ctx) is None
    assert env.subject._job_id_from_job(SimpleNamespace(meta="invalid")) is None
    with pytest.raises(RuntimeError, match="does not implement"):
        env.subject._require_submit_record_method(manager, "new_submit_record")
    monkeypatch.setattr(env.subject, "_new_submit_record", Mock(return_value=[]))
    with pytest.raises(RuntimeError, match="did not return a record"):
        env.subject._ensure_submit_record_job_id(
            manager,
            {},
            study="study",
            submitter={},
            submit_token="token",
            job_content_hash="hash",
            meta={},
            folder_name="training",
            fl_ctx=env.ctx,
        )


def test_download_cleanup_removes_only_transfer_directory_and_invalid_manager_fails_closed(env, monkeypatch, tmp_path):
    transfer = tmp_path / "transfer"
    transfer.mkdir()
    (transfer / "partial").write_text("data")
    monkeypatch.setattr(env.subject, "tx_path", Mock(return_value=str(transfer)))
    env.subject._clean_up_download(env.conn, "tx")
    assert not transfer.exists() and tmp_path.exists()
    env.engine.job_def_manager = object()
    getter = Mock()
    env.subject._download_job_comps(env.conn, ["download", JOB_ID], getter)
    getter.assert_not_called()
    assert env.conn.append_error.call_args.args[0] == "internal error"
