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

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from nvflare.private.fed.task_worker import artifacts as artifacts_module
from nvflare.private.fed.task_worker import supervisor as module
from tests.task_supervisor_test_utils import SETTLED, _attempt


@pytest.mark.parametrize("key", ["bootstrap", "options", "input_payload", "abort_signal", "launcher", "store"])
def test_supervisor_rejects_wrong_boundary_types_before_staging(tmp_path, key):
    arguments, handle = _attempt(tmp_path)
    arguments[key] = object()
    with pytest.raises(TypeError):
        module.TaskSupervisor().run(**arguments)
    handle.poll.assert_not_called()
    assert not list((tmp_path / "artifacts").rglob("bootstrap.json"))


def test_artifact_root_mismatch_and_non_boolean_retention_fail_before_launch(tmp_path):
    arguments, handle = _attempt(tmp_path)
    arguments["bootstrap"] = replace(arguments["bootstrap"], artifact_root=str(tmp_path / "other"))
    with pytest.raises(ValueError, match="artifact root"):
        module.TaskSupervisor().run(**arguments)
    handle.poll.assert_not_called()
    with pytest.raises(TypeError, match="must be bool"):
        module.TaskSupervisor(retain_artifacts=1)


@pytest.mark.parametrize("deadline", [True, "10", float("inf"), 5.0])
def test_caller_deadline_validation_and_expiration_bound_worker_wait(tmp_path, monkeypatch, deadline):
    arguments, handle = _attempt(tmp_path)
    supervisor = module.TaskSupervisor()
    handle.poll.side_effect = AssertionError("expired or invalid deadline must not poll")
    monkeypatch.setattr(module, "time", SimpleNamespace(monotonic=lambda: 10.0, time=lambda: 10.0))
    with pytest.raises((ValueError, TimeoutError), match="deadline must be finite|timed out"):
        supervisor.run(**arguments, deadline_provider=lambda: deadline)
    handle.cancel.assert_called_once_with()
    assert not supervisor.owns_execution


def test_finite_caller_deadline_caps_settlement_wait(tmp_path, monkeypatch):
    arguments, handle = _attempt(tmp_path)
    monkeypatch.setattr(module, "time", SimpleNamespace(monotonic=lambda: 10.0, time=lambda: 10.0))
    handle.poll.return_value = SETTLED
    result = module.TaskSupervisor().run(**arguments, deadline_provider=lambda: 12.0)
    assert result.result["value"] == 2
    handle.wait_for_settlement.assert_called_once_with(timeout=2.0)


@pytest.mark.parametrize("failure", ["handle", "completion", "abort-after-staging"])
def test_attempt_identity_mismatch_and_prelaunch_abort_cannot_validate_result(tmp_path, monkeypatch, failure):
    arguments, handle = _attempt(tmp_path)
    supervisor = module.TaskSupervisor()
    launch = arguments["launcher"].launch_task.side_effect
    if failure == "handle":

        def wrong_handle(request):
            launch(request)
            handle.request = replace(handle.request, attempt_id="other")
            return handle

        arguments["launcher"].launch_task.side_effect = wrong_handle
    elif failure == "completion":
        read = arguments["store"].read_result

        def wrong_completion(identity):
            result, completion = read(identity)
            return result, replace(completion, identity=replace(identity, task_name="other"))

        monkeypatch.setattr(arguments["store"], "read_result", wrong_completion)
    else:
        factory = arguments["request_factory"]

        def abort_factory(identity, path):
            request = factory(identity, path)
            arguments["abort_signal"].trigger(True)
            return request

        arguments["request_factory"] = abort_factory
    if failure == "abort-after-staging":
        outcome = supervisor.run(**arguments)
        assert outcome.aborted and outcome.result is None
        arguments["launcher"].launch_task.assert_not_called()
    else:
        with pytest.raises(ValueError, match="does not match"):
            supervisor.run(**arguments)
    assert not supervisor.owns_execution


def test_retention_marks_known_attempt_failed_and_reports_unwritable_diagnostics(tmp_path, monkeypatch):
    arguments, _ = _attempt(tmp_path)
    supervisor = module.TaskSupervisor()
    outcome = supervisor.run(**arguments)
    attempt = supervisor._attempts[outcome.identity]
    supervisor.retain_attempt(outcome.identity)
    assert attempt.failed
    with pytest.raises(ValueError, match="invalid task result receipt"):
        supervisor.record_receipt(outcome.identity, "invented")
    monkeypatch.setattr(module, "os", SimpleNamespace(makedirs=Mock(side_effect=PermissionError("read only"))))
    supervisor._diagnostic(attempt, "test")
    issues = supervisor.take_issues()
    assert len(issues) == 1 and "read only" in issues[0][1]
    assert supervisor.take_issues() == []
    supervisor._stopping = True
    store = SimpleNamespace(committed_checkpoint_completion=Mock(return_value=outcome.completion))
    with pytest.raises(RuntimeError, match="cannot adopt"):
        supervisor.recover_checkpoint_result(None, store, "checkpoint", str(tmp_path))


@pytest.mark.parametrize("failure", ["oversized", "identity", "error"])
def test_failure_record_is_validated_before_reporting_worker_error(tmp_path, monkeypatch, failure):
    arguments, _ = _attempt(tmp_path)
    store, identity = arguments["store"], arguments["bootstrap"].identity
    store.create_attempt(identity)
    record = {
        "schema_version": artifacts_module.SCHEMA_VERSION,
        "identity": identity.to_dict(),
        "error_type": "ValueError",
        "message": "failed",
    }
    if failure == "identity":
        record["identity"]["attempt_id"] = "other"
    elif failure == "error":
        record["message"] = 123
    encoded = json.dumps(record)
    if failure == "oversized":
        monkeypatch.setattr(artifacts_module, "_MAX_RECORD_BYTES", 10)
    Path(store.attempt_dir(identity), "failure.json").write_text(encoded)
    with pytest.raises(ValueError, match="too large|invalid identity|invalid error"):
        store.read_failure(identity)


@pytest.mark.parametrize("failure", ["manifest", "reference"])
def test_checkpoint_input_rejects_untrusted_manifest_or_wrong_artifact_kind(tmp_path, failure):
    arguments, _ = _attempt(tmp_path)
    store, identity = arguments["store"], arguments["bootstrap"].identity
    store.create_attempt(identity)
    reference = store.stage_result(identity, arguments["input_payload"])
    manifest = {"identity": identity.to_dict(), "artifact": reference.to_dict()}
    if failure == "manifest":
        manifest["extra"] = True
    Path(store.attempt_dir(identity), "checkpoint_input.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="invalid checkpoint input"):
        store.checkpoint_paths(identity)
