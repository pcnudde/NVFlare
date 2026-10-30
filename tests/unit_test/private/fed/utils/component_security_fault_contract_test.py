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

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from nvflare.apis.client import Client
from nvflare.apis.fl_constant import FLContextKey, FLMetaKey
from nvflare.apis.fl_context import FLContext
from nvflare.apis.fl_exception import UnsafeComponentError
from nvflare.apis.job_def import JobMetaKey
from nvflare.private.admin_defs import Message
from nvflare.private.defs import RequestHeader
from nvflare.private.fed.utils import fed_utils as module


@pytest.mark.parametrize("missing", ["workspace", "job"])
def test_component_authorization_requires_trusted_context(missing):
    context = FLContext()
    if missing == "job":
        context.set_prop(FLContextKey.WORKSPACE_OBJECT, object())
    with pytest.raises(RuntimeError, match="missing workspace|missing job"):
        module.authorize_build_component({}, None, None, context, [])


@pytest.mark.parametrize("failure", ["builtin", "reported", "unsafe", "empty-unsafe"])
def test_component_authorization_returns_actionable_policy_failure(monkeypatch, failure):
    context = FLContext()
    context.set_prop(FLContextKey.WORKSPACE_OBJECT, object())
    context.set_prop(FLContextKey.CURRENT_JOB_ID, "job")
    context.set_prop(FLContextKey.JOB_META, {})
    authorizer = Mock()
    if failure == "builtin":
        authorizer.handle_event.side_effect = UnsafeComponentError("")
    elif failure == "reported":
        context.set_prop(FLContextKey.COMPONENT_BUILD_ERROR, "site denied")
    else:
        context.set_prop(
            FLContextKey.EXCEPTIONS, {"handler": UnsafeComponentError("denied" if failure == "unsafe" else "")}
        )
    monkeypatch.setattr(module, "_DEFAULT_COMPONENT_PATH_AUTHORIZER", authorizer)
    monkeypatch.setattr(module, "fire_event", Mock())
    result = module.authorize_build_component({"path": "custom.Component"}, None, None, context, [])
    expected = {
        "builtin": "built-in component path authorizer",
        "reported": "site denied",
        "unsafe": "denied",
        "empty-unsafe": "handler",
    }
    assert expected[failure] in result
    assert context.get_prop(FLContextKey.COMPONENT_CONFIG) == {"path": "custom.Component"}


def test_scope_properties_and_message_submitter_identity_are_preserved(monkeypatch):
    meta = {
        JobMetaKey.SCOPE: "requested",
        JobMetaKey.SUBMITTER_NAME: "alice",
        JobMetaKey.SUBMITTER_ORG: "org",
        JobMetaKey.SUBMITTER_ROLE: "admin",
    }
    scope = SimpleNamespace(name="effective", props={"policy": True})
    monkeypatch.setattr(module, "get_job_meta_from_workspace", Mock(return_value=meta))
    monkeypatch.setattr(module, "PrivacyService", SimpleNamespace(get_scope=Mock(return_value=scope)))
    props = module.create_job_processing_context_properties(Mock(), "job")
    assert props[FLContextKey.SCOPE_PROPERTIES] == {"policy": True}
    assert props[FLContextKey.EFFECTIVE_JOB_SCOPE_NAME] == "effective"
    message = Message("topic", "body")
    module.set_message_security_data(message, SimpleNamespace(meta=meta), FLContext())
    for name, user in [
        (RequestHeader.SUBMITTER_NAME, RequestHeader.USER_NAME),
        (RequestHeader.SUBMITTER_ORG, RequestHeader.USER_ORG),
        (RequestHeader.SUBMITTER_ROLE, RequestHeader.USER_ROLE),
    ]:
        assert (
            message.get_header(name)
            == message.get_header(user)
            == meta[
                {
                    RequestHeader.SUBMITTER_NAME: JobMetaKey.SUBMITTER_NAME,
                    RequestHeader.SUBMITTER_ORG: JobMetaKey.SUBMITTER_ORG,
                    RequestHeader.SUBMITTER_ROLE: JobMetaKey.SUBMITTER_ROLE,
                }[name]
            ]
        )
    assert message.get_header(RequestHeader.JOB_META) == meta


@pytest.mark.parametrize("gpus", ["]0[", "0[1]"])
def test_gpu_group_syntax_rejects_reversed_or_embedded_group(gpus):
    with pytest.raises(ValueError, match="not properly paired|invalid start"):
        module.split_gpus(gpus)


def test_target_normalization_rejects_unsupported_types_and_ignores_empty_names():
    assert module.get_target_names(["", Client("site", "token"), "site"]) == ["site"]
    with pytest.raises(ValueError, match="invalid target"):
        module.get_target_names([42])


def test_bad_process_return_code_file_uses_launcher_exit_status_and_reports_warning(tmp_path):
    job = tmp_path / "job"
    job.mkdir()
    (job / FLMetaKey.PROCESS_RC_FILE).write_text("not an integer")
    handle, logger = Mock(), Mock()
    handle.poll.return_value = 7
    assert module.get_return_code(handle, "job", str(tmp_path), logger) == 7
    assert "falling back" in logger.warning.call_args.args[0]


def test_direct_command_contract_preserves_argument_boundaries_and_disables_shell(monkeypatch):
    run = Mock(return_value=SimpleNamespace(stdout="output"))
    monkeypatch.setattr(module, "subprocess", SimpleNamespace(run=run, PIPE=-1, STDOUT=-2))
    arguments = ["command", "literal argument; no shell"]
    assert module.execute_command_directly(arguments) == "output"
    run.assert_called_once_with(arguments, capture_output=False, text=True, shell=False, stdout=-1, stderr=-2)
