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
from unittest.mock import Mock

import pytest

from nvflare.apis.fl_constant import FLContextKey, ReservedKey
from nvflare.apis.fl_context import FLContext
from nvflare.private.admin_defs import Message, MsgHeader, ReturnCode
from nvflare.private.defs import RequestHeader, ScopeInfoKey
from nvflare.private.fed.client import training_cmds
from nvflare.private.fed.client.client_engine_internal_spec import ClientEngineInternalSpec


@pytest.fixture
def engine():
    engine = Mock(spec=ClientEngineInternalSpec)
    engine.configure_job_log = Mock()
    context = FLContext()
    context.set_prop(ReservedKey.IDENTITY_NAME, "site-1")
    engine.new_context.return_value = context
    return engine


@pytest.mark.parametrize(
    "processor,operation,job_scoped",
    [
        (training_cmds.AbortAppProcessor, "abort_app", True),
        (training_cmds.AbortTaskProcessor, "abort_task", True),
        (training_cmds.ShutdownClientProcessor, "shutdown", False),
        (training_cmds.RestartClientProcessor, "restart", False),
        (training_cmds.DeleteRunNumberProcessor, "delete_run", True),
    ],
)
@pytest.mark.parametrize("engine_result", ["", "operation accepted"])
def test_training_processors_preserve_job_identity_and_acknowledge_empty_success(
    engine, processor, operation, job_scoped, engine_result
):
    handler = processor()
    request = Message(handler.get_topics()[0], None)
    request.set_header(RequestHeader.JOB_ID, "job-1")
    getattr(engine, operation).return_value = engine_result
    reply = handler.process(request, engine)
    getattr(engine, operation).assert_called_once_with(*(["job-1"] if job_scoped else []))
    assert reply.topic == "reply_" + request.topic
    assert reply.body == (engine_result or "OK")
    assert reply.get_header(MsgHeader.RETURN_CODE) == ReturnCode.OK


@pytest.mark.parametrize(
    "processor",
    [
        training_cmds.AbortAppProcessor,
        training_cmds.AbortTaskProcessor,
        training_cmds.ShutdownClientProcessor,
        training_cmds.RestartClientProcessor,
        training_cmds.DeployProcessor,
        training_cmds.DeleteRunNumberProcessor,
        training_cmds.ConfigureJobLogProcessor,
        training_cmds.ClientStatusProcessor,
        training_cmds.NotifyJobStatusProcessor,
    ],
)
def test_training_processors_fail_closed_on_wrong_engine_type(processor):
    with pytest.raises(TypeError, match="ClientEngineInternalSpec"):
        processor().process(Message("topic", {}), object())


@pytest.mark.parametrize("error", [None, "invalid log config"])
def test_job_log_configuration_reports_site_job_and_backend_error(engine, error):
    request = Message("configure_job_log", {"level": "INFO"})
    request.set_header(RequestHeader.JOB_ID, "job-1")
    engine.configure_job_log.return_value = error
    response = training_cmds.ConfigureJobLogProcessor().process(request, engine)
    engine.configure_job_log.assert_called_once_with("job-1", {"level": "INFO"})
    if error:
        assert response.get_header(MsgHeader.RETURN_CODE) == ReturnCode.ERROR and response.body == error
    else:
        assert "site-1 job job-1" in response.body


def test_client_status_serializes_the_full_engine_status_dictionary(engine):
    engine.get_engine_status.return_value = {"client_name": "site-1", "running_jobs": [{"job_id": "job-1"}]}
    reply = training_cmds.ClientStatusProcessor().process(Message("check_status", None), engine)
    assert json.loads(reply.body) == engine.get_engine_status.return_value
    assert reply.get_header(MsgHeader.RETURN_CODE) == ReturnCode.OK


def test_scope_processor_serializes_scope_names_and_default(monkeypatch):
    monkeypatch.setattr(training_cmds, "get_scope_info", lambda: (["public", "private"], "private"))
    reply = training_cmds.ScopeInfoProcessor().process(Message("scopes", None), None)
    assert json.loads(reply.body) == {
        ScopeInfoKey.SCOPE_NAMES: ["public", "private"],
        ScopeInfoKey.DEFAULT_SCOPE: "private",
    }


def test_status_notification_keeps_launch_identity_to_avoid_stale_job_updates(engine):
    request = Message("notify", None)
    request.set_headers(
        {RequestHeader.JOB_ID: "job-1", RequestHeader.JOB_STATUS: "finished", RequestHeader.CJ_LAUNCH_ID: "launch-2"}
    )
    reply = training_cmds.NotifyJobStatusProcessor().process(request, engine)
    engine.notify_job_status.assert_called_once_with("job-1", "finished", "launch-2")
    assert reply.body == "notify status: finished"


def test_secure_deploy_requires_job_meta_and_per_job_credential_before_staging(engine):
    context = engine.new_context.return_value
    context.set_prop(FLContextKey.SECURE_MODE, True)
    request = Message("deploy", b"not a zip")
    handler = training_cmds.DeployProcessor()
    assert handler.process(request, engine).body == "missing job meta"
    request.set_header(RequestHeader.JOB_META, {"job": "metadata"})
    request.set_header(RequestHeader.JOB_ID, "job-1")
    assert "no valid job credential" in handler.process(request, engine).body
    engine.deploy_app.assert_not_called()
