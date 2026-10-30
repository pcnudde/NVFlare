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
from nvflare.apis.fl_constant import FLContextKey, ReservedKey, RunProcessKey
from nvflare.apis.fl_context import FLContext
from nvflare.apis.job_def import JobMetaKey, RunStatus
from nvflare.private.fed.server import job_runner as module


@pytest.fixture
def env(tmp_path, monkeypatch):
    runner = module.JobRunner(str(tmp_path))
    runner.log_info, runner.log_error, runner.fire_event = Mock(), Mock(), Mock()
    client = Client("site", "token")
    engine = Mock()
    engine.get_clients.return_value = [client]
    engine.validate_targets.return_value = ([client], [])
    engine.server.admin_server.send_requests_and_get_reply_dict.return_value = {"token": None}
    context = FLContext()
    context.set_prop(ReservedKey.ENGINE, engine)
    job = Mock()
    job.job_id, job.meta, job.min_sites, job.required_sites = "job", {}, 0, ["site"]
    job.get_deployment.return_value = {"app": ["site"]}
    monkeypatch.setattr(module, "Workspace", Mock())
    deployer = Mock()
    deployer.deploy.return_value = "deployment failed"
    monkeypatch.setattr(module, "AppDeployer", Mock(return_value=deployer))
    return SimpleNamespace(runner=runner, engine=engine, ctx=context, job=job)


@pytest.mark.parametrize("failure", ["server", "unknown-client", "required-client"])
def test_deployment_reports_server_invalid_target_and_required_client_failures(env, failure):
    if failure == "server":
        env.job.get_deployment.return_value = {"app": ["@ALL"]}
    elif failure == "unknown-client":
        env.engine.validate_targets.return_value = ([], ["site"])
    with pytest.raises(RuntimeError, match="Failed to deploy|unknown clients|deploy failure"):
        env.runner._deploy_job(env.job, {"site": object()}, env.ctx)
    details = env.ctx.get_prop(FLContextKey.JOB_DEPLOY_DETAIL)
    assert any(
        {"server": "deployment failed", "unknown-client": "invalid_clients", "required-client": "required client site"}[
            failure
        ]
        in detail
        for detail in details
    )


def test_deploy_message_requires_admin_authorization_and_binds_submitter_and_application(env):
    env.job.meta = {JobMetaKey.SUBMITTER_NAME: "alice"}
    message = env.runner._make_deploy_message(env.job, b"archive", "app", env.ctx)
    assert message.body == b"archive" and message.topic == module.TrainingTopic.DEPLOY
    assert message.get_header(module.RequestHeader.REQUIRE_AUTHZ) == "true"
    assert message.get_header(module.RequestHeader.ADMIN_COMMAND) == module.AdminCommandNames.SUBMIT_JOB
    assert message.get_header(module.RequestHeader.JOB_ID) == "job"
    assert message.get_header(module.RequestHeader.APP_NAME) == "app"
    assert message.get_header(module.RequestHeader.SUBMITTER_NAME) == "alice"


@pytest.mark.parametrize(
    "code,expected", [(-9, RunStatus.FINISHED_ABNORMAL), (7, RunStatus.FINISHED_EXECUTION_EXCEPTION)]
)
def test_unreported_worker_exit_distinguishes_forced_kill_from_execution_error(code, expected):
    assert module.JobRunner._classify_finished_job_status({RunProcessKey.PROCESS_RETURN_CODE: code}) == expected
