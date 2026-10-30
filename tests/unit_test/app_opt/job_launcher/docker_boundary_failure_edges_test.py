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

from unittest.mock import Mock

import pytest

pytest.importorskip("docker")

from nvflare.apis.job_launcher_spec import JobProcessArgs, JobReturnCode  # noqa: E402


@pytest.fixture
def module():
    # Original Docker tests install their SDK boundary during collection. Resolve
    # the production module afterwards so its SDK reference stays consistent.
    from nvflare.app_opt.job_launcher import docker_launcher

    return docker_launcher


@pytest.mark.parametrize(
    "args,reason",
    [
        ({JobProcessArgs.PARENT_URL: ("-p",)}, "malformed"),
        ({JobProcessArgs.PARENT_URL: ("-p", "tcp://host:10"), JobProcessArgs.PARENT_CONN_SEC: ("-s",)}, "malformed"),
        ({JobProcessArgs.PARENT_URL: ("-p", "tcp://[bad")}, "invalid parent URL"),
        ({JobProcessArgs.PARENT_URL: ("-p", "tcp://host:notaport")}, "invalid parent URL"),
        (
            {
                JobProcessArgs.PARENT_URL: ("-p", "shared-file:///shared"),
                JobProcessArgs.PARENT_CONN_SEC: ("-s", "mtls"),
            },
            "scheme does not match",
        ),
    ],
)
def test_parent_endpoint_validation_rejects_malformed_or_inconsistent_transport_before_rewriting(module, args, reason):
    original = dict(args)
    with pytest.raises(ValueError, match=reason):
        module._rewrite_parent_url(args, "site")
    assert args == original


def test_absent_parent_endpoint_is_unchanged_and_ambiguous_gpu_requests_are_rejected(module):
    args = {"unrelated": "value"}
    rewritten, shared_path = module._rewrite_parent_url(args, "site")
    assert rewritten is args and shared_path is None
    with pytest.raises(ValueError, match="single GPU device request"):
        module._get_gpu_request({}, {}, {"device_requests": [{"capabilities": [["gpu"]], "count": 1}] * 2})


def test_container_transport_errors_do_not_claim_exit_confirmation_or_restartability(module):
    client = Mock()
    client.containers.get.side_effect = RuntimeError("daemon unavailable")
    item = module.DockerJobHandle("id", "name", client, watch_exit=False)
    item.logger = Mock()
    assert item._get_container() is None
    assert not item.enter_states(module.DockerStatus.RUNNING)
    item.terminate()
    assert item.terminal_state is None and not item.can_restart()
    assert item._is_terminating()
    assert item.logger.error.call_count == 1
    container = Mock()
    container.wait.side_effect = RuntimeError("connection dropped")
    item._watch_container_exit(container)
    assert item._exit_watch_done.is_set() and not item._exit_confirmed
    assert item.terminal_state is None
    assert item.poll() == JobReturnCode.UNKNOWN
