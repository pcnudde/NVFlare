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


@pytest.fixture
def module():
    # Original Docker tests install their SDK boundary during collection. Resolve
    # the production module afterwards so its SDK reference stays consistent.
    from nvflare.app_opt.job_launcher import docker_launcher

    return docker_launcher


def test_missing_docker_backend_is_reported_before_client_connection(module, monkeypatch):
    launcher = module.ClientDockerJobLauncher(workspace="/workspace")
    monkeypatch.setattr(module, "_DOCKER_AVAILABLE", False)
    with pytest.raises(RuntimeError, match="docker SDK not installed"):
        launcher._get_docker_client()
    assert launcher._docker_client is None


def test_docker_network_api_error_does_not_cache_unvalidated_client(module, monkeypatch):
    launcher = module.ClientDockerJobLauncher(workspace="/workspace")
    api = Mock()
    api.networks.get.side_effect = module.docker.errors.APIError("network permission denied")
    monkeypatch.setattr(module.docker, "from_env", Mock(return_value=api))
    with pytest.raises(RuntimeError, match="error checking Docker network"):
        launcher._get_docker_client()
    api.ping.assert_called_once_with()
    assert launcher._docker_client is None


@pytest.mark.parametrize("failure", ["missing", "api-error"])
def test_container_removal_errors_remain_contained_without_claiming_exit_confirmation(module, failure):
    api = Mock()
    error_type = module.docker.errors.NotFound if failure == "missing" else module.docker.errors.APIError
    api.containers.get.side_effect = error_type("container disappeared")
    item = module.DockerJobHandle("id", "name", api, watch_exit=False)
    item.logger = Mock()
    item._remove_container()
    assert item.terminal_state is None and not item._exit_confirmed
    assert item.logger.warning.call_count == (failure == "api-error")
