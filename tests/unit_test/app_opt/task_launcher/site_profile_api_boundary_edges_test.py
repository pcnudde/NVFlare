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

pytest.importorskip("kubernetes")

from kubernetes import client, config  # noqa: E402

from nvflare.app_opt.task_launcher.k8s_launcher import K8sTaskBackend  # noqa: E402


def backend(tmp_path, **kwargs):
    values = dict(
        namespace="test",
        image="worker:latest",
        workspace_root=str(tmp_path),
        workspace_pvc="jobs",
        worker_service_account="worker",
        startup_secret_name="startup",
    )
    values.update(kwargs)
    return K8sTaskBackend(**values)


@pytest.mark.parametrize("file", [None, "/explicit/site-kubeconfig"])
def test_backend_initializes_only_its_own_kubernetes_configuration_and_caches_client(tmp_path, monkeypatch, file):
    api_client, core_api = object(), object()
    load_file, load_cluster = Mock(), Mock()
    construct_client, construct_core = Mock(return_value=api_client), Mock(return_value=core_api)
    monkeypatch.setattr(config, "load_kube_config", load_file)
    monkeypatch.setattr(config, "load_incluster_config", load_cluster)
    monkeypatch.setattr(client, "ApiClient", construct_client)
    monkeypatch.setattr(client, "CoreV1Api", construct_core)
    item = backend(tmp_path, config_file_path=file)
    assert item._api() is core_api and item._api() is core_api
    construct_client.assert_called_once()
    configuration = construct_client.call_args.args[0]
    assert isinstance(configuration, client.Configuration)
    construct_core.assert_called_once_with(api_client)
    if file:
        load_file.assert_called_once_with(file, client_configuration=configuration)
        load_cluster.assert_not_called()
    else:
        load_cluster.assert_called_once_with(client_configuration=configuration)
        load_file.assert_not_called()


@pytest.mark.parametrize(
    "options,reason",
    [
        ({"image": " "}, "site configured"),
        ({"python_path": "python"}, "absolute image path"),
        ({"workspace_root": "/"}, "dedicated directory"),
        ({"termination_grace_seconds": True}, "positive integer"),
    ],
)
def test_backend_site_authority_configuration_rejects_unsafe_process_or_storage_defaults(tmp_path, options, reason):
    with pytest.raises(ValueError, match=reason):
        backend(tmp_path, **options)


def test_backend_does_not_accept_untyped_task_requests_or_empty_owner_identity(tmp_path):
    item = backend(tmp_path)
    with pytest.raises(TypeError, match="TaskLaunchRequest"):
        item.launch_task({"argv": ["untrusted"]})
    from nvflare.apis.event_type import EventType
    from nvflare.apis.fl_context import FLContext

    with pytest.raises(ValueError, match="owning CJ's job/site identity"):
        item.handle_event(EventType.START_RUN, FLContext())
    assert item.core_v1 is None and item._job_binding is None
