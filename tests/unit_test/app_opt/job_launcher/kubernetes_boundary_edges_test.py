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

from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

pytest.importorskip("kubernetes")

from kubernetes.client.rest import ApiException  # noqa: E402

from nvflare.app_opt.job_launcher import k8s_launcher as subject  # noqa: E402


@pytest.mark.parametrize("name", ["UPPER", "-leading", "trailing-", "x" * 64, ""])
def test_kubernetes_dns_labels_reject_invalid_namespace_before_submission(name):
    with pytest.raises(ValueError, match="DNS label"):
        subject.validate_dns_name("namespace", name)


def test_scheduler_event_timestamp_and_site_labels_are_stable_for_unusual_input():
    assert not subject.is_cpu_memory_gpu_shortage(None)
    assert subject._timestamp_to_seconds("not a time") is None
    instant = datetime(2026, 1, 1, tzinfo=timezone.utc)
    assert subject._timestamp_to_seconds(instant) == instant.timestamp()
    assert not subject._is_recent_event(SimpleNamespace(), instant.timestamp(), 60)
    for name in ("!!!", "9site"):
        label = subject.site_name_to_rfc1123(name)
        assert label[0].isalpha() and len(label) <= 47
        assert label == subject.site_name_to_rfc1123(name)
    assert subject.site_name_to_rfc1123("!!!") != subject.site_name_to_rfc1123("???")


@pytest.mark.parametrize("error", [ApiException(status=500), RuntimeError("transport failed")])
def test_job_handle_contains_secret_cleanup_and_event_query_errors_without_faking_completion(error):
    api = Mock()
    api.delete_namespaced_secret.side_effect = error
    api.list_namespaced_event.side_effect = error
    handle = subject.K8sJobHandle(
        "job", api, {"name": "job", "image": "image", "command": "module"}, credential_secret_name="job-secret"
    )
    handle._delete_credential_secret()
    assert handle.credential_secret_name is None
    api.delete_namespaced_secret.assert_called_once_with(name="job-secret", namespace="default")
    assert handle._query_pod_events() == []
    assert handle._last_event_query_failed
    assert handle.terminal_state is None
    api.list_namespaced_event.side_effect = None
    api.list_namespaced_event.return_value.items = []
    assert handle._query_pod_events() == [] and not handle._last_event_query_failed
