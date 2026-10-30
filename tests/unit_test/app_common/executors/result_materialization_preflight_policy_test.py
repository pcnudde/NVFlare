# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

from nvflare.apis.event_type import EventType
from nvflare.apis.fl_constant import FLContextKey, ReservedKey
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.apis.signal import Signal
from nvflare.app_common.executors import client_api_executor as subject


@pytest.mark.parametrize(
    "settings", [{"attach_id": "trainer"}, {"allow_insecure_attach": True}, {"params_transfer_type": "invalid"}]
)
def test_external_mode_rejects_attach_only_options_and_invalid_transfer_mode(settings):
    with pytest.raises(ValueError):
        subject.ClientAPIExecutor(execution_mode="external_process", command=["python", "train.py"], **settings)


def test_abort_backend_failure_is_logged_without_escaping_event_handler(monkeypatch):
    executor = subject.ClientAPIExecutor(execution_mode="in_process", task_script_path="train.py")
    backend = Mock()
    backend.abort.side_effect = RuntimeError("abort delivery failed")
    executor._backend = backend
    error = Mock()
    monkeypatch.setattr(executor, "log_error", error)
    context = FLContext()
    executor.handle_event(EventType.ABORT_TASK, context)
    backend.abort.assert_called_once_with(context)
    assert error.call_args.kwargs["fire_event"] is False
    assert "abort delivery failed" in error.call_args.args[1]


def test_materialization_requires_cj_identity_and_shareable_result(monkeypatch):
    context = FLContext()
    with pytest.raises(RuntimeError, match="CJ Cell identity"):
        subject.ClientAPIExecutor._route_result_to_cj(Shareable(), context)
    monkeypatch.setattr(subject, "materialize_lazy_download_refs", Mock(return_value={"not": "shareable"}))
    with pytest.raises(TypeError, match="must be Shareable"):
        subject.ClientAPIExecutor._materialize_result(Shareable(), Mock(), Signal())


@pytest.mark.parametrize("components", [None, {}, {"observer": SimpleNamespace()}])
def test_result_stays_lazy_when_no_local_component_declares_concrete_consumption(components):
    context = FLContext()
    context.set_prop(FLContextKey.TASK_NAME, "train")
    context.set_prop(ReservedKey.ENGINE, SimpleNamespace(get_all_components=lambda: components))
    assert not subject.ClientAPIExecutor._requires_materialized_result("train", context)
