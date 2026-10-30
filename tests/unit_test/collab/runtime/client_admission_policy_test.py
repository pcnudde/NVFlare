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

from nvflare.apis.event_type import EventType
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import ReturnCode, Shareable
from nvflare.apis.signal import Signal
from nvflare.collab.api.app import ClientApp
from nvflare.collab.runtime import executor as subject
from nvflare.collab.runtime.defs import SETUP_TASK_NAME, SYNC_TASK_NAME, SyncKey


@pytest.fixture
def runtime(monkeypatch):
    pools = [Mock(), Mock()]
    monkeypatch.setattr(subject, "ThreadPoolExecutor", Mock(side_effect=pools))
    monkeypatch.setattr(subject, "fobs", SimpleNamespace(register=Mock()))
    executor = subject.CollabExecutor("program")
    executor.system_panic = Mock()
    executor.log_error = Mock()
    ctx = Mock(spec=FLContext)
    ctx.get_identity_name.return_value = "site"
    try:
        yield executor, ctx
    finally:
        executor._handle_end_run(EventType.END_RUN, ctx)
        for pool in pools:
            pool.shutdown.assert_called_once_with(wait=True, cancel_futures=True)


@pytest.mark.parametrize("kind", ["missing", "factory", "bad-factory"])
def test_executor_initialization_publishes_only_a_valid_configured_client_app(runtime, kind):
    executor, ctx = runtime
    component = None if kind == "missing" else SimpleNamespace(make_client_app=Mock())
    if component:
        component.make_client_app.return_value = ClientApp(object()) if kind == "factory" else object()
    ctx.get_engine.return_value.get_component.return_value = component
    if kind == "bad-factory":
        with pytest.raises(RuntimeError, match="must be ClientApp"):
            executor._handle_start_run(EventType.START_RUN, ctx)
        assert executor.client_app is None
    else:
        executor._handle_start_run(EventType.START_RUN, ctx)
        if kind == "missing":
            executor.system_panic.assert_called_once()
            assert executor.client_app is None
        else:
            assert executor.client_app.name == "site"
            component.make_client_app.assert_called_once_with("site")
            reply = executor.execute(SYNC_TASK_NAME, Shareable(), ctx, Signal())
            assert reply.get_return_code() == ReturnCode.OK
            assert reply[SyncKey.COLLAB_INTERFACE] == executor.client_app.get_collab_interface()


@pytest.mark.parametrize(
    "payload",
    [{}, {SyncKey.SERVER_FQCN: "server.job"}, {SyncKey.SERVER_FQCN: "server.job", SyncKey.COLLAB_INTERFACE: []}],
)
def test_setup_rejects_missing_authoritative_interfaces_before_registering_runtime(runtime, payload):
    executor, ctx = runtime
    executor.client_app = ClientApp(object(), name="site")
    result = executor.execute(SETUP_TASK_NAME, Shareable(payload), ctx, Signal())
    assert result.get_return_code() == ReturnCode.BAD_TASK_DATA
    ctx.get_engine.assert_not_called()
    assert executor.execute("unknown", Shareable(), ctx, Signal()).get_return_code() == ReturnCode.TASK_UNKNOWN


def test_server_proxy_has_named_object_interfaces_and_shared_outbound_dispatcher(runtime, monkeypatch):
    executor, ctx = runtime
    executor.client_app = ClientApp(object(), name="site")
    factory = Mock()
    monkeypatch.setattr(subject, "CellDispatcher", factory)
    cell, signal = object(), Signal()
    proxy = executor._prepare_server_proxy("server.job", cell, {"": {"main": []}, "helper": {"work": []}}, signal, ctx)
    assert proxy.name == "server"
    assert proxy.get_child("helper").name == "server.helper"
    assert proxy.get_child("helper").backend is proxy.backend
    assert proxy.get_child("helper").target_interface.get_method("work") is not None
    assert factory.call_args.kwargs["thread_executor"] is executor.outbound_executor
    assert factory.call_args.kwargs["target_fqcn"] == "server.job"
