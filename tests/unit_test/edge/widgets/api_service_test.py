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
from unittest.mock import MagicMock, Mock

import pytest

from nvflare.apis.fl_constant import FLContextKey, ReservedKey, SecureTrainConst
from nvflare.apis.fl_context import FLContext
from nvflare.edge.constants import EdgeContextKey, EdgeEventType
from nvflare.edge.widgets import api_service
from nvflare.edge.widgets.api_service import ApiService
from nvflare.fuel.f3.drivers.driver_params import DriverParams


@pytest.mark.parametrize(
    "method,event",
    [
        ("handle_job_request", EdgeEventType.EDGE_JOB_REQUEST_RECEIVED),
        ("handle_task_request", EdgeEventType.EDGE_TASK_REQUEST_RECEIVED),
        ("handle_selection_request", EdgeEventType.EDGE_SELECTION_REQUEST_RECEIVED),
        ("handle_result_report", EdgeEventType.EDGE_RESULT_REPORT_RECEIVED),
    ],
)
@pytest.mark.parametrize("has_reply", [False, True])
def test_request_handlers_publish_request_in_private_context_and_return_event_reply(method, event, has_reply):
    service = ApiService("localhost", 9999)
    ctx, request, reply = FLContext(), object(), {"status": "ok"}
    service.engine = MagicMock()
    service.engine.new_context.return_value.__enter__.return_value = ctx

    def process(event_type, event_ctx):
        assert event_type == event and event_ctx.get_prop(EdgeContextKey.REQUEST_FROM_EDGE) is request
        assert EdgeContextKey.REQUEST_FROM_EDGE not in event_ctx.get_all_public_props()
        if has_reply:
            event_ctx.set_prop(EdgeContextKey.REPLY_TO_EDGE, reply)

    service.fire_event = Mock(side_effect=process)
    assert getattr(service, method)(request) == (reply if has_reply else None)
    service.engine.new_context.return_value.__exit__.assert_called_once()


@pytest.mark.parametrize("secure", [False, True])
def test_startup_builds_credentials_and_starts_only_the_injected_server_boundary(monkeypatch, secure):
    service = ApiService("host", 9999, max_workers=7)
    server, thread = Mock(), Mock()
    factory, threads = Mock(return_value=server), Mock(return_value=thread)
    monkeypatch.setattr(api_service, "EdgeApiServer", factory)
    monkeypatch.setattr(api_service, "threading", SimpleNamespace(Thread=threads))

    def enhance(params):
        if secure:
            params.update({DriverParams.SERVER_CERT.value: "cert", DriverParams.SERVER_KEY.value: "key"})

    monkeypatch.setattr(api_service, "enhance_credential_info", enhance)
    credentials = Mock(return_value="ssl-credentials")
    monkeypatch.setattr(api_service, "get_grpc_server_credentials", credentials)
    ctx = FLContext()
    engine = Mock()
    ctx.set_prop(ReservedKey.ENGINE, engine)
    ctx.set_prop(FLContextKey.CLIENT_CONFIG, {SecureTrainConst.SSL_ROOT_CERT: "ca" if secure else None})
    service._startup("start", ctx)
    assert service.engine is engine and service.server is server
    factory.assert_called_once_with(
        handler=service, address="host:9999", max_workers=7, ssl_credentials="ssl-credentials" if secure else None
    )
    threads.assert_called_once_with(target=server.start, daemon=True)
    thread.start.assert_called_once()
    assert credentials.called is secure
    service._shutdown("end", ctx)
    server.shutdown.assert_called_once()
    service.server = None
    service._shutdown("end", ctx)
