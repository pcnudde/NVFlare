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

from nvflare.apis.fl_constant import FLContextKey
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import ReservedHeaderKey
from nvflare.fuel.f3.message import Message as CellMessage
from nvflare.fuel.hci.server.constants import ConnProps
from nvflare.private.admin_defs import Message, MsgHeader, ReturnCode, ok_reply
from nvflare.private.defs import RequestHeader
from nvflare.private.fed.client import admin


@pytest.fixture
def agent(monkeypatch):
    auditor = Mock(spec=admin.Auditor)
    monkeypatch.setattr(admin.AuditService, "get_auditor", lambda: auditor)
    engine, cell, context = Mock(), Mock(), FLContext()
    engine.new_context.side_effect = lambda: context
    agent = admin.FedAdminAgent("site-1", cell, engine)
    processor = Mock(spec=admin.RequestProcessor)
    processor.get_topics.return_value = ["operation"]
    processor.process.return_value = ok_reply(body="processed")
    agent.register_processor(processor)
    agent._test_processor, agent._test_context = processor, context
    monkeypatch.setattr(admin, "secure_log_traceback", Mock())
    return agent


def _request(**headers):
    request = Message("operation", {"payload": 1})
    request.set_headers(headers)
    return CellMessage(headers={}, payload=request)


def test_admin_constructor_requires_auditor_and_registers_control_boundary(agent, monkeypatch):
    kwargs = agent.cell.register_request_cb.call_args.kwargs
    assert kwargs["channel"] == admin.CellChannel.CLIENT_MAIN and kwargs["topic"] == "*"
    assert kwargs["cb"] == agent._dispatch_request
    monkeypatch.setattr(admin.AuditService, "get_auditor", lambda: None)
    with pytest.raises(TypeError, match="Auditor"):
        admin.FedAdminAgent("site-1", Mock(), Mock())


def test_processor_registration_rejects_wrong_types_and_duplicate_topics(agent):
    with pytest.raises(TypeError, match="RequestProcessor"):
        agent.register_processor(object())
    with pytest.raises(AssertionError, match="duplicate processors"):
        agent.register_processor(agent._test_processor)
    assert agent.processors == {"operation": agent._test_processor}


def test_dispatch_records_audit_identity_and_preserves_peer_properties(agent):
    request = _request(
        **{
            RequestHeader.USER_NAME: "alice",
            ConnProps.EVENT_ID: "audit-17",
            ReservedHeaderKey.PEER_PROPS: {"source": "server"},
        }
    )
    reply = agent._dispatch_request(request).payload
    assert reply.body == "processed" and reply.get_header(MsgHeader.RETURN_CODE) == ReturnCode.OK
    agent.auditor.add_event.assert_called_once_with(user="alice", action="operation", ref="audit-17")
    assert agent._test_context.get_peer_context().get_prop("source") == "server"
    agent._test_processor.process.assert_called_once_with(request.payload, agent.app_ctx)


@pytest.mark.parametrize(
    "reply,expected_code,body",
    [
        (None, ReturnCode.OK, "ok"),
        ("invalid", ReturnCode.ERROR, "failed to produce valid reply"),
        (RuntimeError("processor failure"), ReturnCode.ERROR, "processor failure"),
    ],
)
def test_dispatch_acknowledges_none_and_contains_processor_contract_failures(agent, reply, expected_code, body):
    if isinstance(reply, Exception):
        agent._test_processor.process.side_effect = reply
    else:
        agent._test_processor.process.return_value = reply
    response = agent._dispatch_request(_request()).payload
    assert response.get_header(MsgHeader.RETURN_CODE) == expected_code and body in response.body


def test_dispatch_rejects_invalid_payload_and_unknown_topic(agent):
    with pytest.raises(AssertionError, match="CellMessage"):
        agent._dispatch_request(object())
    with pytest.raises(AssertionError, match="payload must be Message"):
        agent._dispatch_request(CellMessage(payload="bad"))
    response = agent._dispatch_request(CellMessage(payload=Message("unknown", "data"))).payload
    assert response.body == "invalid_request" and response.get_header(MsgHeader.RETURN_CODE) == ReturnCode.ERROR
    agent._test_processor.process.assert_not_called()


@pytest.mark.parametrize(
    "site_allowed,service_allowed,service_error,expected",
    [
        (False, True, "", "site denied"),
        (True, False, "", "not authorized"),
        (True, False, "policy failure", "policy failure"),
        (True, True, "", "processed"),
    ],
)
def test_command_requires_site_and_user_authorization_before_processing(
    agent, monkeypatch, site_allowed, service_allowed, service_error, expected
):
    site_security = Mock()
    site_security.authorization_check.return_value = (site_allowed, "site denied")
    monkeypatch.setattr(admin, "SiteSecurity", lambda: site_security)
    authorize = Mock(return_value=(service_allowed, service_error))
    monkeypatch.setattr(admin.AuthorizationService, "authorize", authorize)
    headers = {
        RequestHeader.ADMIN_COMMAND: "abort_job",
        RequestHeader.REQUIRE_AUTHZ: "true",
        RequestHeader.USER_NAME: "alice",
        RequestHeader.USER_ORG: "org",
        RequestHeader.USER_ROLE: "lead",
        RequestHeader.SUBMITTER_NAME: "bob",
        RequestHeader.SUBMITTER_ORG: "org",
        RequestHeader.SUBMITTER_ROLE: "member",
        RequestHeader.JOB_META: {"job": "metadata"},
    }
    response = agent._dispatch_request(_request(**headers)).payload
    assert expected in response.body
    assert agent._test_processor.process.called is (site_allowed and service_allowed and not service_error)
    security = agent._test_context.get_prop(FLContextKey.SECURITY_ITEMS)
    assert security[FLContextKey.USER_NAME] == "alice" and security[FLContextKey.SUBMITTER_NAME] == "bob"
    assert security[FLContextKey.JOB_META] == {"job": "metadata"}
    if site_allowed:
        authz_context = authorize.call_args.args[0]
        assert authz_context.user.name == "alice" and authz_context.submitter.name == "bob"
    else:
        authorize.assert_not_called()


def test_authorization_flag_without_admin_command_is_rejected_before_processing(agent):
    response = agent._dispatch_request(_request(**{RequestHeader.REQUIRE_AUTHZ: "true"})).payload
    assert "missing admin command" in response.body
    agent._test_processor.process.assert_not_called()
