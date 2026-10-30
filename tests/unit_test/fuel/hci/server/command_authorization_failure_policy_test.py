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

from unittest.mock import Mock

import pytest

from nvflare.fuel.hci.conn import Connection
from nvflare.fuel.hci.proto import MetaStatusValue, ProtoKey, validate_proto
from nvflare.fuel.hci.reg import CommandEntry
from nvflare.fuel.hci.server import authz as subject
from nvflare.fuel.hci.server import login
from nvflare.fuel.hci.server.constants import ConnProps
from nvflare.fuel.hci.server.sess import SessionManager


@pytest.mark.parametrize("policy", ["no-command", "no-check", "preapproved", "pre-rejected", "service-error", "denied"])
def test_command_authorization_stops_before_handler_when_policy_rejects(monkeypatch, policy):
    connection = Connection()
    authorize = Mock(return_value=(False, "policy unavailable" if policy == "service-error" else None))
    monkeypatch.setattr(subject.AuthorizationService, "authorize", authorize)
    if policy != "no-command":
        check = (
            None
            if policy == "no-check"
            else Mock(
                return_value={
                    "preapproved": subject.PreAuthzReturnCode.OK,
                    "pre-rejected": subject.PreAuthzReturnCode.ERROR,
                }.get(policy, subject.PreAuthzReturnCode.REQUIRE_AUTHZ)
            )
        )
        entry = CommandEntry("scope", "command", "description", "usage", Mock(), check, True, None, None)
        connection.set_prop(ConnProps.CMD_ENTRY, entry)
    accepted = subject.AuthzFilter().pre_command(connection, ["command"])
    assert accepted == (policy in ("no-command", "no-check", "preapproved"))
    assert authorize.call_count == int(policy in ("service-error", "denied"))
    if policy in ("service-error", "denied"):
        reply = validate_proto(connection.close())
        assert reply[ProtoKey.DATA][0][ProtoKey.TYPE] == ProtoKey.ERROR
        assert reply[ProtoKey.META]["status"] == MetaStatusValue.NOT_AUTHORIZED


@pytest.mark.parametrize("problem", ["args", "invalid-certificate", "study-type", "study-name"])
def test_cert_login_rejections_are_explicit_before_session_creation(monkeypatch, problem):
    module = login.LoginModule(Mock(spec=SessionManager))
    connection, hci = Connection(), Mock()
    connection.set_prop(ConnProps.HCI_SERVER, hci)
    connection.set_prop(
        ConnProps.CMD_HEADERS,
        {"cert": b"certificate", "signature": "signature", "study": 3 if problem == "study-type" else "Bad Study"},
    )
    hci.get_id_verifier.return_value.verify_common_name.return_value = True
    monkeypatch.setattr(login, "load_crt_chain_bytes", Mock(return_value=[Mock()]))
    monkeypatch.setattr(login, "get_admin_study_entitlements", Mock(return_value=[]))
    validate = Mock(side_effect=ValueError("invalid admin leaf") if problem == "invalid-certificate" else None)
    monkeypatch.setattr(login, "validate_admin_leaf_cert", validate)
    module.handle_cert_login(connection, ["login"] if problem == "args" else ["login", "owner"])
    reply = validate_proto(connection.close())
    value = reply[ProtoKey.DATA][0][ProtoKey.DATA]
    assert value.startswith("REJECT")
    if problem.startswith("study"):
        assert ("AUTH_INVALID_STUDY_NAME" if problem == "study-name" else "AUTH_INVALID_STUDY") in value
    module.session_mgr.create_session.assert_not_called()
