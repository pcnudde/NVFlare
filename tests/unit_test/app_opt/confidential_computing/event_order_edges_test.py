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

from nvflare.apis.event_type import EventType
from nvflare.apis.fl_constant import ReservedKey
from nvflare.apis.fl_context import FLContext
from nvflare.app_opt.confidential_computing import cc_manager as subject
from nvflare.app_opt.confidential_computing.cc_authorizer import CCAuthorizer


def test_attestation_events_preserve_registration_validation_before_issuing_server_token():
    manager = subject.CCManager([], [], cc_enabled_sites=["server"])
    ctx = FLContext()
    engine = Mock()
    ctx.set_prop(ReservedKey.ENGINE, engine)
    ctx.set_prop(ReservedKey.IDENTITY_NAME, "server")
    manager.handle_event(EventType.SYSTEM_BOOTSTRAP, ctx)
    assert manager.site_name == "server" and manager.engine is engine
    events = []
    manager._validate_client_tokens = lambda context: events.append("validate-client")
    manager._generate_and_attach_tokens = lambda context: events.append("issue")
    manager._validate_server_tokens = lambda context: events.append("validate-server")
    manager._register_cc_handlers = lambda context: events.append("register")
    manager._start_cross_site_validation = lambda context: events.append("start")
    manager._stop_cross_site_validation = lambda: events.append("stop")
    for event in [
        EventType.BEFORE_CLIENT_REGISTER,
        EventType.AFTER_CLIENT_REGISTER,
        EventType.CLIENT_REGISTER_RECEIVED,
        EventType.SYSTEM_START,
        EventType.SYSTEM_END,
    ]:
        manager.handle_event(event, ctx)
    assert events == ["issue", "validate-server", "validate-client", "issue", "register", "start", "stop"]


def test_attestation_configuration_rejects_invalid_verifier_and_false_claims_fail_closed():
    manager = subject.CCManager([], ["verifier"], cc_enabled_sites=["site"])
    ctx = FLContext()
    engine = Mock()
    engine.get_component.return_value = object()
    ctx.set_prop(ReservedKey.ENGINE, engine)
    with pytest.raises(RuntimeError, match="must be a CCAuthorizer"):
        manager._setup_cc_authorizers(ctx)
    verifier = Mock(spec=CCAuthorizer)
    verifier.verify.return_value = False
    manager.cc_verifiers = {"proof": verifier}
    result, invalid = manager._verify_participants_tokens(
        {"site": [{subject.CC_NAMESPACE: "proof", subject.CC_TOKEN: "signed"}]}
    )
    assert result == {} and invalid == ["site namespace: {proof}"]
    verifier.verify.assert_called_once_with("signed")


def test_single_site_validation_uses_own_fresh_tokens_without_network_requests():
    manager = subject.CCManager([], [], cc_enabled_sites=["server"])
    manager.site_name = "server"
    manager._generate_fresh_tokens_for_validation = Mock(return_value=[{"token": "fresh"}])
    manager._get_all_cc_enabled_sites = Mock(return_value=[("server", "server")])
    ctx = FLContext()
    engine = Mock()
    ctx.set_prop(ReservedKey.ENGINE, engine)
    assert manager._collect_all_site_tokens(ctx) == {"server": [{"token": "fresh"}]}
    engine.get_cell.return_value.send_request.assert_not_called()
