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
from nvflare.apis.fl_constant import FLContextKey
from nvflare.apis.fl_context import FLContext
from nvflare.apis.fl_exception import NotAuthorized
from nvflare.private.fed.server import site_security as subject


@pytest.mark.parametrize("decision", ["allow", "deny", "exception", "unrelated-exception", "internal"])
def test_site_authorization_honors_handler_denials_and_internal_command_bypass(monkeypatch, decision):
    monkeypatch.setattr(subject.InternalCommands, "contains_command", lambda command: decision == "internal")
    ctx = FLContext()
    engine = Mock()

    def authorize(event, context):
        assert event == EventType.AUTHORIZE_COMMAND_CHECK
        assert context.get_prop(FLContextKey.COMMAND_NAME) == "command"
        if decision == "deny":
            context.set_prop(FLContextKey.AUTHORIZATION_RESULT, False, sticky=False)
            context.set_prop(FLContextKey.AUTHORIZATION_REASON, "policy denial", sticky=False)
        elif decision in ("exception", "unrelated-exception"):
            context.set_prop(
                FLContextKey.EXCEPTIONS,
                {"handler": NotAuthorized("denied") if decision == "exception" else RuntimeError("unrelated")},
            )

    engine.fire_event.side_effect = authorize
    result = subject.SiteSecurity().authorization_check(engine, "command", ctx)
    expected = (False, "policy denial") if decision == "deny" else (decision != "exception", "")
    assert result == expected
    assert engine.fire_event.call_count == int(decision != "internal")
    if decision in ("deny", "exception"):
        assert ctx.get_prop(FLContextKey.AUTHORIZATION_RESULT) is None
        assert ctx.get_prop(FLContextKey.AUTHORIZATION_REASON) is None
