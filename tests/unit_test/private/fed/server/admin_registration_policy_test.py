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

from nvflare.fuel.hci.reg import CommandModule
from nvflare.fuel.sec.audit import Auditor
from nvflare.private.fed.server import admin as subject


@pytest.fixture
def setup(monkeypatch):
    values = {}
    for name in (
        "new_command_register_with_builtin_module",
        "SessionManager",
        "LoginModule",
        "AuthzFilter",
        "CommandAudit",
        "NetAgent",
        "NetManager",
    ):
        values[name] = Mock()
        monkeypatch.setattr(subject, name, values[name])
    monkeypatch.setattr(subject.AuditService, "get_auditor", Mock(return_value=Mock(spec=Auditor)))
    monkeypatch.setattr(subject.mpm, "add_cleanup_cb", Mock())
    monkeypatch.setattr(subject.AdminServer, "__init__", Mock(return_value=None))
    return values


def test_admin_registration_preserves_login_authorization_audit_filter_order(setup):
    cell, engine = Mock(), Mock()
    module = Mock(spec=CommandModule)
    admin = subject.FedAdminServer(cell, engine, [module], "/uploads", "/downloads")
    registry = setup["new_command_register_with_builtin_module"].return_value
    assert [c.args[0] for c in registry.add_filter.call_args_list] == [
        setup["LoginModule"].return_value,
        setup["AuthzFilter"].return_value,
        setup["CommandAudit"].return_value,
    ]
    registry.register_module.assert_any_call(module)
    registry.register_module.assert_any_call(setup["SessionManager"].return_value)
    registry.register_module.assert_any_call(setup["NetManager"].return_value)
    assert admin.clients == {} and admin.timeout == 10
    assert admin.sai is engine and admin.cell is cell
    assert subject.mpm.add_cleanup_cb.call_count == 2
    subject.AdminServer.__init__.assert_called_once()


@pytest.mark.parametrize("modules", ["invalid", [object()]])
def test_admin_registration_rejects_invalid_command_modules_before_runtime_initialization(setup, modules):
    with pytest.raises(TypeError, match="cmd_modules"):
        subject.FedAdminServer(Mock(), Mock(), modules, "/uploads", "/downloads")
    subject.AdminServer.__init__.assert_not_called()


def test_admin_registration_requires_an_auditor_before_authorizing_commands(setup, monkeypatch):
    monkeypatch.setattr(subject.AuditService, "get_auditor", Mock(return_value=object()))
    with pytest.raises(TypeError, match="auditor must be Auditor"):
        subject.FedAdminServer(Mock(), Mock(), [], "/uploads", "/downloads")
    setup["CommandAudit"].assert_not_called()
    subject.AdminServer.__init__.assert_not_called()
