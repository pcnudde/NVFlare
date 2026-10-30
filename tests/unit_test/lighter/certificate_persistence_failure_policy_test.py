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

import os
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from nvflare.lighter.constants import PropKey
from nvflare.lighter.entity import Participant, Project
from nvflare.lighter.impl import cert as module


@pytest.mark.parametrize("explicit_host", [True, False])
def test_long_server_cn_is_truncated_while_preserving_public_connect_host(explicit_host):
    name = "server." + "a" * 60 + ".example"
    server = Participant("server", name, "org", {PropKey.DEFAULT_HOST: "explicit.example"} if explicit_host else {})
    module.CertBuilder._fix_server_name(server)
    assert server.name == server.subject == name[: module.MAX_CN_LENGTH]
    assert server.get_default_host() == ("explicit.example" if explicit_host else name)


def test_listener_certificate_uses_server_identity_and_tightens_private_key_permissions(monkeypatch, tmp_path):
    project = Project("project", "")
    client = project.add_client(
        "site",
        "org",
        {PropKey.LISTENING_HOST: {PropKey.DEFAULT_HOST: "listener.example", PropKey.HOST_NAMES: ["alias.example"]}},
    )
    builder = module.CertBuilder()
    private, certificate = object(), object()
    builder.get_pri_key_cert = Mock(return_value=(private, certificate))
    monkeypatch.setattr(module, "serialize_cert", Mock(return_value=b"certificate-pem"))
    monkeypatch.setattr(module, "serialize_pri_key", Mock(return_value=b"private-pem"))
    context = SimpleNamespace(get_kit_dir=lambda participant: str(tmp_path), get_project=lambda: project)
    builder._build_internal_listener_cert(client, context)
    identity = builder.get_pri_key_cert.call_args.args[0]
    assert identity.type == "server" and identity.name == "site"
    assert identity.get_prop(PropKey.DEFAULT_HOST) == "listener.example"
    assert identity.get_prop(PropKey.HOST_NAMES) == ["alias.example"]
    assert (tmp_path / "server.crt").read_bytes() == b"certificate-pem"
    key = tmp_path / "server.key"
    assert key.read_bytes() == b"private-pem"
    if os.name == "posix":
        assert key.stat().st_mode & 0o777 == 0o600


def test_existing_admin_certificate_cannot_be_reused_for_changed_role(monkeypatch, tmp_path):
    builder = module.CertBuilder()
    state = module._CertState(str(tmp_path))
    admin = Participant("admin", "alice@example.com", "org", {PropKey.ROLE: "project_admin"})
    state.add_subject_cert(admin.subject, "certificate")
    state.add_subject_pri_key(admin.subject, "private-key")
    builder.persistent_state = state
    subject = SimpleNamespace(get_attributes_for_oid=Mock(return_value=[SimpleNamespace(value="member")]))
    certificate = SimpleNamespace(subject=subject)
    monkeypatch.setattr(module, "x509", SimpleNamespace(load_pem_x509_certificate=Mock(return_value=certificate)))
    monkeypatch.setattr(module, "serialization", SimpleNamespace(load_pem_private_key=Mock(return_value=object())))
    context = Mock()
    with pytest.raises(RuntimeError, match="previous role is member but is now project_admin"):
        builder._build_write_cert_pair(admin, "client", context)
    context.get_kit_dir.assert_not_called()
    assert state.get_subject_cert(admin.subject) == "certificate"
    assert state.get_subject_cert("missing") is None


def test_expired_root_validity_error_identifies_requested_subject(monkeypatch):
    builder = module.CertBuilder()
    builder.root_cert = object()
    monkeypatch.setattr(module, "bounded_validity", Mock(side_effect=ValueError("expired yesterday")))
    with pytest.raises(RuntimeError, match="cannot generate certificate for 'site': root CA expired") as caught:
        builder._bounded_not_valid_after("site")
    assert isinstance(caught.value.__cause__, ValueError)
