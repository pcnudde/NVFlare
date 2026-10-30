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

from nvflare.apis.fl_constant import ConnectionSecurity
from nvflare.fuel.f3.cellnet import core_cell as subject
from nvflare.fuel.f3.cellnet.defs import MessageHeaderKey, ReturnCode
from nvflare.fuel.f3.drivers.driver_params import DriverParams
from nvflare.fuel.f3.message import Message


@pytest.fixture
def runtime(monkeypatch):
    configurator = Mock()
    configurator.get_max_message_size.return_value = 1024
    monkeypatch.setattr(subject, "CommConfigurator", Mock(return_value=configurator))
    factories = {}
    for name in ("Communicator", "ConnectorManager", "CredentialManager"):
        factories[name] = Mock()
        monkeypatch.setattr(subject, name, factories[name])
    monkeypatch.setattr(subject, "StatsPoolManager", Mock())
    monkeypatch.setattr(subject, "enhance_credential_info", Mock())
    monkeypatch.setattr(subject.CoreCell, "ALL_CELLS", {})
    return factories


@pytest.mark.parametrize(
    "name,root,parent,error",
    [
        ("server", None, "tcp://parent:1234", "root_url is required"),
        ("site-a", None, None, "neither root_url nor parent_url"),
        ("site-a", "invalid", None, "invalid Root URL"),
        ("server", ["invalid"], None, "invalid Root URL"),
        ("site-a", "tcp://root:1234", "invalid", "invalid Parent URL"),
    ],
)
def test_invalid_cell_topology_is_rejected_before_constructing_communication_boundaries(
    runtime, name, root, parent, error
):
    with pytest.raises(ValueError, match=error):
        subject.CoreCell(name, root, False, {}, parent_url=parent)
    runtime["Communicator"].assert_not_called()
    assert not subject.CoreCell.ALL_CELLS


@pytest.mark.parametrize("security,expected", [(ConnectionSecurity.CLEAR, False), (ConnectionSecurity.MTLS, True)])
def test_explicit_connection_security_controls_cell_and_connector_policy(runtime, security, expected):
    credentials = {DriverParams.CONNECTION_SECURITY.value: security}
    cell = subject.CoreCell(
        "site-a",
        "tcp://root:1234",
        not expected,
        credentials,
        max_timeout=-1,
        auth_identity="site-a",
        auth_identity_map={"server": "server-id"},
    )
    assert cell.secure is expected
    assert cell.max_timeout == 3600
    assert subject.CoreCell.ALL_CELLS == {"site-a": cell}
    assert runtime["ConnectorManager"].call_args.kwargs["secure"] is expected
    assert runtime["CredentialManager"].call_args.kwargs["enforce_identity"] is (security == ConnectionSecurity.MTLS)
    runtime["Communicator"].return_value.start.assert_not_called()
    subject.enhance_credential_info.assert_called_once_with(credentials)
    assert cell.identity_resolver is not None


def test_client_root_selection_accepts_one_server_from_configured_url_list(runtime, monkeypatch):
    monkeypatch.setattr(subject, "random", SimpleNamespace(choice=lambda values: values[-1]))
    cell = subject.CoreCell("site-a", ["tcp://one:1", "tcp://two:2"], False, {})
    assert cell.root_url == "tcp://two:2"
    assert cell.identity_resolver is not None
    with pytest.raises(ValueError, match="already a cell named"):
        subject.CoreCell("site-a", "tcp://root:1", False, {})


def test_certificate_exchange_uses_cached_peer_cert_and_validates_nonempty_reply(runtime):
    owner = Mock()
    credentials = Mock()
    exchanger = subject.CertificateExchanger(owner, credentials)
    credentials.get_certificate.return_value = b"cached"
    assert exchanger.get_certificate("peer") == b"cached"
    owner.send_request.assert_not_called()
    credentials.get_certificate.return_value = None
    credentials.process_response.return_value = b"verified"
    response = Message({MessageHeaderKey.RETURN_CODE: ReturnCode.OK}, {"certificate": "data"})
    owner.send_request.return_value = response
    assert exchanger.get_certificate("peer") == b"verified"
    credentials.process_response.assert_called_once_with(response)
    owner.send_request.return_value = Message({MessageHeaderKey.RETURN_CODE: ReturnCode.TIMEOUT}, None)
    with pytest.raises(RuntimeError, match="Cert exchanged to peer failed"):
        exchanger.exchange_certificate("peer")
    request = Message({}, "request")
    reply = exchanger._handle_cert_request(request)
    credentials.process_request.assert_called_once_with(request)
    assert reply.payload is credentials.process_request.return_value
