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

import socket
import ssl
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from nvflare.apis.fl_constant import ConnectionSecurity
from nvflare.fuel.f3.comm_error import CommError
from nvflare.fuel.f3.drivers import net_utils as subject
from nvflare.fuel.f3.drivers.driver_params import DriverParams as P
from nvflare.fuel.f3.drivers.grpc import utils as grpc_utils


@pytest.mark.parametrize(
    "server,security,custom",
    [
        (False, ConnectionSecurity.TLS, True),
        (False, ConnectionSecurity.TLS, False),
        (False, ConnectionSecurity.MTLS, False),
        (True, ConnectionSecurity.TLS, False),
        (True, ConnectionSecurity.MTLS, False),
    ],
)
def test_ssl_context_selects_role_credentials_and_client_auth_requirement(monkeypatch, server, security, custom):
    context = Mock()
    ssl_proxy = SimpleNamespace(**vars(ssl))
    ssl_proxy.create_default_context = Mock(return_value=context)
    monkeypatch.setattr(subject, "ssl", ssl_proxy)
    params = {
        P.SCHEME.value: "stcp",
        P.CONNECTION_SECURITY.value: security,
        P.CA_CERT.value: "ca.pem",
        P.SERVER_CERT.value: "server.crt",
        P.SERVER_KEY.value: "server.key",
        P.CLIENT_CERT.value: "client.crt",
        P.CLIENT_KEY.value: "client.key",
    }
    if custom:
        params[P.CUSTOM_CA_CERT.value] = "custom.pem"
    assert subject.get_ssl_context(params, server) is context
    context.load_verify_locations.assert_called_once_with("custom.pem" if custom else "ca.pem")
    if not server and security == ConnectionSecurity.TLS:
        context.load_cert_chain.assert_not_called()
    else:
        context.load_cert_chain.assert_called_once_with(
            certfile="server.crt" if server else "client.crt", keyfile="server.key" if server else "client.key"
        )
    assert context.minimum_version == ssl.TLSVersion.TLSv1_2
    assert context.check_hostname is False
    assert context.verify_mode == (
        ssl.CERT_NONE if server and security == ConnectionSecurity.TLS else ssl.CERT_REQUIRED
    )


@pytest.mark.parametrize("server", [False, True])
def test_ssl_context_rejects_missing_trust_anchor_before_loading_credentials(monkeypatch, server):
    ssl_proxy = SimpleNamespace(**vars(ssl))
    context = Mock()
    ssl_proxy.create_default_context = Mock(return_value=context)
    monkeypatch.setattr(subject, "ssl", ssl_proxy)
    params = {P.SCHEME.value: "stcp", P.SERVER_CERT.value: "cert", P.SERVER_KEY.value: "key"}
    with pytest.raises(CommError, match="certificate parameters are missing"):
        subject.get_ssl_context(params, server)
    context.load_verify_locations.assert_not_called()


@pytest.mark.parametrize("failure", [False, True])
def test_port_probe_closes_fake_socket_on_success_and_bind_failure(monkeypatch, failure):
    stream = Mock()
    if failure:
        stream.bind.side_effect = OSError("occupied")
    proxy = SimpleNamespace(**vars(socket))
    proxy.socket = Mock(return_value=stream)
    monkeypatch.setattr(subject, "socket", proxy)
    assert subject.check_tcp_port(1234) is (not failure)
    stream.bind.assert_called_once_with(("", 1234))
    stream.close.assert_called_once_with()


def test_port_selection_uses_bounded_random_probes_and_reports_exhaustion(monkeypatch):
    probe = Mock(return_value=False)
    monkeypatch.setattr(subject, "check_tcp_port", probe)
    monkeypatch.setattr(subject, "random", SimpleNamespace(randint=lambda low, high: low))
    assert subject.get_open_tcp_port({P.PORTS: "1000-2000"}) is None
    assert probe.call_count == subject.RANDOM_TRIES
    probe.reset_mock()
    probe.side_effect = [False, True]
    assert subject.get_open_tcp_port({P.PORTS: "1000-1002"}) == 1001
    assert [c.args[0] for c in probe.call_args_list] == [1000, 1001]


def test_credential_enhancement_adds_custom_ca_without_overwriting_explicit_role_cert(tmp_path):
    for name in (
        subject.CUSTOM_ROOT_CERT,
        subject.SSL_CLIENT_CERT,
        subject.SSL_CLIENT_PRIVATE_KEY,
        subject.SSL_SERVER_CERT,
        subject.SSL_SERVER_PRIVATE_KEY,
    ):
        (tmp_path / name).write_text("certificate")
    params = {P.CA_CERT.value: str(tmp_path / "root.pem"), P.CLIENT_CERT.value: "explicit-client.crt"}
    subject.enhance_credential_info(params)
    assert params[P.CLIENT_CERT.value] == "explicit-client.crt"
    assert params[P.CUSTOM_CA_CERT.value] == str(tmp_path / subject.CUSTOM_ROOT_CERT)
    assert params[P.SERVER_KEY.value] == str(tmp_path / subject.SSL_SERVER_PRIVATE_KEY)


@pytest.mark.parametrize(
    "security,custom",
    [(ConnectionSecurity.TLS, True), (ConnectionSecurity.TLS, False), (ConnectionSecurity.MTLS, False)],
)
def test_grpc_client_credentials_choose_custom_ca_or_provisioned_mutual_auth(monkeypatch, tmp_path, security, custom):
    for name in ("ca", "custom", "cert", "key"):
        (tmp_path / name).write_bytes(name.encode())
    params = {
        P.CONNECTION_SECURITY.value: security,
        P.CA_CERT.value: str(tmp_path / "ca"),
        P.CLIENT_CERT: str(tmp_path / "cert"),
        P.CLIENT_KEY: str(tmp_path / "key"),
    }
    if custom:
        params[P.CUSTOM_CA_CERT] = str(tmp_path / "custom")
    factory = Mock()
    monkeypatch.setattr(grpc_utils, "grpc", SimpleNamespace(ssl_channel_credentials=factory))
    result = grpc_utils.get_grpc_client_credentials(params)
    assert result is factory.return_value
    kwargs = {"root_certificates": b"custom" if custom else b"ca"}
    if security == ConnectionSecurity.MTLS:
        kwargs.update(certificate_chain=b"cert", private_key=b"key")
    factory.assert_called_once_with(**kwargs)


@pytest.mark.parametrize("security", [ConnectionSecurity.TLS, ConnectionSecurity.MTLS])
def test_grpc_server_credentials_require_client_auth_only_for_mutual_tls(monkeypatch, tmp_path, security):
    for name in ("ca", "cert", "key"):
        (tmp_path / name).write_bytes(name.encode())
    params = {
        P.CONNECTION_SECURITY.value: security,
        P.CA_CERT.value: str(tmp_path / "ca"),
        P.SERVER_CERT: str(tmp_path / "cert"),
        P.SERVER_KEY: str(tmp_path / "key"),
    }
    factory = Mock()
    monkeypatch.setattr(grpc_utils, "grpc", SimpleNamespace(ssl_server_credentials=factory))
    assert grpc_utils.get_grpc_server_credentials(params) is factory.return_value
    factory.assert_called_once_with(
        [(b"key", b"cert")], root_certificates=b"ca", require_client_auth=security == ConnectionSecurity.MTLS
    )


def test_grpc_tls_rejects_missing_ca_and_missing_optional_file_is_absent():
    with pytest.raises(RuntimeError, match="cannot get CA cert"):
        grpc_utils.get_grpc_client_credentials({P.CONNECTION_SECURITY: ConnectionSecurity.TLS})
    assert grpc_utils._read_file(None) is None
