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

import json
import socket
from unittest.mock import Mock

import grpc
import pytest

from nvflare.tool.package_checker import utils as subject


@pytest.mark.parametrize("failure", [False, True])
def test_address_probe_closes_socket_after_success_or_bind_error(monkeypatch, failure):
    sock = Mock()
    error = OSError("occupied")
    if failure:
        sock.bind.side_effect = error
    monkeypatch.setattr(subject.socket, "socket", Mock(return_value=sock))
    assert subject.try_bind_address("host", 123) is (error if failure else None)
    sock.bind.assert_called_once_with(("host", 123))
    sock.close.assert_called_once()


@pytest.mark.parametrize(
    "scheme, security, ssl_expected",
    [
        ("https", "mtls", True),
        ("stcp", "mtls", True),
        ("http", "mtls", False),
        ("tcp", "mtls", False),
        ("https", "clear", False),
    ],
)
def test_socket_probe_uses_client_tls_only_for_secure_schemes(tmp_path, monkeypatch, scheme, security, ssl_expected):
    (tmp_path / "fed_client.json").write_text(json.dumps({"client": {"connection_security": security}}))
    sock, tls_context, wrapped = Mock(), Mock(), Mock()
    tls_context.wrap_socket.return_value = wrapped
    monkeypatch.setattr(subject.socket, "socket", Mock(return_value=sock))
    make_tls = Mock(return_value=tls_context)
    monkeypatch.setattr(subject.ssl, "create_default_context", make_tls)
    assert subject.check_socket_server_running(str(tmp_path), "host", 123, scheme)
    sock.settimeout.assert_called_once_with(10)
    connected = wrapped if ssl_expected else sock
    connected.connect.assert_called_once_with(("host", 123))
    assert make_tls.call_count == int(ssl_expected)
    if ssl_expected:
        assert tls_context.minimum_version == subject.ssl.TLSVersion.TLSv1_2
        tls_context.load_verify_locations.assert_called_once_with(str(tmp_path / "rootCA.pem"))
        tls_context.load_cert_chain.assert_called_once_with(str(tmp_path / "client.crt"), str(tmp_path / "client.key"))
        tls_context.wrap_socket.assert_called_once_with(sock, server_hostname="host")
        wrapped.close.assert_called_once()
    assert sock.close.called


@pytest.mark.parametrize(
    "error", [ConnectionRefusedError("refused"), socket.timeout("expired"), subject.ssl.SSLError("TLS")]
)
def test_connection_probe_returns_false_and_suppresses_close_error(tmp_path, monkeypatch, error):
    sock = Mock()
    sock.connect.side_effect = error
    sock.close.side_effect = OSError("already closed")
    monkeypatch.setattr(subject.socket, "socket", Mock(return_value=sock))
    assert subject.check_socket_server_running(str(tmp_path), "host", 123, "tcp") is False
    sock.close.assert_called_once()


@pytest.mark.parametrize("secure, ready", [(False, True), (False, False), (True, True)])
def test_grpc_readiness_probe_composes_certificate_and_token_credentials(tmp_path, monkeypatch, secure, ready):
    (tmp_path / "fed_admin.json").write_text(
        json.dumps({"admin": {"connection_security": "mtls" if secure else "clear"}})
    )
    for name in ("rootCA.pem", "client.key", "client.crt"):
        (tmp_path / name).write_bytes(name.encode())
    channel, future = Mock(), Mock()
    if not ready:
        future.result.side_effect = grpc.FutureTimeoutError()
    insecure = Mock(return_value=channel)
    secured = Mock(return_value=channel)
    metadata = Mock(return_value="token-credentials")
    credentials = Mock(return_value="tls-credentials")
    composite = Mock(return_value="combined")
    monkeypatch.setattr(subject.grpc, "insecure_channel", insecure)
    monkeypatch.setattr(subject.grpc, "secure_channel", secured)
    monkeypatch.setattr(subject.grpc, "metadata_call_credentials", metadata)
    monkeypatch.setattr(subject.grpc, "ssl_channel_credentials", credentials)
    monkeypatch.setattr(subject.grpc, "composite_channel_credentials", composite)
    monkeypatch.setattr(subject.grpc, "channel_ready_future", Mock(return_value=future))
    assert subject.check_grpc_server_running(str(tmp_path), "host", 321, "secret-token") is ready
    future.result.assert_called_once_with(timeout=10)
    if secure:
        secured.assert_called_once_with(target="host:321", credentials="combined")
        credentials.assert_called_once_with(
            certificate_chain=b"client.crt", private_key=b"client.key", root_certificates=b"rootCA.pem"
        )
        callback = Mock()
        metadata.call_args.args[0](None, callback)
        callback.assert_called_once_with((("x-custom-token", "secret-token"),), None)
        composite.assert_called_once_with("tls-credentials", "token-credentials")
        insecure.assert_not_called()
    else:
        insecure.assert_called_once_with(target="host:321")
        secured.assert_not_called()


def test_write_probe_removes_only_its_own_directory_and_returns_permission_failure(tmp_path, monkeypatch):
    existing = tmp_path / "existing"
    existing.mkdir()
    (existing / "keep").write_text("keep")
    assert subject.try_write_dir(str(existing)) is None
    assert [p.name for p in existing.iterdir()] == ["keep"]
    new = tmp_path / "new"
    assert subject.try_write_dir(str(new)) is None
    assert not new.exists()
    error = PermissionError("denied")
    monkeypatch.setattr(subject.tempfile, "mkstemp", Mock(side_effect=error))
    assert subject.try_write_dir(str(existing)) is error


def test_subprocess_probe_preserves_quoted_arguments_without_shell(monkeypatch):
    process = Mock()
    popen = Mock(return_value=process)
    monkeypatch.setattr(subject, "popen_in_new_session", popen)
    assert subject.run_command_in_subprocess('python -c "print(1 + 2)"') is process
    assert popen.call_args.args[0] == ["python", "-c", "print(1 + 2)"]
    assert popen.call_args.kwargs["shell"] is False
    assert popen.call_args.kwargs["universal_newlines"] is True
