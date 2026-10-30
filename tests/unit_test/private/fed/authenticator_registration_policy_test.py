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

from nvflare.apis.fl_context import FLContext
from nvflare.apis.fl_exception import FLCommunicationError
from nvflare.apis.shareable import Shareable
from nvflare.apis.signal import Signal
from nvflare.fuel.f3.cellnet.defs import IdentityChallengeKey, MessageHeaderKey, ReturnCode
from nvflare.fuel.f3.message import Message
from nvflare.private.defs import CellMessageHeaderKeys, ClientRegMsgKey
from nvflare.private.fed import authenticator


@pytest.fixture
def auth(monkeypatch):
    monkeypatch.setattr(authenticator, "_get_client_ip", lambda: "127.0.0.1")
    instance = authenticator.Authenticator(
        Mock(),
        "project",
        "site-1",
        "regular",
        "server.example",
        False,
        "root.pem",
        "key.pem",
        "cert.pem",
        2.0,
        0.1,
        timeout=1.0,
        site_config={"capability": "test"},
    )
    instance.logger = Mock()
    return instance


def _registered(token="token", **extra):
    payload = {
        CellMessageHeaderKeys.TOKEN: token,
        CellMessageHeaderKeys.TOKEN_SIGNATURE: "signature",
        CellMessageHeaderKeys.SSID: "session",
    }
    payload.update(extra)
    return Message(headers={MessageHeaderKey.RETURN_CODE: ReturnCode.OK}, payload=payload)


def test_registration_sends_protocol_identity_site_config_and_public_peer_context(auth):
    auth.cell.send_request.return_value = _registered(_cc_info={"attested": True})
    context = FLContext()
    assert auth.authenticate(context, Signal()) == ("token", "signature", "session", None)
    kwargs = auth.cell.send_request.call_args.kwargs
    assert kwargs["timeout"] == 2.0 and kwargs["optional"] is True
    request = kwargs["request"]
    assert request.get_header(CellMessageHeaderKeys.CLIENT_NAME) == "site-1"
    assert request.get_header(CellMessageHeaderKeys.CLIENT_IP) == "127.0.0.1"
    assert request.get_header(CellMessageHeaderKeys.PROJECT_NAME) == "project"
    assert request.payload[ClientRegMsgKey.SITE_CONFIG] == {"capability": "test"}
    assert request.payload.get_peer_context() is context
    assert context.get_prop("_cc_info") == {"attested": True}


@pytest.mark.parametrize(
    "response",
    [
        object(),
        Message(headers={MessageHeaderKey.RETURN_CODE: ReturnCode.OK}, payload="bad"),
        Message(
            headers={MessageHeaderKey.RETURN_CODE: ReturnCode.UNAUTHENTICATED, MessageHeaderKey.ERROR: "denied"},
            payload={},
        ),
    ],
)
def test_registration_rejects_wrong_wire_type_payload_or_unauthenticated_response(auth, response):
    auth.cell.send_request.return_value = response
    with pytest.raises(FLCommunicationError, match="client_registration"):
        auth.authenticate(FLContext(), Signal())
    assert auth.cell.send_request.call_count == 1


def test_registration_without_token_obeys_finite_timeout_using_isolated_clock(auth, monkeypatch):
    auth.cell.send_request.return_value = _registered(token=None)
    times = iter([0.0, 2.0])
    sleep = Mock()
    monkeypatch.setattr(authenticator, "time", SimpleNamespace(time=lambda: next(times), sleep=sleep))
    with pytest.raises(FLCommunicationError, match="client_registration") as error:
        auth.authenticate(FLContext(), Signal())
    assert "cannot authenticate" in str(error.value.__context__)
    sleep.assert_not_called()


@pytest.mark.parametrize("rc", [ReturnCode.TARGET_UNREACHABLE, ReturnCode.COMM_ERROR])
def test_challenge_transport_failures_request_a_retry(auth, rc):
    auth.cell.send_request.return_value = Message(headers={MessageHeaderKey.RETURN_CODE: rc})
    assert auth.challenge_server() == (None, None)


def test_challenge_refuses_server_identity_mismatch_before_certificate_verification(auth, monkeypatch):
    data = Shareable()
    data.update(
        {
            IdentityChallengeKey.NONCE: "nonce",
            IdentityChallengeKey.CERT: b"chain",
            IdentityChallengeKey.COMMON_NAME: "wrong-server",
        }
    )
    auth.cell.send_request.return_value = Message(headers={MessageHeaderKey.RETURN_CODE: ReturnCode.OK}, payload=data)
    monkeypatch.setattr(authenticator, "load_crt_chain_bytes", lambda data: [object()])
    verifier = Mock()
    monkeypatch.setattr(authenticator, "IdentityVerifier", verifier)
    with pytest.raises(FLCommunicationError, match="expected server identity"):
        auth.challenge_server()
    verifier.assert_not_called()


def test_challenge_verifies_cn_nonce_certificate_chain_and_server_eku(auth, monkeypatch):
    leaf, intermediate = object(), object()
    data = Shareable()
    data.update(
        {
            IdentityChallengeKey.NONCE: "server-nonce",
            IdentityChallengeKey.CERT: b"chain",
            IdentityChallengeKey.COMMON_NAME: "server.example",
            IdentityChallengeKey.SIGNATURE: "signature",
        }
    )
    auth.cell.send_request.return_value = Message(headers={MessageHeaderKey.RETURN_CODE: ReturnCode.OK}, payload=data)
    monkeypatch.setattr(authenticator, "uuid", SimpleNamespace(uuid4=lambda: "client-nonce"))
    monkeypatch.setattr(authenticator, "load_crt_chain_bytes", lambda data: [leaf, intermediate])
    identity_verifier, token_verifier = Mock(), Mock(spec=authenticator.TokenVerifier)
    monkeypatch.setattr(authenticator, "IdentityVerifier", Mock(return_value=identity_verifier))
    monkeypatch.setattr(authenticator, "TokenVerifier", Mock(return_value=token_verifier))
    assert auth.challenge_server() == ("server-nonce", token_verifier)
    kwargs = identity_verifier.verify_common_name.call_args.kwargs
    assert kwargs["asserter_cert"] is leaf and kwargs["intermediate_certs"] == [intermediate]
    assert kwargs["asserted_cn"] == "server.example" and kwargs["nonce"] == "client-nonce"
    assert (
        kwargs["signature"] == "signature" and kwargs["expected_eku"] == authenticator.ExtendedKeyUsageOID.SERVER_AUTH
    )


@pytest.mark.parametrize("valid", [False, True])
def test_secure_registration_asserts_client_identity_and_verifies_received_token(auth, monkeypatch, valid):
    auth.secure_mode = True
    verifier = Mock(spec=authenticator.TokenVerifier)
    verifier.verify.return_value = valid
    auth.challenge_server = Mock(return_value=("server-nonce", verifier))
    asserter = Mock(cert_data=b"client-chain", cn="site-1")
    asserter.sign_common_name.return_value = "signed-nonce"
    monkeypatch.setattr(authenticator, "IdentityAsserter", Mock(return_value=asserter))
    auth.cell.send_request.return_value = _registered()
    if valid:
        assert auth.authenticate(FLContext(), Signal()) == ("token", "signature", "session", verifier)
    else:
        with pytest.raises(RuntimeError, match="invalid token"):
            auth.authenticate(FLContext(), Signal())
    asserter.sign_common_name.assert_called_once_with(nonce="server-nonce")
    payload = auth.cell.send_request.call_args.kwargs["request"].payload
    assert payload[IdentityChallengeKey.CERT] == b"client-chain"
    assert payload[IdentityChallengeKey.SIGNATURE] == "signed-nonce"
    assert payload[IdentityChallengeKey.COMMON_NAME] == "site-1"
    verifier.verify.assert_called_once_with(client_name="site-1", token="token", signature="signature")


def test_secure_registration_abort_and_challenge_timeout_do_not_send_login(auth, monkeypatch):
    auth.secure_mode = True
    auth.challenge_server = Mock(return_value=(None, None))
    signal = Signal()
    signal.trigger(True)
    assert auth.authenticate(FLContext(), signal) == (None, None, None, None)
    auth.cell.send_request.assert_not_called()
    times = iter([0.0, 2.0])
    monkeypatch.setattr(authenticator, "time", SimpleNamespace(time=lambda: next(times), sleep=Mock()))
    with pytest.raises(FLCommunicationError, match="cannot connect"):
        auth.authenticate(FLContext(), Signal())
    auth.cell.send_request.assert_not_called()
