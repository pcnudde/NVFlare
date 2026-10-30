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

import pytest
from cryptography.hazmat.primitives import serialization

from nvflare.fuel.f3.cellnet import credential_manager as subject
from nvflare.fuel.f3.cellnet.defs import MessageHeaderKey
from nvflare.fuel.f3.drivers.driver_params import DriverParams
from nvflare.fuel.f3.endpoint import Endpoint
from nvflare.fuel.f3.message import Message
from nvflare.lighter.utils import Identity, generate_cert, generate_keys


@pytest.fixture(scope="module")
def credentials(tmp_path_factory):
    folder = tmp_path_factory.mktemp("message-credentials")
    root_key, root_public = generate_keys()
    root = generate_cert(
        subject=Identity("root"),
        issuer=Identity("root"),
        signing_pri_key=root_key,
        subject_pub_key=root_public,
        ca=True,
    )
    ca = folder / "root.pem"
    ca.write_bytes(root.public_bytes(serialization.Encoding.PEM))
    managers = []
    for name in ("site-a", "site-b"):
        key, public = generate_keys()
        cert = generate_cert(
            subject=Identity(name), issuer=Identity("root"), signing_pri_key=root_key, subject_pub_key=public
        )
        cert_path, key_path = folder / (name + ".crt"), folder / (name + ".key")
        cert_path.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
        key_path.write_bytes(
            key.private_bytes(
                serialization.Encoding.PEM, serialization.PrivateFormat.TraditionalOpenSSL, serialization.NoEncryption()
            )
        )
        properties = {
            DriverParams.CA_CERT: str(ca),
            DriverParams.CLIENT_CERT: str(cert_path),
            DriverParams.CLIENT_KEY: str(key_path),
        }
        managers.append(subject.CredentialManager(Endpoint(name, conn_props=properties)))
    return managers


def test_certificate_exchange_caches_verified_payload_and_supports_encrypted_roundtrip(credentials):
    sender, receiver = credentials
    sender.clear_cert_cache()
    receiver.clear_cert_cache()
    request = Message(
        {MessageHeaderKey.ORIGIN: "site-a", MessageHeaderKey.DESTINATION: "site-b"}, sender.create_request()
    )
    response = receiver.process_request(request)
    assert receiver.get_certificate("site-a") == sender.local_cert
    certificate = sender.process_response(Message({MessageHeaderKey.ORIGIN: "site-b"}, response))
    assert certificate == receiver.local_cert
    assert sender.get_certificate("site-b") == receiver.local_cert
    encrypted = sender.encrypt(certificate, b"confidential model")
    assert encrypted != b"confidential model"
    assert receiver.decrypt(sender.local_cert, encrypted) == b"confidential model"
    sender.clear_cert_cache()
    receiver.clear_cert_cache()
    assert sender.get_certificate("site-b") is None


def test_certificate_response_errors_and_missing_certificates_do_not_poison_cache(credentials):
    sender, _receiver = credentials
    sender.clear_cert_cache()
    with pytest.raises(RuntimeError, match="failed: denied"):
        sender.process_response(Message({MessageHeaderKey.ORIGIN: "site-b"}, {subject.CERT_ERROR: "denied"}))
    with pytest.raises(RuntimeError, match="missing certificate"):
        sender.process_response(Message({MessageHeaderKey.ORIGIN: "site-b"}, {}))
    assert sender.get_certificate("site-b") is None


def test_clear_mode_explicitly_rejects_secure_message_operations_and_advertises_reason():
    manager = subject.CredentialManager(Endpoint("site-a", conn_props={}))
    for operation, args in (
        (manager.encrypt, (b"cert", b"message")),
        (manager.decrypt, (b"cert", b"cipher")),
        (manager.get_certificate, ("site-b",)),
    ):
        with pytest.raises(RuntimeError, match="not.*secure mode"):
            operation(*args)
    response = manager.process_request(Message({MessageHeaderKey.DESTINATION: "site-a"}, {}))
    assert "not running in secure mode" in response[subject.CERT_ERROR]
    assert manager.read_file(None) is None
