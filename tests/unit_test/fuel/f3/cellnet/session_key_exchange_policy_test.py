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
from cryptography.exceptions import InvalidKey, InvalidSignature

from nvflare.fuel.f3.cellnet import cell_cipher as subject
from nvflare.lighter.utils import Identity, generate_cert, generate_keys


@pytest.fixture(scope="module")
def identities():
    root_key, root_public = generate_keys()
    root = generate_cert(
        subject=Identity("root"),
        issuer=Identity("root"),
        signing_pri_key=root_key,
        subject_pub_key=root_public,
        ca=True,
    )
    values = []
    for name in ("sender", "receiver"):
        key, public = generate_keys()
        cert = generate_cert(
            subject=Identity(name), issuer=Identity("root"), signing_pri_key=root_key, subject_pub_key=public
        )
        values.append((key, cert))
    return root, values


def test_authenticated_session_exchange_establishes_the_same_secret_on_both_peers(identities):
    root, ((sender_key, sender_cert), (receiver_key, receiver_cert)) = identities
    sender, receiver = subject.SessionKeyManager(root), subject.SessionKeyManager(root)
    assert not sender.key_available()
    with pytest.raises(subject.SessionKeyUnavailable):
        sender.get_latest_key()
    response = sender.key_request(receiver_cert, sender_cert, sender_key)
    assert len(response) == subject.KEY_ENC_LENGTH + subject.SIGNATURE_LENGTH
    assert receiver.process_key_response(sender_cert, receiver_cert, receiver_key, response)
    key = sender.get_latest_key()
    assert receiver.get_latest_key() == key
    assert receiver.get_key(subject.get_hash(key)[-subject.HASH_LENGTH :]) == key
    assert receiver.get_key(b"unknown") is None
    assert sender.key_available() and receiver.key_available()


def test_session_request_refuses_untrusted_receiver_without_caching_secret(identities, monkeypatch):
    root, ((key, cert), (_other_key, other_cert)) = identities
    manager = subject.SessionKeyManager(root)
    monkeypatch.setattr(manager, "validate_cert_chain", Mock(side_effect=InvalidSignature()))
    assert manager.key_request(other_cert, cert, key) is False
    assert manager.key_hash_dict == {}


@pytest.mark.parametrize("failure", [InvalidKey, InvalidSignature, subject.InvalidCertChain])
def test_failed_session_response_does_not_publish_unverified_key(identities, monkeypatch, failure):
    root, ((key, cert), (_other_key, other_cert)) = identities
    manager = subject.SessionKeyManager(root)
    monkeypatch.setattr(subject, "_asym_dec", Mock(side_effect=failure()))
    assert manager.process_key_response(other_cert, cert, key, b"invalid response") is False
    assert not manager.key_available()


def test_session_signature_verification_accepts_buffer_views(identities):
    _root, ((key, cert), _other) = identities
    signature = subject._sign(key, b"message")
    subject._verify(cert.public_key(), memoryview(b"message"), bytearray(signature))
    with pytest.raises(InvalidSignature):
        subject._verify(cert.public_key(), b"tampered", signature)
