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

import base64
import datetime
import json

import pytest
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa
from cryptography.x509.oid import NameOID

from nvflare.fuel.sec import security_content_service as subject


@pytest.fixture
def signed_folder(tmp_path, monkeypatch):
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "unit-config")])
    now = datetime.datetime.now(datetime.timezone.utc)
    cert = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(1)
        .not_valid_before(now - datetime.timedelta(days=1))
        .not_valid_after(now + datetime.timedelta(days=1))
        .sign(key, hashes.SHA256())
    )
    (tmp_path / "rootCA.pem").write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    contents = {"valid.json": b'{"value": 3}', "changed.json": b'{"value": 4}', "malformed.json": b"invalid json"}
    signatures = {}
    for filename, data in contents.items():
        signature = key.sign(
            data, padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.MAX_LENGTH), hashes.SHA256()
        )
        signatures[filename] = base64.b64encode(signature).decode()
        (tmp_path / filename).write_bytes(data)
    (tmp_path / "signature.json").write_text(json.dumps(signatures))
    (tmp_path / "changed.json").write_bytes(b'{"value": 9}')
    (tmp_path / "unsigned.json").write_bytes(b'{"value": 5}')
    (tmp_path / "empty.json").write_bytes(b"")
    monkeypatch.setattr(subject.SecurityContentService, "security_content_manager", None)
    monkeypatch.setattr(subject.SecurityContentService, "content_folder", None)
    return tmp_path


@pytest.mark.parametrize(
    "filename, expected",
    [
        ("valid.json", subject.LoadResult.OK),
        ("changed.json", subject.LoadResult.INVALID_SIGNATURE),
        ("unsigned.json", subject.LoadResult.NOT_SIGNED),
        ("missing.json", subject.LoadResult.NO_SUCH_CONTENT),
        ("empty.json", subject.LoadResult.NO_SUCH_CONTENT),
        ("malformed.json", subject.LoadResult.INVALID_CONTENT),
    ],
)
def test_signed_json_distinguishes_signature_content_and_availability_failures(signed_folder, filename, expected):
    manager = subject.SecurityContentManager(str(signed_folder))
    data, result = manager.load_json(filename)
    assert result is expected
    if result is subject.LoadResult.OK:
        assert data == {"value": 3}
    elif result is subject.LoadResult.INVALID_CONTENT:
        assert data is None


def test_service_preserves_first_initialization_and_lists_only_invalid_matching_files(signed_folder):
    service = subject.SecurityContentService
    assert service.load_json("valid.json") == (None, subject.LoadResult.NOT_MANAGED)
    assert service.load_content("valid.json") == (None, subject.LoadResult.NOT_MANAGED)
    assert service.check_json_files(["*.json"]) == []
    service.initialize(str(signed_folder))
    manager = service.security_content_manager
    service.initialize("/not-used")
    assert service.security_content_manager is manager and service.content_folder == str(signed_folder)
    assert service.load_content("valid.json")[1] is subject.LoadResult.OK
    assert service.load_json("valid.json") == ({"value": 3}, subject.LoadResult.OK)
    assert service.check_json_files([]) == []
    invalid = service.check_json_files(
        ["valid.json", "changed.json", "unsigned.json", "malformed.json", "absent*.json"]
    )
    assert set(invalid) == {str(signed_folder / name) for name in ["changed.json", "unsigned.json", "malformed.json"]}
