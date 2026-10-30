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
from unittest.mock import MagicMock

import pytest

from nvflare.app_opt.confidential_computing import az_cvm_authorizer as module
from nvflare.app_opt.confidential_computing.az_cvm_authorizer import AZCVMAuthorizer


@pytest.mark.parametrize("returncode, stdout, expected", [(0, b"  signed-token\n", "signed-token"), (1, b"error", "")])
def test_generation_uses_configured_attester_and_suppresses_failed_output(monkeypatch, returncode, stdout, expected):
    run = MagicMock(return_value=SimpleNamespace(returncode=returncode, stdout=stdout))
    monkeypatch.setattr(module, "subprocess", SimpleNamespace(run=run))
    auth = AZCVMAuthorizer(attester_binary="local-attester", maa_endpoint="attest.example")
    assert auth.generate() == expected
    run.assert_called_once_with(
        ["sudo", "local-attester", "-a", "https://attest.example/", "-o", "token"], capture_output=True, check=False
    )
    assert auth.get_namespace() == "x-az-cvm"


@pytest.mark.parametrize("claims, expected", [({"subject": "vm"}, True), ({}, False)])
def test_verification_uses_endpoint_key_and_token_algorithm(monkeypatch, claims, expected):
    key_client = MagicMock()
    key_client.get_signing_key_from_jwt.return_value.key = "public-key"
    factory = MagicMock(return_value=key_client)
    decode = MagicMock(return_value=claims)
    monkeypatch.setattr(module.jwt, "get_unverified_header", lambda token: {"alg": "RS256"})
    monkeypatch.setattr(module.jwt, "PyJWKClient", factory)
    monkeypatch.setattr(module.jwt, "decode", decode)
    assert AZCVMAuthorizer(maa_endpoint="attest.example").verify("signed-token") is expected
    factory.assert_called_once_with("https://attest.example/certs")
    key_client.get_signing_key_from_jwt.assert_called_once_with("signed-token")
    decode.assert_called_once_with("signed-token", "public-key", algorithms=["RS256"])


@pytest.mark.parametrize("stage", ["header", "key", "decode"])
def test_verification_rejects_parse_key_fetch_and_decode_failures(monkeypatch, stage):
    header = MagicMock(return_value={"alg": "RS256"})
    client = MagicMock()
    decode = MagicMock(return_value={"sub": "vm"})
    {"header": header, "key": client.get_signing_key_from_jwt, "decode": decode}[stage].side_effect = ValueError(
        "untrusted token"
    )
    monkeypatch.setattr(module.jwt, "get_unverified_header", header)
    monkeypatch.setattr(module.jwt, "PyJWKClient", lambda endpoint: client)
    monkeypatch.setattr(module.jwt, "decode", decode)
    assert AZCVMAuthorizer().verify("invalid") is False
