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

from nvflare.private.fed.utils import identity_utils as module


@pytest.mark.parametrize("organization", [[], [SimpleNamespace(value="trusted-org")]])
def test_certificate_identity_requires_common_name_and_optional_organization_is_explicit(organization):
    cert = Mock()
    cert.subject.get_attributes_for_oid.return_value = []
    with pytest.raises(module.MissingCN):
        module.get_cn_from_cert(cert)
    cert.subject.get_attributes_for_oid.return_value = organization
    assert module.get_org_from_cert(cert) == ("trusted-org" if organization else "")
    with pytest.raises(ValueError, match="expect fqsn"):
        module.get_parent_site_name(7)


@pytest.mark.parametrize("failure", [False, True])
def test_asserter_signature_verification_reports_failure_without_losing_content_binding(monkeypatch, failure):
    asserter = module.IdentityAsserter.__new__(module.IdentityAsserter)
    asserter.cert = Mock()
    verify = Mock(side_effect=ValueError("tampered") if failure else None)
    monkeypatch.setattr(module, "verify_content", verify)
    assert asserter.verify_signature(b"content", b"signature") is (not failure)
    verify.assert_called_once_with(
        content=b"content", signature=b"signature", public_key=asserter.cert.public_key.return_value
    )


@pytest.mark.parametrize("failure", ["name", "signature"])
def test_challenge_verifier_distinguishes_common_name_mismatch_and_invalid_nonce_signature(monkeypatch, failure):
    verifier = module.IdentityVerifier.__new__(module.IdentityVerifier)
    verifier.root_cert, verifier.root_public_key = Mock(), Mock()
    cert = Mock()
    cert.subject.get_attributes_for_oid.return_value = [SimpleNamespace(value="site")]
    monkeypatch.setattr(module, "verify_cert", Mock())
    monkeypatch.setattr(module, "_validate_identity_cert_usage", Mock())
    verify = Mock(side_effect=ValueError("signature mismatch"))
    monkeypatch.setattr(module, "verify_content", verify)
    with pytest.raises(module.CNMismatch if failure == "name" else module.InvalidCNSignature):
        verifier.verify_common_name("other" if failure == "name" else "site", "nonce", cert, b"signature")
    if failure == "name":
        verify.assert_not_called()
    else:
        verify.assert_called_once_with(
            content="sitenonce", signature=b"signature", public_key=cert.public_key.return_value
        )
