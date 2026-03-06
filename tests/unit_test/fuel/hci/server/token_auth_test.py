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
import time

import jwt
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa

from nvflare.fuel.hci.server.token_auth import (
    ClaimMapper,
    ClaimMappingConfig,
    ClaimMappingError,
    TokenValidationConfig,
    TokenValidationError,
    TokenValidator,
)

ISSUER = "https://kc.example.com/realms/nvflare"
AUDIENCE = "nvflare-admin"


@pytest.fixture()
def signing_material():
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    private_key_pem = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    ).decode("ascii")
    public_jwk = json.loads(jwt.algorithms.RSAAlgorithm.to_jwk(private_key.public_key()))
    public_jwk["kid"] = "primary-kid"
    public_jwk["use"] = "sig"
    public_jwk["alg"] = "RS256"
    return private_key_pem, {"keys": [public_jwk]}


def _build_claims(now: int, **overrides):
    claims = {
        "sub": "alice-sub",
        "preferred_username": "alice",
        "org": "org_a",
        "nvf_role": "lead",
        "iss": ISSUER,
        "aud": AUDIENCE,
        "iat": now - 1,
        "nbf": now - 1,
        "exp": now + 300,
    }
    claims.update(overrides)
    return claims


def _make_token(private_key_pem: str, claims: dict, kid="primary-kid", alg="RS256"):
    return jwt.encode(claims, private_key_pem, algorithm=alg, headers={"kid": kid})


def _validator(alg_allowlist=None):
    return TokenValidator(
        TokenValidationConfig(
            issuer=ISSUER,
            audience=AUDIENCE,
            alg_allowlist=alg_allowlist or ["RS256"],
        )
    )


def test_validation_config_enforces_minimum_required_claim_floor():
    config = TokenValidationConfig(
        issuer=ISSUER,
        audience=AUDIENCE,
        required_claims=("iss", "aud"),
    )

    assert config.required_claims == ("iss", "aud", "exp", "iat")


def test_validation_config_rejects_negative_clock_skew():
    with pytest.raises(ValueError, match="non-negative"):
        TokenValidationConfig(
            issuer=ISSUER,
            audience=AUDIENCE,
            clock_skew_seconds=-1,
        )


def test_validate_jwt_success(signing_material):
    now = int(time.time())
    private_key_pem, jwks = signing_material
    token = _make_token(private_key_pem, _build_claims(now))
    claims = _validator().validate(token=token, jwks=jwks, now=now)
    assert claims["sub"] == "alice-sub"
    assert claims["org"] == "org_a"
    assert claims["nvf_role"] == "lead"


def test_validate_jwt_rejects_invalid_issuer(signing_material):
    now = int(time.time())
    private_key_pem, jwks = signing_material
    token = _make_token(private_key_pem, _build_claims(now, iss="https://wrong-issuer"))
    with pytest.raises(TokenValidationError, match="issuer"):
        _validator().validate(token=token, jwks=jwks, now=now)


def test_validate_jwt_rejects_invalid_audience(signing_material):
    now = int(time.time())
    private_key_pem, jwks = signing_material
    token = _make_token(private_key_pem, _build_claims(now, aud="wrong-aud"))
    with pytest.raises(TokenValidationError, match="audience"):
        _validator().validate(token=token, jwks=jwks, now=now)


def test_validate_jwt_rejects_expired_token(signing_material):
    now = int(time.time())
    private_key_pem, jwks = signing_material
    token = _make_token(private_key_pem, _build_claims(now, exp=now - 1))
    with pytest.raises(TokenValidationError, match="expired"):
        _validator().validate(token=token, jwks=jwks, now=now)


def test_validate_jwt_allows_expiry_within_clock_skew(signing_material):
    now = int(time.time())
    private_key_pem, jwks = signing_material
    token = _make_token(private_key_pem, _build_claims(now, exp=now - 1))
    validator = TokenValidator(
        TokenValidationConfig(
            issuer=ISSUER,
            audience=AUDIENCE,
            alg_allowlist=["RS256"],
            clock_skew_seconds=2,
        )
    )

    claims = validator.validate(token=token, jwks=jwks, now=now)
    assert claims["sub"] == "alice-sub"


def test_validate_jwt_rejects_expiry_at_clock_skew_boundary(signing_material):
    now = int(time.time())
    private_key_pem, jwks = signing_material
    token = _make_token(private_key_pem, _build_claims(now, exp=now - 2))
    validator = TokenValidator(
        TokenValidationConfig(
            issuer=ISSUER,
            audience=AUDIENCE,
            alg_allowlist=["RS256"],
            clock_skew_seconds=2,
        )
    )

    with pytest.raises(TokenValidationError, match="expired"):
        validator.validate(token=token, jwks=jwks, now=now)


def test_validate_jwt_rejects_future_nbf(signing_material):
    now = int(time.time())
    private_key_pem, jwks = signing_material
    token = _make_token(private_key_pem, _build_claims(now, nbf=now + 10))
    with pytest.raises(TokenValidationError, match="not before"):
        _validator().validate(token=token, jwks=jwks, now=now)


def test_validate_jwt_allows_nbf_within_clock_skew(signing_material):
    now = int(time.time())
    private_key_pem, jwks = signing_material
    token = _make_token(private_key_pem, _build_claims(now, nbf=now + 2))
    validator = TokenValidator(
        TokenValidationConfig(
            issuer=ISSUER,
            audience=AUDIENCE,
            alg_allowlist=["RS256"],
            clock_skew_seconds=2,
        )
    )

    claims = validator.validate(token=token, jwks=jwks, now=now)
    assert claims["sub"] == "alice-sub"


def test_validate_jwt_rejects_future_iat(signing_material):
    now = int(time.time())
    private_key_pem, jwks = signing_material
    token = _make_token(private_key_pem, _build_claims(now, iat=now + 10))
    with pytest.raises(TokenValidationError, match="issued at"):
        _validator().validate(token=token, jwks=jwks, now=now)


def test_validate_jwt_allows_iat_within_clock_skew(signing_material):
    now = int(time.time())
    private_key_pem, jwks = signing_material
    token = _make_token(private_key_pem, _build_claims(now, iat=now + 2))
    validator = TokenValidator(
        TokenValidationConfig(
            issuer=ISSUER,
            audience=AUDIENCE,
            alg_allowlist=["RS256"],
            clock_skew_seconds=2,
        )
    )

    claims = validator.validate(token=token, jwks=jwks, now=now)
    assert claims["sub"] == "alice-sub"


def test_validate_jwt_rejects_disallowed_algorithm(signing_material):
    now = int(time.time())
    _, jwks = signing_material
    token = jwt.encode(
        _build_claims(now),
        "this-is-a-minimum-length-hs256-secret-key",
        algorithm="HS256",
        headers={"kid": "primary-kid"},
    )
    with pytest.raises(TokenValidationError, match="algorithm"):
        _validator().validate(token=token, jwks=jwks, now=now)


def test_validate_jwt_rejects_missing_kid(signing_material):
    now = int(time.time())
    private_key_pem, jwks = signing_material
    token = jwt.encode(_build_claims(now), private_key_pem, algorithm="RS256")
    with pytest.raises(TokenValidationError, match="kid"):
        _validator().validate(token=token, jwks=jwks, now=now)


def test_validate_jwt_rejects_unknown_kid(signing_material):
    now = int(time.time())
    private_key_pem, jwks = signing_material
    token = _make_token(private_key_pem, _build_claims(now), kid="wrong-kid")
    with pytest.raises(TokenValidationError, match="kid"):
        _validator().validate(token=token, jwks=jwks, now=now)


def test_validate_jwt_allows_missing_nbf_when_not_required(signing_material):
    now = int(time.time())
    private_key_pem, jwks = signing_material
    claims = _build_claims(now)
    claims.pop("nbf", None)
    token = _make_token(private_key_pem, claims)
    validator = TokenValidator(
        TokenValidationConfig(
            issuer=ISSUER,
            audience=AUDIENCE,
            alg_allowlist=["RS256"],
            required_claims=("iss", "aud", "exp", "iat"),
        )
    )
    parsed = validator.validate(token=token, jwks=jwks, now=now)
    assert parsed["sub"] == "alice-sub"


def _mapper():
    return ClaimMapper(
        ClaimMappingConfig(
            user_name_claims=("preferred_username", "email"),
            user_org_claim="org",
            user_role_claim="nvf_role",
            groups_claim="groups",
            role_mappings={"kc_project_admin": "project_admin"},
            group_role_mappings={
                "/nvflare/org-a/lead": "lead",
                "/nvflare/org-a/member": "member",
            },
        )
    )


def test_claim_mapping_with_direct_role():
    claims = {"sub": "alice-sub", "preferred_username": "alice", "org": "org_a", "nvf_role": "lead"}
    mapped = _mapper().map(claims)
    assert mapped.user_name == "alice"
    assert mapped.user_org == "org_a"
    assert mapped.user_role == "lead"
    assert mapped.subject == "alice-sub"


def test_claim_mapping_with_role_value_mapping():
    claims = {"sub": "alice-sub", "email": "alice@orga.com", "org": "org_a", "nvf_role": "kc_project_admin"}
    mapped = _mapper().map(claims)
    assert mapped.user_name == "alice@orga.com"
    assert mapped.user_role == "project_admin"


def test_claim_mapping_with_group_fallback():
    claims = {"sub": "alice-sub", "preferred_username": "alice", "org": "org_a", "groups": ["/nvflare/org-a/lead"]}
    mapped = _mapper().map(claims)
    assert mapped.user_role == "lead"


def test_claim_mapping_rejects_missing_org():
    claims = {"sub": "alice-sub", "preferred_username": "alice", "nvf_role": "lead"}
    with pytest.raises(ClaimMappingError, match="organization"):
        _mapper().map(claims)


def test_claim_mapping_rejects_unmapped_role():
    claims = {"sub": "alice-sub", "preferred_username": "alice", "org": "org_a", "nvf_role": "super_admin"}
    with pytest.raises(ClaimMappingError, match="role"):
        _mapper().map(claims)


def test_claim_mapping_rejects_malformed_groups():
    claims = {"sub": "alice-sub", "preferred_username": "alice", "org": "org_a", "groups": "/nvflare/org-a/lead"}
    with pytest.raises(ClaimMappingError, match="groups"):
        _mapper().map(claims)


def test_claim_mapping_rejects_ambiguous_group_roles():
    claims = {
        "sub": "alice-sub",
        "preferred_username": "alice",
        "org": "org_a",
        "groups": ["/nvflare/org-a/lead", "/nvflare/org-a/member"],
    }
    with pytest.raises(ClaimMappingError, match="ambiguous"):
        _mapper().map(claims)


def test_claim_mapping_rejects_missing_user_name():
    claims = {"sub": "alice-sub", "org": "org_a", "nvf_role": "lead"}
    with pytest.raises(ClaimMappingError, match="user name"):
        _mapper().map(claims)
