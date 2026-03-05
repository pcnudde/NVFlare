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

import time
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Sequence, Union

import jwt
from jwt import InvalidAudienceError, InvalidIssuerError, InvalidTokenError

DEFAULT_ALLOWED_ROLES = ("platform_admin", "project_admin", "org_admin", "lead", "member")


class TokenValidationError(ValueError):
    pass


class ClaimMappingError(ValueError):
    pass


@dataclass(frozen=True)
class TokenValidationConfig:
    issuer: str
    audience: Union[str, Sequence[str]]
    alg_allowlist: Sequence[str] = ("RS256",)
    clock_skew_seconds: int = 0
    required_claims: Sequence[str] = ("iss", "aud", "exp", "iat", "nbf")


@dataclass(frozen=True)
class ClaimMappingConfig:
    user_name_claims: Sequence[str] = ("preferred_username", "email")
    user_org_claim: str = "org"
    user_role_claim: str = "nvf_role"
    groups_claim: str = "groups"
    role_mappings: Dict[str, str] = field(default_factory=dict)
    group_role_mappings: Dict[str, str] = field(default_factory=dict)
    allowed_roles: Sequence[str] = DEFAULT_ALLOWED_ROLES


@dataclass(frozen=True)
class MappedIdentity:
    subject: str
    user_name: str
    user_org: str
    user_role: str


class TokenValidator:
    def __init__(self, config: TokenValidationConfig):
        if not isinstance(config, TokenValidationConfig):
            raise TypeError(f"config must be TokenValidationConfig but got {type(config)}")
        self.config = config

    def validate(self, token: str, jwks: Mapping[str, Any], now: Optional[float] = None) -> Dict[str, Any]:
        if not isinstance(token, str) or not token:
            raise TokenValidationError("token must be non-empty string")

        try:
            header = jwt.get_unverified_header(token)
        except InvalidTokenError as e:
            raise TokenValidationError(f"invalid token header: {e}") from e

        alg = header.get("alg")
        if alg not in self.config.alg_allowlist:
            raise TokenValidationError(
                f"token algorithm '{alg}' is not allowed: expected one of {list(self.config.alg_allowlist)}"
            )

        kid = header.get("kid")
        if not kid:
            raise TokenValidationError("token header missing kid")

        signing_key = self._resolve_key(jwks=jwks, kid=kid, alg=alg)
        claims = self._decode_with_static_claim_checks(token=token, key=signing_key, alg=alg)
        self._validate_temporal_claims(claims=claims, now=now)
        return claims

    def _resolve_key(self, jwks: Mapping[str, Any], kid: str, alg: str):
        keys = jwks.get("keys") if isinstance(jwks, Mapping) else None
        if not isinstance(keys, list):
            raise TokenValidationError("jwks must contain a 'keys' list")

        for key_data in keys:
            if not isinstance(key_data, Mapping):
                continue
            if key_data.get("kid") != kid:
                continue
            try:
                py_jwk = jwt.PyJWK.from_dict(dict(key_data), algorithm=alg)
                return py_jwk.key
            except Exception as e:
                raise TokenValidationError(f"failed to parse JWK for kid '{kid}': {e}") from e

        raise TokenValidationError(f"unable to resolve signing key for kid '{kid}'")

    def _decode_with_static_claim_checks(self, token: str, key, alg: str) -> Dict[str, Any]:
        options = {
            "require": list(self.config.required_claims),
            "verify_exp": False,
            "verify_iat": False,
            "verify_nbf": False,
        }
        try:
            return jwt.decode(
                token,
                key=key,
                algorithms=[alg],
                issuer=self.config.issuer,
                audience=self.config.audience,
                options=options,
            )
        except InvalidIssuerError as e:
            raise TokenValidationError(f"invalid issuer: {e}") from e
        except InvalidAudienceError as e:
            raise TokenValidationError(f"invalid audience: {e}") from e
        except InvalidTokenError as e:
            raise TokenValidationError(f"invalid token: {e}") from e

    def _validate_temporal_claims(self, claims: Mapping[str, Any], now: Optional[float]):
        clock = float(now if now is not None else time.time())
        skew = float(self.config.clock_skew_seconds)
        exp = _get_optional_numeric_claim(claims, "exp", required=("exp" in self.config.required_claims))
        nbf = _get_optional_numeric_claim(claims, "nbf", required=("nbf" in self.config.required_claims))
        iat = _get_optional_numeric_claim(claims, "iat", required=("iat" in self.config.required_claims))

        if exp is not None and clock >= exp + skew:
            raise TokenValidationError("token expired")
        if nbf is not None and clock + skew < nbf:
            raise TokenValidationError("token not before violation")
        if iat is not None and iat > clock + skew:
            raise TokenValidationError("token issued at time is in the future")


class ClaimMapper:
    def __init__(self, config: ClaimMappingConfig):
        if not isinstance(config, ClaimMappingConfig):
            raise TypeError(f"config must be ClaimMappingConfig but got {type(config)}")
        self.config = config
        self._allowed_roles = set(config.allowed_roles)

    def map(self, claims: Mapping[str, Any]) -> MappedIdentity:
        if not isinstance(claims, Mapping):
            raise ClaimMappingError(f"claims must be mapping but got {type(claims)}")

        subject = _get_string_claim(claims, "sub", "subject")
        user_name = self._resolve_user_name(claims)
        org = _get_string_claim(claims, self.config.user_org_claim, "organization")
        role = self._resolve_role(claims)
        return MappedIdentity(subject=subject, user_name=user_name, user_org=org, user_role=role)

    def _resolve_user_name(self, claims: Mapping[str, Any]) -> str:
        for claim_name in self.config.user_name_claims:
            value = claims.get(claim_name)
            if isinstance(value, str) and value.strip():
                return value.strip()
        raise ClaimMappingError(f"missing required user name claim: expected one of {list(self.config.user_name_claims)}")

    def _resolve_role(self, claims: Mapping[str, Any]) -> str:
        role_value = claims.get(self.config.user_role_claim)
        if role_value is not None:
            return self._map_role_value(role_value)
        return self._map_role_from_groups(claims)

    def _map_role_value(self, role_value: Any) -> str:
        if not isinstance(role_value, str) or not role_value.strip():
            raise ClaimMappingError("role claim must be non-empty string")
        raw_role = role_value.strip()
        mapped_role = self.config.role_mappings.get(raw_role, raw_role)
        if mapped_role not in self._allowed_roles:
            raise ClaimMappingError(f"role '{raw_role}' does not map to a valid NVFlare role")
        return mapped_role

    def _map_role_from_groups(self, claims: Mapping[str, Any]) -> str:
        groups = claims.get(self.config.groups_claim)
        if groups is None:
            raise ClaimMappingError(
                f"missing role claim '{self.config.user_role_claim}' and no groups claim '{self.config.groups_claim}'"
            )
        if not isinstance(groups, (list, tuple)):
            raise ClaimMappingError("groups claim must be a list or tuple")

        matched_roles = set()
        for g in groups:
            if isinstance(g, str):
                mapped = self.config.group_role_mappings.get(g)
                if mapped:
                    if mapped not in self._allowed_roles:
                        raise ClaimMappingError(f"group '{g}' maps to invalid role '{mapped}'")
                    matched_roles.add(mapped)

        if not matched_roles:
            raise ClaimMappingError("no role mapping found from groups")
        if len(matched_roles) > 1:
            raise ClaimMappingError(f"ambiguous group role mapping produced roles: {sorted(matched_roles)}")
        return list(matched_roles)[0]


def _get_numeric_claim(claims: Mapping[str, Any], claim_name: str) -> float:
    value = claims.get(claim_name)
    if not isinstance(value, (int, float)):
        raise TokenValidationError(f"claim '{claim_name}' must be numeric timestamp")
    return float(value)


def _get_optional_numeric_claim(claims: Mapping[str, Any], claim_name: str, required: bool) -> Optional[float]:
    value = claims.get(claim_name)
    if value is None:
        if required:
            raise TokenValidationError(f"claim '{claim_name}' must be numeric timestamp")
        return None
    if not isinstance(value, (int, float)):
        raise TokenValidationError(f"claim '{claim_name}' must be numeric timestamp")
    return float(value)


def _get_string_claim(claims: Mapping[str, Any], claim_name: str, display_name: Optional[str] = None) -> str:
    value = claims.get(claim_name)
    if isinstance(value, str) and value.strip():
        return value.strip()
    label = display_name if display_name else claim_name
    raise ClaimMappingError(f"missing required {label} claim '{claim_name}'")
