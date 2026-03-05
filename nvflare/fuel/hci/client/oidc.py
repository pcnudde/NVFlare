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
import hashlib
import json
import secrets
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Dict, Mapping, Optional
from urllib.parse import parse_qs, urlencode, urlparse
from urllib.request import Request, urlopen

from nvflare.fuel.hci.client.api_spec import AdminConfigKey


def _get_required_non_empty_str(config: Mapping[str, Any], key: str) -> str:
    value = config.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"missing required config '{key}'")
    return value.strip()


def _b64url_encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode("ascii").rstrip("=")


class OIDCTokenManager:
    """Token lifecycle manager for OIDC authorization-code (PKCE) + refresh-token flows."""

    def __init__(self, config: Mapping[str, Any]):
        self.issuer = _get_required_non_empty_str(config, AdminConfigKey.OIDC_ISSUER)
        self.client_id = _get_required_non_empty_str(config, AdminConfigKey.OIDC_CLIENT_ID)

        self.audience = config.get(AdminConfigKey.OIDC_AUDIENCE)
        self.scopes = str(config.get(AdminConfigKey.OIDC_SCOPES, "openid profile email")).strip()
        if not self.scopes:
            self.scopes = "openid profile email"

        self.discovery_url = config.get(AdminConfigKey.OIDC_DISCOVERY_URL)
        self.authorization_endpoint = config.get(AdminConfigKey.OIDC_AUTHORIZATION_ENDPOINT)
        self.token_endpoint = config.get(AdminConfigKey.OIDC_TOKEN_ENDPOINT)

        self.callback_host = str(config.get(AdminConfigKey.OIDC_CALLBACK_HOST, "127.0.0.1")).strip() or "127.0.0.1"
        callback_port = config.get(AdminConfigKey.OIDC_CALLBACK_PORT, 39123)
        self.callback_port = int(callback_port) if callback_port is not None else 39123
        if self.callback_port <= 0:
            raise ValueError("oidc_callback_port must be > 0")

        self.callback_path = str(config.get(AdminConfigKey.OIDC_CALLBACK_PATH, "/callback")).strip() or "/callback"
        if not self.callback_path.startswith("/"):
            self.callback_path = "/" + self.callback_path

        auth_timeout = config.get(AdminConfigKey.OIDC_AUTH_TIMEOUT_SECONDS, 180)
        self.auth_timeout_seconds = float(auth_timeout) if auth_timeout is not None else 180.0
        if self.auth_timeout_seconds <= 0:
            raise ValueError("oidc_auth_timeout_seconds must be > 0")

        refresh_skew = config.get(AdminConfigKey.OIDC_REFRESH_SKEW_SECONDS, 60)
        self.refresh_skew_seconds = float(refresh_skew) if refresh_skew is not None else 60.0
        if self.refresh_skew_seconds < 0:
            raise ValueError("oidc_refresh_skew_seconds must be >= 0")

        self.open_browser = bool(config.get(AdminConfigKey.OIDC_OPEN_BROWSER, True))

        self._lock = threading.Lock()
        self._access_token: Optional[str] = None
        self._refresh_token: Optional[str] = None
        self._access_token_expiry: Optional[float] = None
        self._resolved_endpoints: Optional[Dict[str, str]] = None

    def invalidate_access_token(self):
        with self._lock:
            self._access_token_expiry = 0.0

    def get_access_token(self, allow_browser: bool = True) -> str:
        with self._lock:
            if self._access_token and not self._needs_refresh(now=time.time()):
                return self._access_token

            if self._refresh_token:
                try:
                    token_response = self._refresh_with_refresh_token(self._refresh_token)
                    self._update_tokens(token_response)
                    return self._access_token
                except Exception:
                    # fallback to interactive browser flow
                    pass

            if not allow_browser:
                raise RuntimeError("no valid access token and browser flow is disabled")

            token_response = self._authorize_code_with_browser()
            self._update_tokens(token_response)
            return self._access_token

    def _needs_refresh(self, now: float) -> bool:
        if not self._access_token:
            return True
        if self._access_token_expiry is None:
            return False
        return now + self.refresh_skew_seconds >= self._access_token_expiry

    def _resolve_endpoints(self) -> Dict[str, str]:
        if self._resolved_endpoints:
            return dict(self._resolved_endpoints)

        if self.authorization_endpoint and self.token_endpoint:
            self._resolved_endpoints = {
                "authorization_endpoint": str(self.authorization_endpoint).strip(),
                "token_endpoint": str(self.token_endpoint).strip(),
            }
            return dict(self._resolved_endpoints)

        discovery_url = self.discovery_url
        if not discovery_url:
            discovery_url = f"{self.issuer.rstrip('/')}/.well-known/openid-configuration"

        metadata = self._fetch_json(discovery_url)
        auth_endpoint = metadata.get("authorization_endpoint")
        token_endpoint = metadata.get("token_endpoint")
        if not isinstance(auth_endpoint, str) or not auth_endpoint.strip():
            raise ValueError(f"oidc discovery metadata from '{discovery_url}' missing authorization_endpoint")
        if not isinstance(token_endpoint, str) or not token_endpoint.strip():
            raise ValueError(f"oidc discovery metadata from '{discovery_url}' missing token_endpoint")

        self._resolved_endpoints = {
            "authorization_endpoint": auth_endpoint.strip(),
            "token_endpoint": token_endpoint.strip(),
        }
        return dict(self._resolved_endpoints)

    def _fetch_json(self, url: str) -> Dict[str, Any]:
        with urlopen(url, timeout=self.auth_timeout_seconds) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"url '{url}' did not return a JSON object")
        return payload

    def _post_form(self, url: str, form_data: Dict[str, Any]) -> Dict[str, Any]:
        body = urlencode(form_data).encode("utf-8")
        req = Request(url=url, data=body, method="POST")
        req.add_header("content-type", "application/x-www-form-urlencoded")
        with urlopen(req, timeout=self.auth_timeout_seconds) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"token endpoint '{url}' did not return a JSON object")
        if payload.get("error"):
            error_description = payload.get("error_description", "")
            raise RuntimeError(f"oidc token endpoint error: {payload['error']} {error_description}".strip())
        return payload

    def _refresh_with_refresh_token(self, refresh_token: str) -> Dict[str, Any]:
        endpoints = self._resolve_endpoints()
        form_data = {
            "grant_type": "refresh_token",
            "client_id": self.client_id,
            "refresh_token": refresh_token,
        }
        if isinstance(self.audience, str) and self.audience.strip():
            form_data["audience"] = self.audience.strip()
        return self._post_form(endpoints["token_endpoint"], form_data)

    def _authorize_code_with_browser(self) -> Dict[str, Any]:
        endpoints = self._resolve_endpoints()
        redirect_uri = f"http://{self.callback_host}:{self.callback_port}{self.callback_path}"

        code_verifier = _b64url_encode(secrets.token_bytes(32))
        code_challenge = _b64url_encode(hashlib.sha256(code_verifier.encode("ascii")).digest())
        state = _b64url_encode(secrets.token_bytes(16))

        query = {
            "response_type": "code",
            "client_id": self.client_id,
            "redirect_uri": redirect_uri,
            "scope": self.scopes,
            "state": state,
            "code_challenge": code_challenge,
            "code_challenge_method": "S256",
        }
        if isinstance(self.audience, str) and self.audience.strip():
            query["audience"] = self.audience.strip()

        auth_url = f"{endpoints['authorization_endpoint']}?{urlencode(query)}"
        result = {}
        done = threading.Event()
        expected_path = self.callback_path

        class _CallbackHandler(BaseHTTPRequestHandler):
            def log_message(self, format, *args):
                return

            def do_GET(self):
                parsed = urlparse(self.path)
                if parsed.path != expected_path:
                    self.send_response(404)
                    self.end_headers()
                    return

                params = parse_qs(parsed.query)
                callback_state = params.get("state", [None])[0]
                if callback_state != state:
                    result["error"] = "state_mismatch"
                elif "error" in params:
                    result["error"] = params.get("error", ["unknown_error"])[0]
                    result["error_description"] = params.get("error_description", [""])[0]
                else:
                    result["code"] = params.get("code", [None])[0]
                    if not result["code"]:
                        result["error"] = "missing_code"

                self.send_response(200)
                self.send_header("content-type", "text/html")
                self.end_headers()
                self.wfile.write(b"<html><body><h3>Login complete. You can close this window.</h3></body></html>")
                done.set()

        server = HTTPServer((self.callback_host, self.callback_port), _CallbackHandler)
        server_thread = threading.Thread(target=server.handle_request, daemon=True)
        server_thread.start()
        try:
            print(f"OIDC login URL: {auth_url}")
            if self.open_browser:
                webbrowser.open(auth_url)
            if not done.wait(self.auth_timeout_seconds):
                raise TimeoutError("timeout waiting for OIDC browser callback")
        finally:
            server.server_close()

        if result.get("error"):
            desc = result.get("error_description", "")
            raise RuntimeError(f"oidc authorization failed: {result['error']} {desc}".strip())

        code = result.get("code")
        if not isinstance(code, str) or not code:
            raise RuntimeError("oidc authorization failed: no code received")

        form_data = {
            "grant_type": "authorization_code",
            "client_id": self.client_id,
            "code": code,
            "redirect_uri": redirect_uri,
            "code_verifier": code_verifier,
        }
        if isinstance(self.audience, str) and self.audience.strip():
            form_data["audience"] = self.audience.strip()
        return self._post_form(endpoints["token_endpoint"], form_data)

    def _update_tokens(self, token_response: Mapping[str, Any]):
        access_token = token_response.get("access_token")
        if not isinstance(access_token, str) or not access_token.strip():
            raise ValueError("oidc token response missing access_token")

        self._access_token = access_token.strip()

        refresh_token = token_response.get("refresh_token")
        if isinstance(refresh_token, str) and refresh_token.strip():
            self._refresh_token = refresh_token.strip()

        expires_in = token_response.get("expires_in")
        expiry_time = None
        if isinstance(expires_in, (int, float)):
            expiry_time = time.time() + float(expires_in)
        if expiry_time is None:
            expiry_time = self._extract_exp_from_jwt(self._access_token)
        self._access_token_expiry = expiry_time

    @staticmethod
    def _extract_exp_from_jwt(token: str) -> Optional[float]:
        parts = token.split(".")
        if len(parts) < 2:
            return None
        payload_b64 = parts[1]
        padded = payload_b64 + "=" * (-len(payload_b64) % 4)
        try:
            payload_bytes = base64.urlsafe_b64decode(padded.encode("ascii"))
            payload = json.loads(payload_bytes.decode("utf-8"))
        except Exception:
            return None
        exp = payload.get("exp")
        if isinstance(exp, (int, float)):
            return float(exp)
        return None
