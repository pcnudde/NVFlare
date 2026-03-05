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

import os
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import requests


class KeycloakHarnessError(RuntimeError):
    pass


@dataclass
class KeycloakRuntime:
    base_url: str
    realm: str
    client_id: str
    client_secret: Optional[str]
    expected_audience: str
    openid_config: Dict[str, str]
    started_by_harness: bool
    container_engine: Optional[str] = None
    container_name: Optional[str] = None

    @property
    def issuer(self) -> str:
        return self.openid_config.get("issuer", "")

    @property
    def token_endpoint(self) -> str:
        return self.openid_config.get("token_endpoint", "")

    @property
    def jwks_uri(self) -> str:
        return self.openid_config.get("jwks_uri", "")

    def request_password_token(
        self,
        username: str,
        password: str,
        timeout: float = 10.0,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
    ) -> str:
        data = {
            "grant_type": "password",
            "client_id": client_id or self.client_id,
            "username": username,
            "password": password,
        }
        secret = client_secret if client_secret is not None else self.client_secret
        if secret:
            data["client_secret"] = secret
        response = requests.post(self.token_endpoint, data=data, timeout=timeout)
        if response.status_code != 200:
            raise KeycloakHarnessError(
                f"failed to obtain token for '{username}': HTTP {response.status_code} body={response.text}"
            )
        access_token = response.json().get("access_token")
        if not isinstance(access_token, str) or not access_token:
            raise KeycloakHarnessError(f"token endpoint response missing access_token for '{username}'")
        return access_token

    def get_jwks(self, timeout: float = 10.0) -> Dict:
        response = requests.get(self.jwks_uri, timeout=timeout)
        response.raise_for_status()
        jwks = response.json()
        if not isinstance(jwks, dict):
            raise KeycloakHarnessError("JWKS response is not a dict")
        return jwks

    def exchange_token(
        self,
        subject_token: str,
        subject_issuer: str,
        timeout: float = 10.0,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
    ) -> str:
        data = {
            "grant_type": "urn:ietf:params:oauth:grant-type:token-exchange",
            "client_id": client_id or self.client_id,
            "subject_token": subject_token,
            "subject_token_type": "urn:ietf:params:oauth:token-type:access_token",
            "subject_issuer": subject_issuer,
            "requested_token_type": "urn:ietf:params:oauth:token-type:access_token",
        }
        secret = client_secret if client_secret is not None else self.client_secret
        if secret:
            data["client_secret"] = secret
        response = requests.post(self.token_endpoint, data=data, timeout=timeout)
        if response.status_code != 200:
            raise KeycloakHarnessError(
                f"failed token exchange for issuer '{subject_issuer}': "
                f"HTTP {response.status_code} body={response.text}"
            )
        access_token = response.json().get("access_token")
        if not isinstance(access_token, str) or not access_token:
            raise KeycloakHarnessError(f"token exchange response missing access_token for issuer '{subject_issuer}'")
        return access_token


class KeycloakHarness:
    def __init__(
        self,
        import_path: Path,
        realm: Optional[str] = None,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
        expected_audience: Optional[str] = None,
    ):
        self.import_path = Path(import_path).resolve()
        if not self.import_path.exists():
            raise KeycloakHarnessError(f"realm import path does not exist: {self.import_path}")
        if not (self.import_path.is_file() or self.import_path.is_dir()):
            raise KeycloakHarnessError(f"realm import path must be file or dir: {self.import_path}")

        self.realm = realm or os.environ.get("KEYCLOAK_REALM", "nvflare")
        self.client_id = client_id or os.environ.get("KEYCLOAK_CLIENT_ID", "nvflare-admin")
        self.client_secret = client_secret if client_secret is not None else os.environ.get("KEYCLOAK_CLIENT_SECRET")
        self.expected_audience = expected_audience or os.environ.get("KEYCLOAK_EXPECTED_AUDIENCE", self.client_id)
        self.container_name = os.environ.get("KEYCLOAK_CONTAINER_NAME", "nvflare-keycloak-phase-c")
        self.host_port = int(os.environ.get("KEYCLOAK_PORT", "38080"))
        self.image = os.environ.get("KEYCLOAK_IMAGE", "quay.io/keycloak/keycloak:26.0.7")
        self.features = os.environ.get("KEYCLOAK_FEATURES", "token-exchange")
        self.admin_user = os.environ.get("KEYCLOAK_ADMIN", "admin")
        self.admin_password = os.environ.get("KEYCLOAK_ADMIN_PASSWORD", "admin")
        self.start_timeout_seconds = int(os.environ.get("KEYCLOAK_START_TIMEOUT", "120"))

        self._runtime: Optional[KeycloakRuntime] = None

    def start(self) -> KeycloakRuntime:
        base_url = os.environ.get("KEYCLOAK_BASE_URL")
        if base_url:
            self._runtime = self._make_runtime(base_url=base_url.strip(), started_by_harness=False)
            return self._runtime

        auto_start = os.environ.get("KEYCLOAK_AUTO_START", "0").lower() in ("1", "true", "yes")
        if not auto_start:
            raise KeycloakHarnessError(
                "Keycloak integration disabled. Set KEYCLOAK_BASE_URL or KEYCLOAK_AUTO_START=1 to run Phase C tests."
            )

        engine = self._detect_container_engine()
        self._remove_existing_container(engine)
        self._start_container(engine)
        base_url = f"http://127.0.0.1:{self.host_port}"
        self._runtime = self._make_runtime(base_url=base_url, started_by_harness=True, engine=engine)
        return self._runtime

    def stop(self):
        if not self._runtime or not self._runtime.started_by_harness:
            return
        assert self._runtime.container_engine
        assert self._runtime.container_name
        subprocess.run(
            [self._runtime.container_engine, "rm", "-f", self._runtime.container_name],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        self._runtime = None

    def _make_runtime(self, base_url: str, started_by_harness: bool, engine: Optional[str] = None) -> KeycloakRuntime:
        openid_config = self._wait_for_openid_config(base_url=base_url)
        return KeycloakRuntime(
            base_url=base_url,
            realm=self.realm,
            client_id=self.client_id,
            client_secret=self.client_secret,
            expected_audience=self.expected_audience,
            openid_config=openid_config,
            started_by_harness=started_by_harness,
            container_engine=engine,
            container_name=self.container_name if started_by_harness else None,
        )

    def _detect_container_engine(self) -> str:
        preferred = os.environ.get("KEYCLOAK_CONTAINER_ENGINE")
        candidates = [preferred] if preferred else ["podman", "docker"]
        for candidate in candidates:
            if candidate and shutil.which(candidate) and self._is_engine_usable(candidate):
                return candidate
        raise KeycloakHarnessError(
            "no usable container engine found (tried podman/docker). "
            "Set KEYCLOAK_BASE_URL to use an existing Keycloak instance."
        )

    @staticmethod
    def _is_engine_usable(engine: str) -> bool:
        try:
            proc = subprocess.run(
                [engine, "info"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
                timeout=10,
            )
            return proc.returncode == 0
        except Exception:
            return False

    def _remove_existing_container(self, engine: str):
        subprocess.run(
            [engine, "rm", "-f", self.container_name],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )

    def _start_container(self, engine: str):
        if self.import_path.is_file():
            volume_spec = f"{self.import_path}:/opt/keycloak/data/import/nvflare_realm_phase_c.json:ro"
        else:
            volume_spec = f"{self.import_path}:/opt/keycloak/data/import:ro"

        cmd = [
            engine,
            "run",
            "--detach",
            "--name",
            self.container_name,
            "--publish",
            f"{self.host_port}:8080",
            "--env",
            f"KEYCLOAK_ADMIN={self.admin_user}",
            "--env",
            f"KEYCLOAK_ADMIN_PASSWORD={self.admin_password}",
            "--volume",
            volume_spec,
            self.image,
            "start-dev",
            "--http-port=8080",
            "--import-realm",
        ]
        if self.features:
            cmd.append(f"--features={self.features}")
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if proc.returncode != 0:
            raise KeycloakHarnessError(
                f"failed to start Keycloak container with {engine}: return={proc.returncode} "
                f"stdout={proc.stdout.strip()} stderr={proc.stderr.strip()}"
            )

    def _wait_for_openid_config(self, base_url: str) -> Dict:
        config_url = f"{base_url.rstrip('/')}/realms/{self.realm}/.well-known/openid-configuration"
        deadline = time.time() + self.start_timeout_seconds
        last_error = ""
        while time.time() < deadline:
            try:
                response = requests.get(config_url, timeout=5.0)
                if response.status_code == 200:
                    cfg = response.json()
                    if isinstance(cfg, dict) and "issuer" in cfg and "jwks_uri" in cfg and "token_endpoint" in cfg:
                        return cfg
                    last_error = f"incomplete openid config: {cfg}"
                else:
                    last_error = f"HTTP {response.status_code}"
            except Exception as e:
                last_error = str(e)
            time.sleep(1.0)
        raise KeycloakHarnessError(
            f"timed out waiting for Keycloak realm '{self.realm}' at {config_url}: last_error={last_error}"
        )
