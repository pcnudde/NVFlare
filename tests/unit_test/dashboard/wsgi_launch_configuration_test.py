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

import runpy
import ssl
from unittest.mock import Mock

import pytest

from nvflare.dashboard import application, utils


@pytest.mark.parametrize("secure", [False, True])
def test_wsgi_launch_uses_configured_port_and_loads_tls_only_with_complete_certificate_pair(
    monkeypatch, tmp_path, secure
):
    app, context = Mock(), Mock()
    monkeypatch.setattr(application, "init_app", Mock(return_value=app))
    monkeypatch.setattr(utils, "get_web_root", lambda: str(tmp_path))
    monkeypatch.setenv(utils.EnvVar.WEB_PORT, "9443")
    certificate = tmp_path / "cert"
    certificate.mkdir()
    (certificate / "web.crt").write_text("certificate placeholder")
    if secure:
        (certificate / "web.key").write_text("key placeholder")
    factory = Mock(return_value=context)
    monkeypatch.setattr(ssl, "SSLContext", factory)
    runpy.run_module("nvflare.dashboard.wsgi", run_name="__main__")
    app.run.assert_called_once_with(host="0.0.0.0", port="9443", ssl_context=context if secure else None)
    if secure:
        factory.assert_called_once_with(ssl.PROTOCOL_TLS_SERVER)
        context.load_cert_chain.assert_called_once_with(str(certificate / "web.crt"), str(certificate / "web.key"))
    else:
        factory.assert_not_called()
