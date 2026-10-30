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

from nvflare.lighter.constants import CtxKey, PropKey, ProvFileName, TemplateSectionKey
from nvflare.lighter.impl.edge import EdgeBuilder


@pytest.mark.parametrize("mode", ["branch", "no_port", "no_host", "ready"])
def test_edge_builder_only_generates_resources_for_configured_leaf(mode):
    properties = {PropKey.IS_LEAF: mode != "branch", PropKey.EDGE_SERVICE_PORT: None if mode == "no_port" else 9090}
    client = SimpleNamespace(
        name="leaf",
        get_prop=lambda key, default=None: properties.get(key, default),
        get_listening_host=lambda: None if mode == "no_host" else object(),
    )
    context = Mock()
    context.get_local_dir.return_value = "local-dir"
    project = SimpleNamespace(get_clients=lambda: [client])
    builder = EdgeBuilder()
    builder.initialize(project, context)
    context.load_templates.assert_called_once_with("edge_template.yml")
    builder.build(project, context)
    if mode == "ready":
        context.build_from_template.assert_called_once_with(
            dest_dir="local-dir",
            file_name=ProvFileName.EDGE_RESOURCES_JSON,
            temp_section=TemplateSectionKey.EDGE_LCP_RESOURCES,
            replacement={"host": "0.0.0.0", "port": 9090},
        )
        context.error.assert_not_called()
    else:
        context.build_from_template.assert_not_called()
        assert context.error.call_count == int(mode != "branch")
        if mode != "branch":
            assert (
                PropKey.EDGE_SERVICE_PORT if mode == "no_port" else PropKey.LISTENING_HOST
            ) in context.error.call_args.args[0]


@pytest.mark.parametrize("has_server", [True, False])
def test_he_builder_serializes_client_secrets_separately_from_server_content(monkeypatch, tmp_path, has_server):
    pytest.importorskip("tenseal")
    from nvflare.lighter.impl import he as module

    crypto = Mock()
    crypto.serialize.side_effect = lambda **kwargs: b"client-secret" if kwargs["save_secret_key"] else b"server-public"
    factory = Mock(return_value=crypto)
    monkeypatch.setattr(module.ts, "context", factory)
    server, client = object(), object()
    context = {CtxKey.PUBLIC_ROOT_ONLY: False}
    context = SimpleNamespace(
        get=context.get,
        get_kit_dir=lambda participant: str(tmp_path / ("server" if participant is server else "client")),
    )
    (tmp_path / "server").mkdir()
    (tmp_path / "client").mkdir()
    project = SimpleNamespace(get_server=lambda: server if has_server else None, get_clients=lambda: [client])
    builder = module.HEBuilder(poly_modulus_degree=4096, coeff_mod_bit_sizes=[30, 30], scale_bits=20)
    builder.initialize(project, context)
    assert crypto.global_scale == 2**20
    crypto.generate_relin_keys.assert_called_once_with()
    factory.assert_called_once_with(
        module.ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=4096,
        coeff_mod_bit_sizes=[30, 30],
        encryption_type=module.ts.ENCRYPTION_TYPE.SYMMETRIC,
    )
    builder.build(project, context)
    assert (tmp_path / "client" / ProvFileName.CLIENT_CONTEXT_TENSEAL).read_bytes() == b"client-secret"
    if has_server:
        assert (tmp_path / "server" / ProvFileName.SERVER_CONTEXT_TENSEAL).read_bytes() == b"server-public"
    assert crypto.serialize.call_args.kwargs == dict(
        save_public_key=True, save_secret_key=True, save_galois_keys=False, save_relin_keys=True
    )


def test_he_builder_rejects_public_root_only_before_creating_secret_key(monkeypatch):
    pytest.importorskip("tenseal")
    from nvflare.lighter.impl import he as module

    factory = Mock()
    monkeypatch.setattr(module.ts, "context", factory)
    with pytest.raises(ValueError, match="public-root-only"):
        module.HEBuilder().initialize(object(), {CtxKey.PUBLIC_ROOT_ONLY: True})
    factory.assert_not_called()
