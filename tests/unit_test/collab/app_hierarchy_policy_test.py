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

import pytest

from nvflare.apis.signal import Signal
from nvflare.apis.workspace import Workspace
from nvflare.collab import collab
from nvflare.collab.api.app import App, ClientApp, ServerApp
from nvflare.collab.api.constants import ContextKey
from nvflare.collab.api.context import get_call_context, set_call_context
from nvflare.collab.api.proxy import Proxy


class LifecycleObject:
    def __init__(self):
        self.events = []

    @collab.init
    def initialize(self, context):
        self.events.append(("init", context))

    @collab.final
    def finalize(self, context):
        self.events.append(("final", context))

    @collab.main
    def run(self):
        return 7

    @collab.publish
    def exchange(self, value):
        return value + 1

    def private(self):
        return "private"


@pytest.fixture
def hierarchy(tmp_path):
    previous = get_call_context()
    (tmp_path / "startup").mkdir()
    (tmp_path / "local").mkdir()
    workspace = Workspace(str(tmp_path))
    app = ClientApp(LifecycleObject(), name="branch")
    backend = object()
    proxies = [
        Proxy(app, name, fqn, backend, {})
        for name, fqn in [("branch", "branch"), ("leaf", "branch.leaf"), ("other", "other")]
    ]
    server = Proxy(app, "server", "server", backend, {})
    abort = Signal()
    app.setup(workspace, server, proxies, abort)
    try:
        yield app, server, proxies, abort
    finally:
        set_call_context(previous)


def test_client_hierarchy_and_facade_return_defensive_site_views(hierarchy):
    app, server, proxies, abort = hierarchy
    app.set_fqn("branch")
    app.update_props({"rounds": 3})
    app.update_props(None)
    assert app.fqn == "branch"
    assert app.backend is proxies[0].backend
    assert app.get_props() == {"rounds": 3}
    assert app.get_children() == [proxies[1]]
    assert app.has_children()
    assert app.get_leaf_clients() == [proxies[1], proxies[2]]
    context = app.new_context("server", "branch.exchange")
    context.set_prop(ContextKey.RESULT, 23)
    assert collab.callee == "branch.exchange"
    assert collab.call_info == "branch:server=>branch.exchange"
    assert collab.server is server
    assert list(collab.other_clients) == proxies[1:]
    assert list(collab.clients) == proxies
    assert list(collab.child_clients) == [proxies[1]]
    assert list(collab.leaf_clients) == [proxies[1], proxies[2]]
    assert collab.has_children
    assert collab.get_result() == 23
    collab.set_prop("step", 2)
    assert collab.get_prop("step") == 2
    assert collab.get_prop("missing", "fallback") == "fallback"
    assert collab.get_clients(["leaf", "branch"])[0] is proxies[1]
    with pytest.raises(RuntimeError, match="no client 'absent'"):
        collab.get_clients(["absent"])
    copy = app.client_proxies
    copy.clear()
    assert app.client_proxies == proxies
    abort.trigger(True)
    assert collab.is_aborted


def test_leaf_and_server_views_validate_their_required_hierarchy(hierarchy):
    app, server, proxies, abort = hierarchy
    leaf = ClientApp(object(), name="leaf")
    leaf.setup(app.workspace, server, proxies, abort)
    assert leaf.get_children() == []
    assert not leaf.has_children()
    leaf.new_context("server", "leaf")
    with pytest.raises(RuntimeError, match="no child clients"):
        _ = collab.child_clients
    isolated = App(object(), "isolated")
    assert isolated.backend is None
    assert isolated.get_children() == []
    assert not isolated.has_children()
    with pytest.raises(RuntimeError, match="must be Forest"):
        isolated.get_leaf_clients()
    server_app = ServerApp(LifecycleObject())
    with pytest.raises(RuntimeError, match="must be Forest"):
        server_app.get_children()
    server_app.setup(app.workspace, server, proxies, abort)
    assert server_app.has_children()
    assert server_app.get_children() == [proxies[0], proxies[2]]
    missing = ClientApp(object(), "absent")
    with pytest.raises(ValueError, match="cannot find site"):
        missing.setup(app.workspace, server, proxies, abort)
    with pytest.raises(RuntimeError, match="must be Forest"):
        ClientApp(object()).get_children()
    leaf._client_hierarchy.nodes.pop("leaf")
    with pytest.raises(RuntimeError, match="must be a Node"):
        leaf.has_children()
    leaf._client_hierarchy.nodes.clear()
    leaf.new_context("server", "leaf")
    with pytest.raises(RuntimeError, match="no leaf clients"):
        _ = collab.leaf_clients


@pytest.mark.parametrize(
    "name,match", [("1bad", "invalid name"), ("branch", "existing collab"), ("workspace", "reserved name")]
)
def test_app_rejects_ambiguous_collaboration_target_names(name, match):
    app = ClientApp(object(), "branch")
    with pytest.raises(ValueError, match=match):
        app.add_collab_object(name, object())


def test_app_lifecycle_preserves_context_and_resolves_only_published_methods(hierarchy):
    app, _, _, _ = hierarchy
    child = LifecycleObject()
    app.add_collab_object("extra", child)
    prior = app.new_context("prior", "prior")
    ctx = app.new_context("server", "branch", set_call_ctx=False)
    app.initialize(ctx)
    app.finalize(ctx)
    assert get_call_context() is prior
    assert app.obj.events == [("init", ctx), ("final", ctx)]
    assert child.events == [("init", ctx), ("final", ctx)]
    assert app.find_method(app, "exchange")(4) == 5
    child.unique = lambda: 9
    assert app.find_method(app, "unique")() == 9
    assert app.find_method(app, "missing") is None
    assert app.find_collab_method(app, "private") is None
    assert app.get_target_object_publish_interface(None) is app.get_target_object_publish_interface("APP")
    assert app.get_target_object_publish_interface("extra").get_method("exchange") is not None
    with pytest.raises(TypeError, match="workspace must"):
        app.setup(object(), None, [], None)


@pytest.mark.parametrize("cls", [ClientApp, ServerApp])
def test_app_requires_a_primary_object(cls):
    with pytest.raises(ValueError, match="object must be specified"):
        cls(None)
