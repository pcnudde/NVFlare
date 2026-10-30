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

from nvflare.apis.client import Client
from nvflare.apis.fl_component import FLComponent
from nvflare.apis.fl_constant import FLContextKey, ProcessType, ReservedKey, SiteType
from nvflare.apis.fl_context import FLContext
from nvflare.apis.job_def import JobMetaKey
from nvflare.apis.shareable import Shareable
from nvflare.fuel.f3.cellnet.defs import ReturnCode
from nvflare.private.fed.client import client_run_manager as run_module
from nvflare.private.fed.client.client_run_manager import ClientRunManager


@pytest.fixture
def manager():
    manager = ClientRunManager.__new__(ClientRunManager)
    manager.name_to_clients = {"site-1": Client("site-1", "one"), "site-2": Client("site-2", "two")}
    manager.name_to_clients["site-2"].set_fqcn("relay.site-2")
    manager.all_clients = list(manager.name_to_clients.values())
    manager.client = Mock(client_name="site-1")
    manager.aux_runner = Mock()
    manager.object_streamer = Mock()
    manager.logger = Mock()
    manager.components = {"component": object()}
    manager.handlers = []
    manager.widgets = {"widget": object()}
    manager.workspace = object()
    manager.run_info = SimpleNamespace(job_id="job")
    manager.cell = object()
    return manager


def test_constructor_sets_job_identity_and_context_properties(monkeypatch):
    client = SimpleNamespace(client_args={"fqsn": "parent.site-1", "is_leaf": False}, client_name="site-1")
    monkeypatch.setattr(
        ClientRunManager, "create_job_processing_context_properties", lambda *args: {"policy": "strict"}
    )
    manager = ClientRunManager("site-1", "job-1", object(), client, {}, handlers=[])
    with manager.new_context() as context:
        assert context.get_identity_name() == "site-1"
        assert context.get_job_id() == "job-1"
        assert context.get_prop(ReservedKey.FQSN) == "parent.site-1"
        assert context.get_prop(ReservedKey.IS_LEAF) is False
        assert context.get_prop(FLContextKey.PROCESS_TYPE) == ProcessType.CLIENT_JOB
        assert context.get_prop("policy") == "strict"
    assert manager.aux_runner in manager.handlers
    assert manager.object_streamer in manager.handlers
    assert len(manager.widgets) == 2
    assert all(widget in manager.handlers for widget in manager.widgets.values())


@pytest.mark.parametrize("success", [False, True])
def test_task_pull_builds_assignment_only_for_successful_response(manager, success):
    context = FLContext()
    reply = Shareable()
    reply.set_header(FLContextKey.TASK_ID, "task-17")
    manager.client.fetch_task.return_value = (success, "train", reply)
    manager.client.extract_shareable.return_value = reply
    task = manager.get_task_assignment(context, timeout=3)
    manager.client.fetch_task.assert_called_once_with(context, 3)
    if success:
        assert task.name == "train" and task.task_id == "task-17" and task.data is reply
    else:
        assert task is None
        manager.client.extract_shareable.assert_not_called()


@pytest.mark.parametrize("result,expected", [(ReturnCode.OK, True), (ReturnCode.TIMEOUT, False)])
def test_task_push_success_is_determined_by_cell_return_code(manager, result, expected):
    data = Shareable()
    context = FLContext()
    manager.client.push_results.return_value = result
    assert manager.send_task_result(data, context, timeout=4) is expected
    manager.client.push_results.assert_called_once_with(data, context, 4)
    assert manager.logger.error.called is not expected


def test_target_validation_partitions_known_clients_without_reordering(manager):
    valid, invalid = manager.validate_targets(["bad", "site-2", "server", "site-1", "missing"])
    assert valid == ["site-2", "server", "site-1"]
    assert invalid == ["bad", "missing"]
    assert manager.get_clients() == manager.all_clients
    assert manager.get_client_from_name("missing") is None


@pytest.mark.parametrize(
    "names,expected",
    [
        (None, [("server", "server", True)]),
        ("server", [("server", "server", True)]),
        ("SERVER_PARENT", [("server", "server", False)]),
        ("site-2", [("site-2", "relay.site-2", True)]),
        (["site-2", "server"], [("site-2", "relay.site-2", True), ("server", "server", True)]),
        (SiteType.ALL, [("server", "server", True), ("site-2", "relay.site-2", True)]),
    ],
)
def test_auxiliary_targets_preserve_parent_scope_and_omit_self_from_all(manager, names, expected):
    targets = manager._to_aux_msg_targets(names)
    assert [(t.name, t.fqcn, t.job_scoped) for t in targets] == expected


@pytest.mark.parametrize("names", ["missing", ["server", "missing"]])
def test_unknown_auxiliary_target_rejects_the_entire_request(manager, names):
    assert manager._to_aux_msg_targets(names) is None
    manager.logger.error.assert_called_once()
    manager.send_aux_request(names, "topic", Shareable(), 1, FLContext())
    manager.aux_runner.send_aux_request.assert_not_called()


@pytest.mark.parametrize("names", [17, ("server",), ["server", 17]])
def test_auxiliary_targets_reject_non_list_container_or_non_string_entry(manager, names):
    with pytest.raises(TypeError, match="target"):
        manager._to_aux_msg_targets(names)


def test_auxiliary_request_keeps_timeout_and_security_flags(manager):
    context, request = FLContext(), Shareable()
    manager.send_aux_request("site-2", "topic", request, 7, context, optional=True, secure=True)
    args, kwargs = manager.aux_runner.send_aux_request.call_args
    assert args[0][0].fqcn == "relay.site-2"
    assert args[1:] == ("topic", request, 7, context)
    assert kwargs == {"optional": True, "secure": True}


def test_multicast_resolves_all_destinations_before_sending_anything(manager):
    context = FLContext()
    first, second = Shareable(), Shareable()
    assert manager.multicast_aux_requests("topic", {}, 1, context) == {}
    assert manager.multicast_aux_requests("topic", {"server": first, "unknown": second}, 1, context) == {}
    manager.aux_runner.multicast_aux_requests.assert_not_called()
    manager.multicast_aux_requests("topic", {"server": first, "site-2": second}, 2, context, True, True)
    args, kwargs = manager.aux_runner.multicast_aux_requests.call_args
    assert args[0] == "topic" and args[2:] == (2, context)
    assert [(t.name, request) for t, request in args[1]] == [("server", first), ("site-2", second)]
    assert kwargs == {"optional": True, "secure": True}


@pytest.mark.parametrize(
    "meta,error", [(None, "job meta type"), ({}, "missing"), ({JobMetaKey.JOB_CLIENTS: {}}, "expected list")]
)
def test_job_membership_requires_structured_job_metadata(manager, meta, error):
    context = FLContext()
    context.set_prop(FLContextKey.JOB_META, meta)
    with pytest.raises(RuntimeError, match=error):
        manager.get_job_clients(context)


def test_job_membership_populates_lookup_and_private_sticky_hierarchy(manager):
    context = FLContext()
    context.set_prop(FLContextKey.JOB_META, {JobMetaKey.JOB_CLIENTS: [{"name": "alpha"}, {"name": "beta"}]})
    manager.get_job_clients(context)
    assert [c.name for c in manager.all_clients] == ["alpha", "beta"]
    assert manager.get_client_from_name("alpha").name == "alpha"
    assert context.get_prop(FLContextKey.CLIENT_HIERARCHY) is not None


@pytest.mark.parametrize(
    "conf,expected",
    [
        (None, "No configurator"),
        (SimpleNamespace(), "No component build authorizer"),
        (SimpleNamespace(has_component_build_authorizer=lambda: False), "No component build authorizer"),
    ],
)
def test_build_component_requires_an_authorizing_configurator(manager, conf, expected):
    manager.conf = conf
    with pytest.raises(RuntimeError, match=expected):
        manager.build_component({"path": "unsafe.Class"})


def test_component_building_uses_the_authorized_configurator(manager):
    manager.conf = Mock()
    manager.conf.has_component_build_authorizer.return_value = True
    manager.build_component({"path": "allowed.Class"})
    manager.conf.build_component.assert_called_once_with({"path": "allowed.Class"})


def test_streaming_resolves_targets_and_preserves_callbacks(manager):
    context, stream_context, producer, factory = FLContext(), {}, object(), object()
    done, consumed = Mock(), Mock()
    manager.stream_objects("channel", "topic", stream_context, ["site-2"], producer, context, True, True)
    kwargs = manager.object_streamer.stream.call_args.kwargs
    assert kwargs["targets"][0].fqcn == "relay.site-2"
    assert kwargs["producer"] is producer and kwargs["fl_ctx"] is context
    assert kwargs["secure"] is True and kwargs["optional"] is True
    manager.register_stream_processing("channel", "topic", factory, done, consumed, marker="retained")
    manager.object_streamer.register_stream_processing.assert_called_once_with(
        channel="channel", topic="topic", factory=factory, stream_done_cb=done, consumed_cb=consumed, marker="retained"
    )
    manager.shutdown_streamer()
    manager.object_streamer.shutdown.assert_called_once_with()


def test_missing_streamer_is_rejected_without_partial_registration(manager):
    manager.object_streamer = None
    with pytest.raises(RuntimeError, match="not been created"):
        manager.stream_objects("channel", "topic", {}, [], object(), FLContext())
    with pytest.raises(RuntimeError, match="not been created"):
        manager.register_stream_processing("channel", "topic", object())
    manager.shutdown_streamer()


def test_event_handlers_and_job_aborts_use_owned_components(manager, monkeypatch):
    context = FLContext()
    handler = FLComponent()
    manager.add_handler(handler)
    fire = Mock()
    monkeypatch.setattr(run_module, "fire_event", fire)
    manager.fire_event("custom", context)
    fire.assert_called_once_with(event="custom", handlers=[handler], ctx=context)
    runner = Mock(spec=run_module.ClientRunner)
    context.set_prop(FLContextKey.RUNNER, runner)
    manager.abort_app("job", context)
    runner.abort.assert_called_once_with()
    context.set_prop(FLContextKey.RUNNER, object())
    manager.abort_app("job", context)


def test_component_registration_routes_to_client_config_and_retains_lookup(manager):
    component = object()
    manager.add_component("new", component)
    manager.client.runner_config.add_component.assert_called_once_with("new", component)
    assert manager.get_component("component") is manager.components["component"]
    assert manager.get_component("missing") is None
    assert manager.get_all_components() is manager.components
    assert manager.get_workspace() is manager.workspace
    assert manager.get_cell() is manager.cell
    assert manager.get_widget("widget") is manager.widgets["widget"]
    assert manager.get_run_info().job_id == "job"
    assert manager.show_errors() is manager.run_info and manager.reset_errors() is manager.run_info


def test_zero_timeout_auxiliary_send_and_handler_registration(manager):
    context, request, handler = FLContext(), Shareable(), Mock()
    manager.fire_and_forget_aux_request("notice", request, context, optional=True, secure=True)
    args, kwargs = manager.aux_runner.send_aux_request.call_args
    assert args[1:] == ("notice", request, 0.0, context)
    assert args[0][0].name == "server"
    assert kwargs == {"optional": True, "secure": True}
    manager.register_aux_message_handler("notice", handler)
    manager.aux_runner.register_aux_message_handler.assert_called_once_with("notice", handler)
    manager.dispatch("notice", request, context)
    manager.aux_runner.dispatch.assert_called_once_with(topic="notice", request=request, fl_ctx=context)
    manager.persist_components(context, completed=True)
    manager.logger.warning.assert_called_once()
