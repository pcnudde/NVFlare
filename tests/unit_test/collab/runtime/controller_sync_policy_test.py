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

from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import ReturnCode, Shareable
from nvflare.apis.signal import Signal
from nvflare.collab import collab
from nvflare.collab.api.app import ServerApp
from nvflare.collab.runtime import controller as subject
from nvflare.collab.runtime.defs import SETUP_TASK_NAME, SYNC_TASK_NAME, SyncKey


class ServerProgram:
    @collab.main
    def run(self):
        return "finished"


@pytest.fixture
def runtime(monkeypatch):
    pools = []

    def pool(**kwargs):
        result = Mock()
        pools.append(result)
        return result

    monkeypatch.setattr(subject, "ThreadPoolExecutor", pool)
    # Registration is a global boundary; the real optional class can still be located.
    monkeypatch.setattr(subject, "fobs", SimpleNamespace(register=Mock()))
    controller = subject.CollabController(server_obj_id="program", collab_obj_ids=["helper"], props={"rounds": 2})
    controller.system_panic = Mock()
    controller.log_info = Mock()
    controller.log_error = Mock()
    engine = Mock()
    components = {"program": ServerProgram(), "helper": object()}
    engine.get_component.side_effect = components.get
    engine.get_cell.return_value.get_fqcn.return_value = "server.job"
    clients = [SimpleNamespace(name="alpha", get_fqcn=lambda: "alpha", get_fqsn=lambda: "alpha")]
    engine.get_clients.return_value = clients
    ctx = Mock(spec=FLContext)
    ctx.get_engine.return_value = engine
    ctx.get_job_id.return_value = "job"
    try:
        yield controller, engine, components, clients, ctx
    finally:
        controller.stop_controller(ctx)
        for executor in pools:
            executor.shutdown.assert_called_once_with(wait=True, cancel_futures=True)


@pytest.mark.parametrize("failure", [None, "missing-primary", "missing-helper"])
def test_controller_initialization_rejects_missing_objects_and_populates_real_app(runtime, failure):
    controller, _, components, _, ctx = runtime
    if failure:
        components.pop("program" if failure == "missing-primary" else "helper")
    controller.start_controller(ctx)
    if failure:
        assert controller.server_app is None
        controller.system_panic.assert_called_once()
        assert (
            "no component" if failure == "missing-primary" else "does not exist"
        ) in controller.system_panic.call_args.args[0]
    else:
        assert isinstance(controller.server_app, ServerApp)
        assert controller.server_app.get_prop("rounds") == 2
        assert controller.server_app.get_collab_objects()["helper"] is components["helper"]
        controller.system_panic.assert_not_called()


@pytest.mark.parametrize("stage", ["success", "sync-failure", "setup-failure"])
def test_controller_requires_sync_and_setup_quorums_before_starting_server(runtime, monkeypatch, stage):
    controller, engine, _, clients, ctx = runtime
    controller.server_app = Mock()
    controller.server_app.name = "server"
    controller.server_app.get_collab_interface.return_value = {"": {"work": []}, "helper": {"query": []}}
    app = controller.server_app
    app.get_collab_objects.return_value = {"helper": object()}
    remote = Mock()
    run = Mock()
    monkeypatch.setattr(subject, "prepare_for_remote_call", remote)
    monkeypatch.setattr(subject, "run_server", run)
    dispatcher = Mock()
    monkeypatch.setattr(subject, "CellDispatcher", dispatcher)
    tasks = []

    def broadcast(**kwargs):
        task = kwargs["task"]
        tasks.append(task)
        result = Shareable({SyncKey.COLLAB_INTERFACE: {"": {"work": []}, "extra": {"query": []}}})
        failed = (stage == "sync-failure" and task.name == SYNC_TASK_NAME) or (
            stage == "setup-failure" and task.name == SETUP_TASK_NAME
        )
        result.set_return_code(ReturnCode.EXECUTION_EXCEPTION if failed else ReturnCode.OK)
        task.result_received_cb(SimpleNamespace(client=clients[0], result=result), ctx)

    controller.broadcast_and_wait = Mock(side_effect=broadcast)
    controller.control_flow(Signal(), ctx)
    assert tasks[0].name == SYNC_TASK_NAME
    assert (
        controller.client_info["alpha"] is not None
        if stage != "sync-failure"
        else controller.client_info["alpha"] is None
    )
    if stage != "sync-failure":
        assert tasks[1].data[SyncKey.SERVER_FQCN] == "server.job"
        assert tasks[1].data[SyncKey.CLIENT_INTERFACES] == {"alpha": {"": {"work": []}, "extra": {"query": []}}}
    if stage == "success":
        controller.system_panic.assert_not_called()
        run.assert_called_once_with(app, controller.logger)
        server_proxy, client_proxies = app.setup.call_args.args[1:3]
        assert server_proxy.get_child("helper").name == "server.helper"
        assert client_proxies[0].get_child("extra").name == "alpha.extra"
        assert dispatcher.call_args_list[0].kwargs["target_fqcn"] == "server.job"
        assert dispatcher.call_args_list[1].kwargs["target_fqcn"] == "alpha.job"
        remote.assert_called_once()
    else:
        controller.system_panic.assert_called_once()
        assert ("sync" if stage == "sync-failure" else "set up") in controller.system_panic.call_args.args[0]
        run.assert_not_called()
        remote.assert_not_called()
