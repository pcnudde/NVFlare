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

import threading
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from nvflare.apis.fl_constant import FLContextKey
from nvflare.apis.fl_context import FLContext
from nvflare.private.fed.client import client_executor as module


@pytest.fixture
def env(monkeypatch, tmp_path):
    monkeypatch.setattr(
        module,
        "ConfigService",
        SimpleNamespace(
            get_float_var=lambda **kwargs: kwargs["default"], get_int_var=lambda **kwargs: kwargs["default"]
        ),
    )
    client = Mock()
    client.client_name = "site"
    client.cell.get_internal_listener_params.return_value = {module.ConnPropKey.CONNECTION_SECURITY: "secure"}
    executor = module.JobExecutor(client, "startup")
    executor.logger = Mock()
    context = FLContext()
    workspace = Mock()
    path = tmp_path / "meta.json"
    path.write_text("{}")
    workspace.get_job_meta_path.return_value = str(path)
    context.set_prop(FLContextKey.WORKSPACE_OBJECT, workspace)
    context.set_prop(FLContextKey.SERVER_CONFIG, [{"service": {"target": "server:8002"}}])
    monkeypatch.setattr(module, "Workspace", Mock(return_value=workspace))
    monkeypatch.setattr(module, "get_job_launcher", Mock(return_value=Mock()))
    monkeypatch.setattr(module, "has_cj_resource_spec", Mock(return_value=False))
    worker = Mock()
    monkeypatch.setattr(module, "threading", SimpleNamespace(Lock=threading.Lock, Thread=Mock(return_value=worker)))
    monkeypatch.setattr(executor, "_launch_job", Mock())
    return SimpleNamespace(
        executor=executor,
        client=client,
        ctx=context,
        args=SimpleNamespace(workspace=str(tmp_path), set=["custom=value"]),
        meta_file=path,
        worker=worker,
    )


@pytest.mark.parametrize("failure", ["missing-meta", "missing-server", "invalid-server", "success"])
def test_launch_checks_deployed_metadata_and_server_configuration_before_fake_worker_start(env, failure):
    if failure == "missing-meta":
        env.meta_file.unlink()
    elif failure == "missing-server":
        env.ctx.remove_prop(FLContextKey.SERVER_CONFIG, force_removal=True)
    elif failure == "invalid-server":
        env.ctx.set_prop(FLContextKey.SERVER_CONFIG, [{"service": "invalid"}])
    if failure == "success":
        env.executor.start_app(env.client, "job", {}, env.args, {}, "token", Mock(), env.ctx)
        restart = env.executor._launch_job.call_args.args[1]
        assert restart.args[module.JobProcessArgs.PARENT_CONN_SEC] == ("--parent_conn_sec", "secure")
        assert "custom=value" in restart.args[module.JobProcessArgs.OPTIONS][1]
        assert env.args.set == ["custom=value"]
        env.worker.start.assert_called_once_with()
    else:
        with pytest.raises(RuntimeError, match="missing deployed|missing.*server_config|expect server config"):
            env.executor.start_app(env.client, "job", {}, env.args, {}, "token", Mock(), env.ctx)
        env.executor._launch_job.assert_not_called()
        env.worker.start.assert_not_called()


def test_launcher_lookup_requires_event_selected_list_and_preserves_job_metadata(env):
    engine = Mock()
    env.ctx.set_prop(module.ReservedKey.ENGINE, engine)
    meta = {"name": "training"}
    with pytest.raises(RuntimeError, match="no job launcher"):
        env.executor._get_job_launcher(meta, env.ctx)
    launcher = Mock()
    env.ctx.set_prop(FLContextKey.JOB_LAUNCHER, [launcher])
    assert env.executor._get_job_launcher(meta, env.ctx) is launcher
    assert env.ctx.get_prop(FLContextKey.JOB_META) == meta
    assert engine.fire_event.call_count == 2


def test_pending_handle_rejects_replacement_and_termination_wait_has_finite_deadline(env, monkeypatch):
    pending = module._PendingJobHandle()
    original = Mock()
    pending.attach(original)
    with pytest.raises(RuntimeError, match="different job handle"):
        pending.attach(Mock())
    env.executor.run_processes["job"] = {}
    handle = Mock()
    monkeypatch.setattr(module, "time", SimpleNamespace(time=Mock(side_effect=[0, 11]), sleep=Mock(side_effect=[])))
    env.executor.run_processes["job"] = {"active": True}
    env.executor._terminate_job(handle, "job")
    handle.terminate.assert_called_once_with()
    assert "terminated" in env.executor.logger.info.call_args.args[0]


def test_negative_restart_budget_is_rejected_before_any_worker_is_created(monkeypatch):
    monkeypatch.setattr(
        module, "ConfigService", SimpleNamespace(get_float_var=lambda **kwargs: 5, get_int_var=lambda **kwargs: -1)
    )
    with pytest.raises(ValueError, match="nonnegative"):
        module.JobExecutor(Mock(), "startup")
