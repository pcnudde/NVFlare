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
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from nvflare.apis.analytix import AnalyticsData, AnalyticsDataType
from nvflare.apis.dxo import DXO, DataKind
from nvflare.apis.fl_constant import FLContextKey, FLMetaKey
from nvflare.app_common.app_constant import AlgorithmConstants
from nvflare.app_common.executors.client_api_executor import ClientAPIExecutor, ExecutionMode
from nvflare.private.fed.client import client_api_task_runner as subject


@pytest.fixture
def task_env(monkeypatch):
    executor = object.__new__(ClientAPIExecutor)
    executor._execution_mode = ExecutionMode.IN_PROCESS
    context = SimpleNamespace(
        executor=executor,
        task_script_path="train.py",
        submit_model_task_name="submit_model",
        build_task_metadata=Mock(return_value={"task": "train"}),
        memory_gc_rounds=2,
        cuda_empty_cache=False,
        task_script_args="--epochs 1",
    )
    host = Mock()
    host._build_backend_context.return_value = context
    ctx = Mock()
    ctx.get_prop.return_value = "train"
    ctx.get_identity_name.return_value = "site-a"
    ctx.get_job_id.return_value = "job-a"
    ctx.get_workspace.return_value.get_app_custom_dir.return_value = "/custom"
    api = Mock()
    api.system_info.return_value = {}
    api_factory = Mock(return_value=api)
    script = Mock()
    script_factory = Mock(return_value=script)
    adapter = Mock()
    adapter_factory = Mock(return_value=adapter)
    bus_data = {}
    bus = SimpleNamespace(get_data=bus_data.get, put_data=lambda key, value: bus_data.__setitem__(key, value))
    monkeypatch.setattr(subject, "InProcessClientAPI", api_factory)
    monkeypatch.setattr(subject, "TaskScriptRunner", script_factory)
    monkeypatch.setattr(subject, "TaskInProcessAdapter", adapter_factory)
    monkeypatch.setattr(subject, "DataBus", lambda: bus)
    return SimpleNamespace(
        host=host,
        context=context,
        ctx=ctx,
        api=api,
        script=script,
        adapter=adapter,
        api_factory=api_factory,
        script_factory=script_factory,
        adapter_factory=adapter_factory,
        bus_data=bus_data,
    )


def test_initialization_configures_metadata_checkpoints_and_analytics(task_env):
    env = task_env
    log = Mock()
    worker = subject._ClientAPITaskRunner(env.host, on_log=log, checkpoint_paths={"load": "/checkpoint"})
    worker.initialize(env.ctx)
    env.context.build_task_metadata.assert_called_once_with("site-a", "job-a", "train")
    assert env.api.system_info()["task_checkpoint"] == {"load": "/checkpoint"}
    env.api.configure_memory_management.assert_called_once_with(2, False)
    env.script_factory.assert_called_once_with(custom_dir="/custom", script_path="train.py", script_args="--epochs 1")
    handler = env.adapter_factory.call_args.args[1]
    handler({"key": "loss", "value": 0.25, "data_type": AnalyticsDataType.SCALAR})
    assert AnalyticsData.from_dxo(log.call_args.args[0]).tag == "loss"
    assert AnalyticsData.from_dxo(log.call_args.args[0]).value == 0.25
    worker.finalize(env.ctx)


@pytest.mark.parametrize("failure", ["executor", "mode", "extension", "submit"])
def test_initialization_rejects_unsupported_executor_and_task_modes(task_env, failure):
    env = task_env
    if failure == "executor":
        env.context.executor = Mock()
    elif failure == "mode":
        env.context.executor._execution_mode = "external"
    elif failure == "extension":
        env.context.task_script_path = "train.sh"
    else:
        env.ctx.get_prop.return_value = "submit_model"
    worker = subject._ClientAPITaskRunner(env.host)
    with pytest.raises((RuntimeError, ValueError)):
        worker.initialize(env.ctx)
    env.api.init.assert_not_called()
    env.script_factory.assert_not_called()


def test_partial_initialization_closes_api_and_preserves_original_error(task_env):
    env = task_env
    env.api.init.side_effect = ValueError("initialize failed")
    env.api.close.side_effect = OSError("close failed")
    worker = subject._ClientAPITaskRunner(env.host)
    with pytest.raises(ValueError, match="initialize failed"):
        worker.initialize(env.ctx)
    env.api.close.assert_called_once_with()
    assert worker._api is None


@pytest.mark.parametrize("exit_code", [None, 0, 5])
def test_execution_stages_one_assignment_publishes_api_and_handles_script_exit(task_env, exit_code):
    env = task_env
    started = Mock()
    worker = subject._ClientAPITaskRunner(env.host, on_started=started)
    worker.initialize(env.ctx)
    shareable = DXO(DataKind.WEIGHTS, {"weight": [1.0]}).to_shareable()
    staged_metadata = []
    env.adapter.stage_input.side_effect = lambda value: staged_metadata.append(
        (value.get_header(FLMetaKey.JOB_ID), value.get_header(FLMetaKey.SITE_NAME))
    )
    env.script.run.side_effect = SystemExit(exit_code)
    if exit_code == 5:
        with pytest.raises(SystemExit) as raised:
            worker.execute("train", shareable, env.ctx, SimpleNamespace(triggered=False))
        assert raised.value.code == 5
    else:
        assert (
            worker.execute("train", shareable, env.ctx, SimpleNamespace(triggered=False))
            is env.adapter.get_result.return_value
        )
    staged = env.adapter.stage_input.call_args.args[0]
    assert staged is shareable and not shareable
    assert staged_metadata == [("job-a", "site-a")]
    started.assert_called_once_with()
    assert env.bus_data[subject.CLIENT_API_KEY] is env.api
    with pytest.raises(RuntimeError, match="only its one assigned task"):
        worker.execute("train", shareable, env.ctx, SimpleNamespace(triggered=False))
    worker.finalize(env.ctx)
    assert env.bus_data[subject.CLIENT_API_KEY] is None
    env.script.release_runtime.assert_called_once_with()
    env.api.close.assert_called_once_with()


@pytest.mark.parametrize("failure", ["wrong-task", "abort", "scaffold-meta", "scaffold-collection"])
def test_execution_rejects_wrong_aborted_and_unsupported_scaffold_assignments(task_env, failure):
    env = task_env
    worker = subject._ClientAPITaskRunner(env.host)
    worker.initialize(env.ctx)
    dxo = DXO(DataKind.WEIGHTS, {"w": [1]})
    if failure == "scaffold-meta":
        dxo.meta[AlgorithmConstants.SCAFFOLD_CTRL_GLOBAL] = {}
    if failure == "scaffold-collection":
        dxo = DXO(DataKind.COLLECTION, {AlgorithmConstants.SCAFFOLD_CTRL_GLOBAL: DXO(DataKind.WEIGHTS, {"w": [1]})})
    with pytest.raises((RuntimeError, ValueError)):
        worker.execute(
            "other" if failure == "wrong-task" else "train",
            dxo.to_shareable(),
            env.ctx,
            SimpleNamespace(triggered=failure == "abort"),
        )
    env.script.run.assert_not_called()
    env.adapter.stage_input.assert_not_called()
    worker.finalize(env.ctx)


@pytest.mark.parametrize("runner_fails,api_fails", [(True, True), (False, True), (False, False)])
def test_finalize_attempts_both_cleanups_and_calls_finished_only_after_success(task_env, runner_fails, api_fails):
    env = task_env
    finished = Mock()
    worker = subject._ClientAPITaskRunner(env.host, on_finished=finished)
    worker.initialize(env.ctx)
    if runner_fails:
        env.script.release_runtime.side_effect = RuntimeError("script cleanup")
    if api_fails:
        env.api.close.side_effect = ValueError("api cleanup")
    if runner_fails or api_fails:
        with pytest.raises(RuntimeError if runner_fails else ValueError, match="cleanup"):
            worker.finalize(env.ctx)
        finished.assert_not_called()
    else:
        worker.finalize(env.ctx)
        worker.finalize(env.ctx)
        finished.assert_called_once_with()
    env.script.release_runtime.assert_called_once_with()
    env.api.close.assert_called_once_with()
    assert worker._api is worker._runner is None


@pytest.mark.parametrize(
    "record,message",
    [
        ({"identity": {"id": "wrong"}, "monotonic": 1}, "identity"),
        ({"identity": {"id": "a"}, "monotonic": True}, "timestamp"),
        ({"identity": {"id": "a"}, "monotonic": -1}, "timestamp"),
        ({"identity": {"id": "a"}, "monotonic": float("inf")}, "timestamp"),
        ({"identity": {"id": "a"}, "monotonic": "1"}, "timestamp"),
    ],
)
def test_progress_markers_reject_untrusted_identity_and_timestamp(tmp_path, record, message):
    identity = SimpleNamespace(to_dict=lambda: {"id": "a"})
    store = SimpleNamespace(attempt_dir=lambda identity: str(tmp_path))
    (tmp_path / "progress.json").write_text(json.dumps(record))
    with pytest.raises(ValueError, match=message):
        subject._read_progress(store, identity, "progress.json")


def test_progress_markers_limit_size_and_missing_marker_is_absent(tmp_path):
    identity = SimpleNamespace(to_dict=lambda: {"id": "a"})
    store = SimpleNamespace(attempt_dir=lambda identity: str(tmp_path))
    assert subject._read_progress(store, identity, "missing.json") is None
    (tmp_path / "large.json").write_bytes(b"x" * 65537)
    with pytest.raises(ValueError, match="too large"):
        subject._read_progress(store, identity, "large.json")


def test_context_binding_wraps_result_and_analytics_callbacks_in_engine_operation(task_env, monkeypatch, tmp_path):
    env = task_env
    identity = SimpleNamespace(site_name="site-a", to_dict=lambda: {"id": "a"})
    store = Mock()
    store.attempt_dir.return_value = str(tmp_path)
    store.checkpoint_paths.return_value = {"load": "/load"}
    store.stage_result.return_value = "reference"
    env.ctx.get_engine.return_value.operation.side_effect = lambda: nullcontext()
    log = Mock()
    factory = Mock()
    monkeypatch.setattr(subject, "_ClientAPITaskRunner", factory)
    adapt = subject.bind_client_task_context(env.ctx, identity, store, log)
    env.ctx.set_prop.assert_any_call(FLContextKey.CLIENT_NAME, "site-a", private=True, sticky=True)
    env.ctx.set_prop.assert_any_call(FLContextKey.PROCESS_TYPE, "client_task_worker", private=True, sticky=True)
    other = Mock()
    assert adapt(other) is other
    executor = object.__new__(ClientAPIExecutor)
    assert adapt(executor) is factory.return_value
    kwargs = factory.call_args.kwargs
    kwargs["on_started"]()
    kwargs["on_finished"]()
    result = DXO(DataKind.WEIGHTS, {"w": [1]}).to_shareable()
    assert kwargs["on_result"](result) == "reference"
    store.stage_result.assert_called_once_with(identity, result)
    analytics = Mock()
    kwargs["on_log"](analytics)
    log.assert_called_once_with(analytics.to_shareable.return_value)
    assert env.ctx.get_engine.return_value.operation.call_count == 2
    for name in (
        "result_wait_started.json",
        "script_finalized.json",
        "result_staging_started.json",
        "script_result_sent.json",
    ):
        assert subject._read_progress(store, identity, name) >= 0
    no_log = subject.bind_client_task_context(env.ctx, identity, store, None)
    with pytest.raises(RuntimeError, match="cell session"):
        no_log(executor)
