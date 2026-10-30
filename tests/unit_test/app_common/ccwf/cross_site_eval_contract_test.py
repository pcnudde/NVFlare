# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from nvflare.apis.dxo import DXO, DataKind, from_shareable
from nvflare.apis.executor import Executor
from nvflare.apis.fl_constant import FLContextKey, ReservedKey
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import ReturnCode, Shareable, make_reply
from nvflare.apis.signal import Signal
from nvflare.app_common.abstract.model import make_model_learnable
from nvflare.app_common.abstract.model_persistor import ModelPersistor
from nvflare.app_common.app_constant import AppConstants, ModelName, ValidateType
from nvflare.app_common.app_event_type import AppEventType
from nvflare.app_common.ccwf import cse_client_ctl as client_module
from nvflare.app_common.ccwf import cse_server_ctl as server_module
from nvflare.app_common.ccwf.common import Constant, ModelType


@pytest.fixture
def client_runtime():
    ctl = client_module.CrossSiteEvalClientController()
    ctl.me = "site-1"
    ctl.persistor = Mock(spec=ModelPersistor)
    ctl.validate_executor = Mock(spec=Executor)
    ctl.submit_model_executor = Mock(spec=Executor)
    ctl.config = {Constant.PRIVATE_P2P: False}
    ctx = FLContext()
    ctx.set_prop(ReservedKey.IDENTITY_NAME, "site-1", private=True, sticky=True)
    peer = FLContext()
    peer.set_prop(ReservedKey.IDENTITY_NAME, "site-2", private=True, sticky=True)
    ctx.set_peer_context(peer)
    return ctl, ctx


def eval_request(owner="site-2", model_type=ModelType.LOCAL):
    return Shareable(
        {
            Constant.MODEL_TYPE: model_type,
            Constant.MODEL_OWNER: owner,
            Constant.MODEL_NAME: "best",
            AppConstants.CURRENT_ROUND: 3,
        }
    )


@pytest.mark.parametrize("owner", ["site-1", "site-2"])
@pytest.mark.parametrize("kind", [ModelType.LOCAL, ModelType.GLOBAL])
def test_client_evaluation_preserves_model_identity_and_validation_headers(client_runtime, owner, kind):
    ctl, ctx = client_runtime
    model = DXO(DataKind.WEIGHTS, {"w": [1, 2]}).to_shareable()
    ctl._set_prepared_model(kind, "best", model)
    ctl.broadcast_and_wait = Mock(return_value={owner: model})
    metrics = DXO(DataKind.METRICS, {"accuracy": 0.8}).to_shareable()
    ctl.validate_executor.execute.return_value = metrics
    reply = ctl._do_eval(eval_request(owner, kind), ctx, Signal())
    assert reply is metrics
    assert reply.get_header(Constant.MODEL_OWNER) == owner
    assert reply.get_header(Constant.MODEL_TYPE) == kind
    assert reply.get_header(Constant.MODEL_NAME) == "best"
    assert reply.get_header(AppConstants.CURRENT_ROUND) == 3
    submitted = ctl.validate_executor.execute.call_args.kwargs["shareable"]
    assert submitted.get_header(AppConstants.VALIDATE_TYPE) == ValidateType.MODEL_VALIDATE
    assert submitted.get_header(FLContextKey.TASK_NAME) == ctl.validation_task_name
    assert from_shareable(submitted).data == {"w": [1, 2]}
    if owner == "site-1":
        assert submitted is not model
        assert model.get_header(AppConstants.VALIDATE_TYPE) is None
        ctl.broadcast_and_wait.assert_not_called()
    else:
        task = ctl.broadcast_and_wait.call_args.kwargs["task"]
        assert task.name == ctl.ask_for_model_task_name
        assert task.data[Constant.MODEL_TYPE] == kind
        assert ctl.broadcast_and_wait.call_args.kwargs["targets"] == [owner]


@pytest.mark.parametrize(
    "failure", ["validator", "local-model", "missing-reply", "wrong-reply", "error-reply", "abort"]
)
def test_client_evaluation_rejects_unavailable_models_and_cancellation(client_runtime, failure):
    ctl, ctx = client_runtime
    owner = "site-1" if failure == "local-model" else "site-2"
    response = DXO(DataKind.WEIGHTS, {"w": [1]}).to_shareable()
    if failure == "validator":
        ctl.validate_executor = None
    elif failure == "missing-reply":
        response = None
    elif failure == "wrong-reply":
        response = "not-shareable"
    elif failure == "error-reply":
        response = make_reply(ReturnCode.BAD_REQUEST_DATA)
    ctl.broadcast_and_wait = Mock(return_value={owner: response})
    signal = Signal()
    if failure == "abort":
        signal.trigger(True)
    reply = ctl._do_eval(eval_request(owner), ctx, signal)
    expected = {
        "validator": Constant.RC_UNABLE_TO_EVAL,
        "error-reply": ReturnCode.BAD_REQUEST_DATA,
        "abort": ReturnCode.TASK_ABORTED,
    }.get(failure, ReturnCode.EXECUTION_EXCEPTION)
    assert reply.get_return_code() == expected
    if ctl.validate_executor:
        ctl.validate_executor.execute.assert_not_called()


@pytest.mark.parametrize(
    "scenario,expected",
    [("no-local", Constant.RC_NO_LOCAL_MODEL), ("no-validator", Constant.RC_UNABLE_TO_EVAL), ("global", ReturnCode.OK)],
)
def test_configuration_checks_assigned_evaluation_roles_and_advertises_inventory(client_runtime, scenario, expected):
    ctl, ctx = client_runtime
    ctl.config.update(
        {
            Constant.EVAL_LOCAL: True,
            Constant.EVAL_GLOBAL: True,
            Constant.EVALUATEES: ["site-1"],
            Constant.EVALUATORS: ["site-1"],
            Constant.GLOBAL_CLIENT: "site-1",
        }
    )
    if scenario == "no-local":
        ctl.submit_model_executor = None
    if scenario == "no-validator":
        ctl.validate_executor = None
    ctl.persistor.get_model_inventory.return_value = {"best": "location", "last": "location2"}
    reply = ctl.process_config(ctx)
    assert reply.get_return_code() == expected
    if scenario == "global":
        assert reply[Constant.GLOBAL_NAMES] == ["best", "last"]


def test_prepare_global_model_invalidates_old_cache_and_serves_copies(client_runtime):
    ctl, ctx = client_runtime
    assert ctl._prepare_global_model("best", ctx).get_return_code() == ReturnCode.BAD_REQUEST_DATA
    ctl.inventory = {"best": "location"}
    ctl.persistor.get.return_value = make_model_learnable({"w": [7]}, {"round": 3})
    ctl._set_prepared_model(ModelType.LOCAL, "old", Shareable({"w": 0}))
    request = Shareable({Constant.MODEL_TYPE: ModelType.GLOBAL, Constant.MODEL_NAME: "best"})
    assert ctl._prepare_model(request, ctx, Signal()).get_return_code() == ReturnCode.OK
    assert ctl._get_prepared_model(ModelType.LOCAL, "old") is None
    reply = ctl._process_get_model_request(request, ctx)
    assert from_shareable(reply).data == {"w": [7]}
    assert reply is not ctl._get_prepared_model(ModelType.GLOBAL, "best")
    request[Constant.MODEL_NAME] = "absent"
    assert ctl._process_get_model_request(request, ctx).get_return_code() == ReturnCode.BAD_REQUEST_DATA


@pytest.mark.parametrize("missing", [None, "submit", "validate"])
def test_start_run_resolves_role_executors_and_reports_missing_executor(client_runtime, monkeypatch, missing):
    ctl, ctx = client_runtime
    monkeypatch.setattr(client_module.ClientSideController, "start_run", Mock())
    ctl.system_panic = Mock()
    runner = Mock()
    runner.find_executor.side_effect = lambda task: (
        None
        if (missing == "submit" and task == ctl.submit_model_task_name)
        or (missing == "validate" and task == ctl.validation_task_name)
        else Mock(spec=Executor)
    )
    ctx.set_prop(FLContextKey.RUNNER, runner, private=True, sticky=True)
    ctl.start_run(ctx)
    assert ctl.system_panic.called is (missing is not None)
    if missing is None:
        assert ctl.submit_model_executor is not None and ctl.validate_executor is not None


@pytest.fixture
def server_runtime():
    ctl = server_module.CrossSiteEvalServerController(
        participating_clients=["site-1", "site-2"], max_parallel_actions=2
    )
    ctl.evaluators = ["site-1", "site-2"]
    ctl.evaluatees = ["site-1", "site-2"]
    ctl.system_panic = Mock()
    ctl.fire_event = Mock()
    ctl.eval_manager = Mock()
    return ctl, FLContext()


def test_global_inventory_keeps_first_owner_and_config_exposes_assignments(server_runtime):
    ctl, ctx = server_runtime
    assert ctl.process_config_reply("site-1", Shareable({Constant.GLOBAL_NAMES: ["best", "last"]}), ctx)
    ctl.process_config_reply("site-2", Shareable({Constant.GLOBAL_NAMES: ["best", "extra"]}), ctx)
    assert ctl.global_names == {"best": "site-1", "last": "site-1", "extra": "site-2"}
    config = ctl.prepare_config()
    assert config[Constant.EVALUATORS] == ["site-1", "site-2"]
    assert config[Constant.EVALUATEES] == ["site-1", "site-2"]


@pytest.mark.parametrize("ready", [False, True])
def test_server_preparation_requires_every_owner_to_acknowledge(server_runtime, ready):
    ctl, ctx = server_runtime

    def broadcast(**kwargs):
        task = kwargs["task"]
        assert task.data[Constant.MODEL_TYPE] == ModelType.GLOBAL
        assert task.data[Constant.MODEL_NAME] == "best"
        assert kwargs["min_responses"] == 2
        for name in kwargs["targets"]:
            task.result_received_cb(
                SimpleNamespace(
                    task=task,
                    client=SimpleNamespace(name=name),
                    result=make_reply(ReturnCode.OK if ready else ReturnCode.ERROR),
                ),
                ctx,
            )

    ctl.broadcast_and_wait = Mock(side_effect=broadcast)
    assert ctl._ask_to_prepare_model(ModelType.GLOBAL, "best", ["site-1", "site-2"], Signal(), ctx) is ready


@pytest.mark.parametrize("aborted", [False, True])
def test_eval_actions_encode_owner_round_and_cancel_pending_tasks_on_abort(server_runtime, aborted):
    ctl, ctx = server_runtime
    ctl.broadcast = Mock()
    ctl.get_num_standing_tasks = Mock(return_value=1 if aborted else 0)
    ctl.cancel_task = Mock()
    signal = Signal()
    if aborted:
        signal.trigger(True)
    ctl._ask_to_eval([("site-1", "site-2"), ("site-2", "site-1")], ModelType.LOCAL, "best", signal, ctx)
    tasks = [c.kwargs["task"] for c in ctl.broadcast.call_args_list]
    assert [t.data[Constant.MODEL_OWNER] for t in tasks] == ["site-2", "site-1"]
    assert all(t.data[AppConstants.CURRENT_ROUND] == 1 and t.data[Constant.MODEL_NAME] == "best" for t in tasks)
    assert ctl.cancel_task.call_count == (2 if aborted else 0)


@pytest.mark.parametrize("model_type", [ModelType.LOCAL, ModelType.GLOBAL])
@pytest.mark.parametrize("success", [False, True])
def test_received_metrics_fire_validation_event_and_only_persist_success(server_runtime, model_type, success):
    ctl, ctx = server_runtime
    result = DXO(DataKind.METRICS, {"accuracy": 0.9}).to_shareable()
    result.set_header(Constant.MODEL_TYPE, model_type)
    result.set_header(Constant.MODEL_NAME, "best")
    result.set_header(Constant.MODEL_OWNER, "site-2")
    result.set_return_code(ReturnCode.OK if success else ReturnCode.ERROR)
    ctl._process_eval_result(SimpleNamespace(client=SimpleNamespace(name="site-1"), result=result), ctx)
    assert ctx.get_prop(AppConstants.MODEL_OWNER) == ("GLOBAL_best" if model_type == ModelType.GLOBAL else "site-2")
    assert ctx.get_prop(AppConstants.DATA_CLIENT) == "site-1"
    assert ctx.get_prop(AppConstants.VALIDATION_RESULT) is result
    assert sum(c.args == (AppEventType.VALIDATION_RESULT_RECEIVED, ctx) for c in ctl.fire_event.call_args_list) == 1
    assert ctl.eval_manager.add_result.called is success
    if success:
        assert ctl.eval_manager.add_result.call_args.kwargs["result"].data == {"accuracy": 0.9}


@pytest.mark.parametrize("prepared", [False, True])
def test_local_evaluation_filters_clients_without_training_models(server_runtime, prepared):
    ctl, ctx = server_runtime
    ctx.set_prop(Constant.PROP_KEY_TRAIN_CLIENTS, ["site-2"], private=True, sticky=True)
    ctl._ask_to_prepare_model = Mock(return_value=prepared)
    ctl._do_eval_actions = Mock()
    signal = Signal()
    ctl._evaluate_local_models(signal, ctx)
    ctl._ask_to_prepare_model.assert_called_once_with(ModelType.LOCAL, ModelName.BEST_MODEL, ["site-2"], signal, ctx)
    assert ctl._do_eval_actions.called is prepared


def test_global_evaluation_requires_enabled_inventory_and_successful_preparation(server_runtime):
    ctl, ctx = server_runtime
    ctl.eval_global = True
    ctl._ask_to_prepare_model = Mock(return_value=True)
    ctl._ask_to_eval = Mock()
    signal = Signal()
    ctl._evaluate_global_models(signal, ctx)
    ctl._ask_to_prepare_model.assert_not_called()
    ctl.global_names = {"best": "site-2"}
    ctl._evaluate_global_models(signal, ctx)
    assert ctl._ask_to_eval.call_args.args[:3] == (
        [("site-1", "site-2"), ("site-2", "site-2")],
        ModelType.GLOBAL,
        "best",
    )
    ctl.get_num_standing_tasks = Mock(return_value=0)
    assert ctl.is_sub_flow_done(ctx)
    ctl.global_names.clear()
    ctl.evaluatees.clear()
    ctl.sub_flow(signal, ctx)
    ctl.system_panic.assert_called_once()


def test_server_start_resolves_roles_and_creates_result_directory(tmp_path):
    ctl = server_module.CrossSiteEvalServerController(participating_clients=["site-1", "site-2"])
    engine = Mock()
    engine.get_clients.return_value = [SimpleNamespace(name="site-1"), SimpleNamespace(name="site-2")]
    engine.get_workspace.return_value.get_run_dir.return_value = str(tmp_path)
    ctl._engine = engine
    ctx = FLContext()
    ctx.set_prop(ReservedKey.ENGINE, engine, private=True, sticky=True)
    ctx.set_prop(FLContextKey.WORKFLOW, "wf", private=True, sticky=True)
    ctl.start_controller(ctx)
    assert ctl.eval_local and ctl.eval_global
    assert ctl.evaluators == ["site-1", "site-2"]
    assert ctl.global_model_client == "site-1"
    assert ctl.eval_manager.get_results() == {}
    assert Path(ctl.eval_manager.result_dir).is_dir()


def test_server_start_rejects_configuration_without_any_models(server_runtime, monkeypatch):
    ctl, ctx = server_runtime
    monkeypatch.setattr(server_module.ServerSideController, "start_controller", Mock())
    monkeypatch.setattr(server_module, "validate_candidates", Mock(side_effect=[["site-1"], []]))
    monkeypatch.setattr(server_module, "validate_candidate", Mock(return_value=""))
    with pytest.raises(RuntimeError, match="nothing to evaluate"):
        ctl.start_controller(ctx)


def test_local_evaluation_skips_when_no_evaluatee_trained(server_runtime):
    ctl, ctx = server_runtime
    ctx.set_prop(Constant.PROP_KEY_TRAIN_CLIENTS, ["another-site"], private=True, sticky=True)
    ctl._ask_to_prepare_model = Mock()
    ctl._evaluate_local_models(Signal(), ctx)
    ctl._ask_to_prepare_model.assert_not_called()
