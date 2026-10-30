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

from unittest.mock import MagicMock

import numpy as np
import pytest
import xgboost as xgb

from nvflare.apis.fl_constant import FLContextKey, ReservedKey
from nvflare.apis.fl_context import FLContext
from nvflare.app_common.tracking.log_writer import LogWriter
from nvflare.app_opt.xgboost.data_loader import XGBDataLoader
from nvflare.app_opt.xgboost.histogram_based_v2.defs import Constant
from nvflare.app_opt.xgboost.histogram_based_v2.runners import xgb_client_runner as train_module
from nvflare.app_opt.xgboost.histogram_based_v2.runners import xgb_eval_runner as eval_module


@pytest.fixture
def context(tmp_path):
    ctx = FLContext()
    engine = MagicMock()
    ctx.set_prop(ReservedKey.ENGINE, engine)
    workspace = MagicMock()
    workspace.get_app_custom_dir.return_value = str(tmp_path)
    ctx.set_prop(FLContextKey.WORKSPACE_OBJECT, workspace)
    ctx.set_prop(FLContextKey.CURRENT_JOB_ID, "job")
    return ctx


@pytest.mark.parametrize("secure,custom_plugin", [(False, False), (True, False), (True, True)])
def test_client_runner_marshals_secure_plugin_training_options_and_model_output(
    monkeypatch, tmp_path, context, secure, custom_plugin
):
    runner = train_module.XGBClientRunner("loader", "saved.json", metrics_writer_id="metrics")
    loader = MagicMock(spec=XGBDataLoader)
    loader.load_data.return_value = (object(), object())
    writer = MagicMock(spec=LogWriter)
    context.get_engine().get_component.side_effect = lambda name: {"loader": loader, "metrics": writer}[name]
    runner.initialize(context)
    communicator = MagicMock()
    monkeypatch.setattr(train_module.xgb.collective, "CommunicatorContext", communicator)
    monkeypatch.setattr(train_module.xgb.collective, "communicator_print", MagicMock())
    train = MagicMock()
    monkeypatch.setattr(train_module.xgb, "train", train)
    config = {"xgb_plugin_name": "custom", "xgb_plugin_path": "/custom/plugin.so"} if custom_plugin else {}
    monkeypatch.setattr(train_module.ConfigService, "get_str_var", lambda name, **kw: config.get(name))
    monkeypatch.setattr(train_module.ConfigService, "get_dict_var", lambda **kw: {})
    options = {"use_gpus": True, "enable_explainability": False, "early_stopping_rounds": 2, "verbose_eval": False}
    params = {"objective": "reg:squarederror"}
    task = {
        Constant.RUNNER_CTX_CLIENT_NAME: "site",
        Constant.RUNNER_CTX_RANK: 0,
        Constant.RUNNER_CTX_WORLD_SIZE: 2,
        Constant.RUNNER_CTX_NUM_ROUNDS: 3,
        Constant.RUNNER_CTX_XGB_PARAMS: params,
        Constant.RUNNER_CTX_SERVER_ADDR: "localhost:1234",
        Constant.RUNNER_CTX_MODEL_DIR: str(tmp_path),
        Constant.RUNNER_CTX_XGB_OPTIONS: options,
        Constant.RUNNER_CTX_SECURE_TRAINING: secure,
    }
    assert runner.is_stopped() == (False, 0)
    runner.run(task)
    assert runner.is_stopped() == (True, 0)
    assert params["device"] == "cuda"
    kwargs = communicator.call_args.kwargs
    assert kwargs["federated_rank"] == 0 and kwargs["federated_world_size"] == 2
    if secure:
        assert kwargs[train_module.PLUGIN_PARAM_KEY]["name"] == ("custom" if custom_plugin else "cuda_paillier")
        assert kwargs[train_module.PLUGIN_PARAM_KEY]["path"].endswith(
            "plugin.so" if custom_plugin else "libcuda_paillier.so"
        )
    else:
        assert train_module.PLUGIN_PARAM_KEY not in kwargs
    assert train.call_args.args[2] == 3
    assert len(train.call_args.kwargs["callbacks"]) == 2
    assert train.call_args.kwargs["early_stopping_rounds"] == 2
    assert train.call_args.kwargs["xgb_model"] is None
    train.return_value.save_model.assert_called_once_with(str(tmp_path / "saved.json"))
    (tmp_path / "model.json").write_text("pretrained")
    runner._xgb_train(2, params, {}, object(), object())
    assert train.call_args.kwargs["xgb_model"] == str(tmp_path / "model.json")


@pytest.mark.parametrize(
    "key",
    [
        Constant.RUNNER_CTX_CLIENT_NAME,
        Constant.RUNNER_CTX_RANK,
        Constant.RUNNER_CTX_WORLD_SIZE,
        Constant.RUNNER_CTX_SERVER_ADDR,
    ],
)
def test_evaluation_context_requires_client_rank_world_size_and_server(key):
    task = {
        Constant.RUNNER_CTX_CLIENT_NAME: "site",
        Constant.RUNNER_CTX_RANK: 0,
        Constant.RUNNER_CTX_WORLD_SIZE: 1,
        Constant.RUNNER_CTX_SERVER_ADDR: "localhost:1234",
    }
    del task[key]
    with pytest.raises(RuntimeError, match="Missing"):
        eval_module._check_ctx(task)
    with pytest.raises(RuntimeError, match="Missing"):
        train_module._check_ctx({})


def test_eval_runner_loads_real_small_booster_and_reports_auc_only_on_label_owner(monkeypatch, tmp_path, context):
    data = xgb.DMatrix(np.arange(10, dtype=np.float32).reshape(-1, 1), label=np.array([0] * 5 + [1] * 5))
    booster = xgb.train(
        {"objective": "binary:logistic", "max_depth": 2, "min_child_weight": 0, "eta": 1, "nthread": 1},
        data,
        num_boost_round=3,
    )
    directory = tmp_path / "site/simulate_job"
    directory.mkdir(parents=True)
    booster.save_model(directory / "model.json")
    runner = eval_module.XGBEvalRunner("loader", str(tmp_path))
    loader = MagicMock(spec=XGBDataLoader)
    loader.load_data.return_value = (data, data)
    context.get_engine().get_component.return_value = loader
    runner.initialize(context)
    runner._client_name = "site"
    loaded = runner._load_trained_model()
    runner._rank = 0
    assert runner._evaluate_model(loaded, data) == pytest.approx(1.0)
    runner._rank = 1
    assert runner._evaluate_model(loaded, data) == 0.0
    communicator = MagicMock()
    printer = MagicMock()
    monkeypatch.setattr(eval_module.xgb.collective, "CommunicatorContext", communicator)
    monkeypatch.setattr(eval_module.xgb.collective, "communicator_print", printer)
    runner.run(
        {
            Constant.RUNNER_CTX_CLIENT_NAME: "site",
            Constant.RUNNER_CTX_RANK: 0,
            Constant.RUNNER_CTX_WORLD_SIZE: 1,
            Constant.RUNNER_CTX_SERVER_ADDR: "localhost:1234",
        }
    )
    assert runner.is_stopped() == (True, 0)
    assert communicator.call_args.kwargs["federated_server_address"] == "localhost:1234"
    loader.initialize.assert_called_once_with(client_id="site", rank=0, data_split_mode=0)
    printer.assert_called_once_with("Finished evaluation\n")
    runner._client_name = "missing"
    with pytest.raises(FileNotFoundError, match="Trained model not found"):
        runner._load_trained_model()
