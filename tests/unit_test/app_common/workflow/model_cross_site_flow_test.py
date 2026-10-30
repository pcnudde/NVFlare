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

import json
from unittest.mock import Mock

from nvflare.apis.fl_context import FLContext
from nvflare.apis.signal import Signal
from nvflare.app_common.abstract.fl_model import FLModel
from nvflare.app_common.abstract.model import make_model_learnable
from nvflare.app_common.abstract.model_persistor import ModelPersistor
from nvflare.app_common.app_constant import AppConstants
from nvflare.app_common.workflows import cross_site_eval as subject
from nvflare.fuel.utils import fobs
from tests.timing_utils import isolate_time


def configured_controller(tmp_path, monkeypatch, server_models=None):
    controller = subject.CrossSiteEval(server_models=server_models or [], participating_clients=["one", "two"])
    controller.fl_ctx = FLContext()
    controller.abort_signal = Signal()
    controller.get_run_dir = Mock(return_value=str(tmp_path))
    controller._cross_val_models_dir = str(tmp_path / "models")
    controller._cross_val_results_dir = str(tmp_path / "results")
    (tmp_path / "models").mkdir()
    (tmp_path / "results").mkdir()
    controller._client_models = {"one": None, "two": None}
    controller._val_results = {"one": {}, "two": {}}
    return controller


def test_cross_site_model_exchange_saves_local_models_validation_results_and_summary(tmp_path, monkeypatch):
    controller = configured_controller(tmp_path, monkeypatch, ["global"])
    controller.persistor = Mock(spec=ModelPersistor)
    controller.persistor.get_model.return_value = make_model_learnable({"w": [10]}, {})
    sent = []

    def dispatch(**kwargs):
        sent.append(kwargs)
        if kwargs["task_name"] == AppConstants.TASK_SUBMIT_MODEL:
            kwargs["callback"](FLModel(params={"w": [3]}, meta={"client_name": "one"}))
        else:
            owner = kwargs["data"].meta[AppConstants.MODEL_OWNER]
            kwargs["callback"](
                FLModel(
                    metrics={"accuracy": 0.75}, meta={"client_name": "two", "props": {AppConstants.MODEL_OWNER: owner}}
                )
            )

    controller.send_model = Mock(side_effect=dispatch)
    controller.get_num_standing_tasks = Mock(side_effect=[1, 0])
    clock = isolate_time(monkeypatch, subject)
    clock.sleep = Mock()
    controller.run()
    assert [call["task_name"] for call in sent] == [
        AppConstants.TASK_SUBMIT_MODEL,
        AppConstants.TASK_VALIDATION,
        AppConstants.TASK_VALIDATION,
    ]
    assert all(call["targets"] == ["one", "two"] for call in sent)
    assert fobs.loadf(controller._client_models["one"]).params == {"w": [3]}
    assert set(controller._val_results["two"]) == {"one", "global"}
    assert fobs.loadf(controller._val_results["two"]["global"]).metrics == {"accuracy": 0.75}
    summary = json.loads((tmp_path / AppConstants.CROSS_VAL_DIR / "cross_val_results.json").read_text())
    assert summary == {"two": {"one": {"accuracy": 0.75}, "global": {"accuracy": 0.75}}}
    clock.sleep.assert_called_once_with(controller._task_check_period)


def test_cross_site_file_server_model_and_abort_stop_wait_before_summary(tmp_path, monkeypatch):
    path = tmp_path / "global.model"
    fobs.dumpf(FLModel(params={"w": [2]}), str(path))
    controller = configured_controller(tmp_path, monkeypatch, [str(path)])
    controller.persistor = object()
    controller.send_model = Mock()
    controller.get_num_standing_tasks = Mock(return_value=1)
    controller.abort_signal.trigger(True)
    controller.save_results = Mock()
    controller.run()
    assert controller.persistor is None
    assert controller.send_model.call_count == 2
    assert controller.send_model.call_args.kwargs["data"].params == {"w": [2]}
    controller.save_results.assert_not_called()


def test_tracking_reports_missing_fields_and_missing_validation_result(tmp_path, monkeypatch):
    controller = configured_controller(tmp_path, monkeypatch)
    controller.error = Mock()
    controller.track_results(None, None, None)
    assert controller.error.call_count == 3
    assert controller._json_val_results == {}
