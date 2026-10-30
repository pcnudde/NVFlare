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

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from nvflare.apis.dxo import DXO, DataKind
from nvflare.apis.fl_constant import ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.apis.signal import Signal
from nvflare.app_common.app_constant import AppConstants
from nvflare.app_common.workflows import cross_site_model_eval as subject
from tests.timing_utils import isolate_time


@pytest.mark.parametrize("component", ["locator", "formatter"])
def test_invalid_cross_validation_component_panics_before_creating_tasks(tmp_path, monkeypatch, component):
    controller = subject.CrossSiteModelEval(
        **{"model_locator_id" if component == "locator" else "formatter_id": "invalid"}
    )
    controller._engine = Mock()
    controller._engine.get_clients.return_value = [SimpleNamespace(name="site")]
    controller._engine.get_workspace.return_value.get_run_dir.return_value = str(tmp_path)
    controller._engine.get_component.return_value = object()
    monkeypatch.setattr(controller, "fire_event", Mock())
    panic = Mock()
    monkeypatch.setattr(controller, "system_panic", panic)
    controller.start_controller(FLContext())
    assert panic.call_count == 1
    assert component in panic.call_args.kwargs["reason"]
    assert controller._client_models == {}


@pytest.mark.parametrize("kind", ["local", "validation"])
def test_malformed_or_mixed_collection_result_cannot_save_invalid_leaf(tmp_path, monkeypatch, kind):
    controller, context = subject.CrossSiteModelEval(), FLContext()
    monkeypatch.setattr(controller, "fire_event", Mock())
    error, exception = Mock(), Mock()
    monkeypatch.setattr(controller, "log_error", error)
    monkeypatch.setattr(controller, "log_exception", exception)
    save = Mock()
    monkeypatch.setattr(controller, "_save_client_model" if kind == "local" else "_save_validation_result", save)
    callback = controller._accept_local_model if kind == "local" else controller._accept_val_result
    invalid = Shareable()
    invalid.add_cookie(AppConstants.MODEL_OWNER, "owner")
    callback("site", invalid, context)
    save.assert_not_called()
    assert error.called if kind == "local" else exception.called
    error.reset_mock()
    collection = DXO(DataKind.COLLECTION, {"good": DXO(DataKind.METRICS, {"m": 3}), "bad": 3}).to_shareable()
    collection.add_cookie(AppConstants.MODEL_OWNER, "owner")
    callback("site", collection, context)
    save.assert_called_once()
    assert error.called
    assert "bad" in error.call_args.args[1]


def test_failed_validation_creates_empty_result_only_for_returning_client(monkeypatch):
    controller, context = subject.CrossSiteModelEval(), FLContext()
    monkeypatch.setattr(controller, "fire_event", Mock())
    result = Shareable()
    result.set_return_code(ReturnCode.EXECUTION_EXCEPTION)
    result.add_cookie(AppConstants.MODEL_OWNER, "owner")
    controller._accept_val_result("late-site", result, context)
    assert controller._val_results == {"late-site": {"owner": {}}}


def test_waiting_for_first_client_stops_after_explicit_abort(monkeypatch):
    controller = subject.CrossSiteModelEval()
    controller._engine = Mock()
    controller._engine.get_clients.side_effect = [[]]
    abort = Signal()
    clock = isolate_time(monkeypatch, subject)
    clock.time = Mock(side_effect=[0, 1])
    clock.sleep = Mock(side_effect=lambda delay: abort.trigger(True))
    controller.control_flow(abort, FLContext())
    controller._engine.get_clients.assert_called_once()
    clock.sleep.assert_called_once_with(2.0)


def test_model_save_failure_retains_filename_in_diagnostic(tmp_path):
    controller = subject.CrossSiteModelEval()
    with pytest.raises(ValueError, match="Unable to save DXO.*model"):
        controller._save_dxo_content("model", str(tmp_path / "absent"), DXO(DataKind.WEIGHTS, {"w": 1}), FLContext())
    assert list(tmp_path.iterdir()) == []
