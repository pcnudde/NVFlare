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

import numpy as np
import pytest

from nvflare.apis.dxo import DXO, DataKind
from nvflare.apis.event_type import EventType
from nvflare.apis.fl_constant import ReservedKey
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.app_common.app_constant import AppConstants
from nvflare.app_common.app_event_type import AppEventType
from nvflare.app_common.widgets.validation_json_generator import ValidationJsonGenerator, to_serializable


@pytest.fixture
def runtime(tmp_path):
    widget = ValidationJsonGenerator()
    ctx = FLContext()
    engine = Mock()
    engine.get_workspace.return_value.get_run_dir.return_value = str(tmp_path)
    ctx.set_prop(ReservedKey.ENGINE, engine, private=True, sticky=True)
    ctx.set_prop(AppConstants.MODEL_OWNER, "global", private=True, sticky=False)
    ctx.set_prop(AppConstants.DATA_CLIENT, "site", private=True, sticky=False)
    return widget, ctx, tmp_path


def test_validation_widget_preserves_nested_metrics_and_serializes_numpy_scalars(runtime):
    widget, ctx, directory = runtime
    widget.handle_event(EventType.START_RUN, ctx)
    leaf = DXO(DataKind.METRICS, {"accuracy": np.float32(0.75)})
    payload = DXO(DataKind.COLLECTION, {"child": leaf, "broken": object()}).to_shareable()
    ctx.set_prop(AppConstants.VALIDATION_RESULT, payload, private=True, sticky=False)
    widget.log_error = Mock()
    widget.handle_event(AppEventType.VALIDATION_RESULT_RECEIVED, ctx)
    widget.log_error.assert_called_once()
    widget.handle_event(EventType.END_RUN, ctx)
    result = json.loads((directory / AppConstants.CROSS_VAL_DIR / "cross_val_results.json").read_text())
    assert result == {"site.child": {"global": {"accuracy": 0.75}}}
    assert to_serializable(123) == "123"
    widget.handle_event(EventType.START_RUN, ctx)
    assert widget._val_results == {}


@pytest.mark.parametrize("payload", [None, Shareable(), DXO(DataKind.WEIGHTS, {"w": [1]}).to_shareable()])
def test_validation_widget_contains_missing_malformed_and_wrong_kind_results(runtime, payload):
    widget, ctx, _ = runtime
    ctx.set_prop(AppConstants.VALIDATION_RESULT, payload, private=True, sticky=False)
    widget.log_error, widget.log_exception = Mock(), Mock()
    widget.handle_event(AppEventType.VALIDATION_RESULT_RECEIVED, ctx)
    assert widget._val_results == {}
    assert widget.log_error.called or widget.log_exception.called
