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

from unittest.mock import Mock

import pytest

from nvflare.apis.fl_context import FLContext
from nvflare.apis.fl_exception import TaskExecutionError
from nvflare.apis.shareable import Shareable
from nvflare.app_common.utils.fl_component_wrapper import FLComponentWrapper


def test_component_wrapper_uses_explicit_defaults_until_context_and_payload_arrive():
    wrapper = FLComponentWrapper()
    assert wrapper.get_shareable_header("round", 4) == 4
    assert wrapper.get_context_prop("identity", "missing") == "missing"
    assert wrapper.get_component("model") is None
    wrapper.fl_ctx = FLContext()
    wrapper.fl_ctx.set_prop("identity", "site")
    wrapper.shareable = Shareable({"model": "payload"})
    wrapper.shareable.set_header("round", 5)
    wrapper.engine = Mock()
    model = object()
    wrapper.engine.get_component.return_value = model
    assert wrapper.get_shareable_header("round", 4) == 5
    assert wrapper.get_context_prop("identity", "missing") == "site"
    assert wrapper.get_component("model") is model
    wrapper.engine.get_component.assert_called_once_with("model")


def test_component_stop_task_raises_specific_error_after_logging_reason(monkeypatch):
    wrapper = FLComponentWrapper()
    wrapper.fl_ctx = FLContext()
    error = Mock()
    monkeypatch.setattr(wrapper, "log_error", error)
    with pytest.raises(TaskExecutionError, match="invalid training data"):
        wrapper.stop_task("invalid training data")
    error.assert_called_once_with(wrapper.fl_ctx, "Task stopped: invalid training data")


def test_critical_log_keeps_the_current_framework_context(monkeypatch):
    wrapper = FLComponentWrapper()
    wrapper.fl_ctx = FLContext()
    critical = Mock()
    monkeypatch.setattr(wrapper, "log_critical", critical)
    wrapper.critical("training stopped")
    critical.assert_called_once_with(wrapper.fl_ctx, "training stopped")
