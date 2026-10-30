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

from unittest.mock import Mock

import pytest

from nvflare.apis.event_type import EventType
from nvflare.apis.fl_constant import FLContextKey
from nvflare.apis.fl_context import FLContext
from nvflare.edge.simulation.device_task_processor import DeviceTaskProcessor
from nvflare.edge.simulation.simulated_device import DeviceFactory
from nvflare.edge.widgets import device_runner, tpo_runner


@pytest.mark.parametrize("valid", [True, False])
def test_device_simulator_factory_validates_component_and_preserves_resource_limits(monkeypatch, valid):
    widget = device_runner.DeviceRunner("factory", num_workers=2, num_devices=10)
    ctx, engine = FLContext(), Mock()
    ctx.set_prop(FLContextKey.JOB_META, {"name": "simulation"})
    monkeypatch.setattr(ctx, "get_engine", lambda: engine)
    monkeypatch.setattr(widget, "system_panic", Mock())
    monkeypatch.setattr(widget, "log_debug", Mock())
    factory = Mock(spec=DeviceFactory) if valid else object()
    engine.get_component.return_value = factory
    construct = Mock()
    monkeypatch.setattr(device_runner, "Simulator", construct)
    result = widget.create_simulator(ctx)
    if valid:
        assert result is construct.return_value
        construct.assert_called_once_with(job_name="simulation", device_factory=factory, num_devices=10, num_workers=2)
        widget.system_panic.assert_not_called()
    else:
        assert result is None
        widget.system_panic.assert_called_once()
        construct.assert_not_called()


@pytest.mark.parametrize(
    "kwargs",
    [{"num_devices": 1}, {"num_devices": 1000001}, {"num_workers": 0}, {"num_workers": 101}, {"device_factory_id": 3}],
)
def test_device_runner_rejects_invalid_factory_or_resource_bounds(kwargs):
    with pytest.raises((ValueError, TypeError)):
        device_runner.DeviceRunner(**{"device_factory_id": "factory", **kwargs})


def test_task_processor_runner_requires_typed_component_before_building_simulator(monkeypatch):
    widget = tpo_runner.TPORunner("processor", job_timeout=8, num_devices=20, num_workers=3)
    ctx, engine = FLContext(), Mock()
    ctx.set_prop(FLContextKey.JOB_META, {"name": "simulation"})
    monkeypatch.setattr(ctx, "get_engine", lambda: engine)
    monkeypatch.setattr(widget, "log_info", Mock())
    engine.get_component.return_value = object()
    with pytest.raises(ValueError, match="should be DeviceTaskProcessor"):
        widget._tpo_about_to_start(EventType.ABOUT_TO_START_RUN, ctx)
    processor = Mock(spec=DeviceTaskProcessor)
    engine.get_component.return_value = processor
    widget._tpo_about_to_start(EventType.ABOUT_TO_START_RUN, ctx)
    factory, create = Mock(), Mock()
    monkeypatch.setattr(tpo_runner, "TPODeviceFactory", factory)
    monkeypatch.setattr(tpo_runner, "Simulator", create)
    assert widget.create_simulator(ctx) is create.return_value
    factory.assert_called_once_with(processor)
    create.assert_called_once_with(
        job_name="simulation", get_job_timeout=8, device_factory=factory.return_value, num_devices=20, num_workers=3
    )
