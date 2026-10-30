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

from nvflare.apis.analytix import AnalyticsDataType, LogWriterName
from nvflare.apis.event_type import EventType
from nvflare.apis.fl_constant import ReservedKey
from nvflare.apis.fl_context import FLContext
from nvflare.app_common.tracking import log_writer as subject


class RecordingWriter(subject.LogWriter):
    def get_writer_name(self):
        return LogWriterName.TORCH_TB


@pytest.mark.parametrize("configured", [False, True])
def test_log_writer_routes_records_through_configured_or_owned_analytics_sender(monkeypatch, configured):
    engine, context, sender = Mock(), FLContext(), Mock()
    context.set_prop(ReservedKey.ENGINE, engine)
    engine.get_component.return_value = sender
    create = Mock(return_value=sender)
    monkeypatch.setattr(subject, "AnalyticsSender", create)
    writer = RecordingWriter(metrics_sender_id="sender" if configured else None)
    writer.handle_event(EventType.ABOUT_TO_START_RUN, context)
    assert writer.sender is sender
    if configured:
        engine.get_component.assert_called_once_with("sender")
        assert sender.writer == LogWriterName.TORCH_TB
        create.assert_not_called()
    else:
        assert sender.engine is engine
        create.assert_called_once_with(writer.event_type, LogWriterName.TORCH_TB)
    writer.write("loss", 0.2, AnalyticsDataType.SCALAR, global_step=3, precision=4)
    sender.add.assert_called_once_with(
        tag="loss", value=0.2, data_type=AnalyticsDataType.SCALAR, global_step=3, precision=4
    )
    assert writer.get_default_metric_data_type() == AnalyticsDataType.METRICS
