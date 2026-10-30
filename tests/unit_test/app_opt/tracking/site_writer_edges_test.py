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

from nvflare.apis.analytix import AnalyticsDataType, LogWriterName
from nvflare.apis.event_type import EventType
from nvflare.apis.fl_constant import ReservedKey
from nvflare.apis.fl_context import FLContext
from nvflare.app_opt.tracking.mlflow.mlflow_writer import MLflowWriter
from nvflare.app_opt.tracking.wandb.wandb_writer import WandBWriter


@pytest.mark.parametrize(
    "kind,method,args,tag,value,dtype,step",
    [
        (WandBWriter, "log", ({"loss": 2.0}, 3), "metrics", {"loss": 2.0}, AnalyticsDataType.METRICS, 3),
        (MLflowWriter, "log_param", ("lr", 0.1), "lr", 0.1, AnalyticsDataType.PARAMETER, None),
        (MLflowWriter, "log_params", ({"lr": 0.1},), "params", {"lr": 0.1}, AnalyticsDataType.PARAMETERS, None),
        (MLflowWriter, "log_metric", ("loss", 2.0, 3), "loss", 2.0, AnalyticsDataType.METRIC, 3),
        (MLflowWriter, "log_metrics", ({"loss": 2.0}, 3), "metrics", {"loss": 2.0}, AnalyticsDataType.METRICS, 3),
        (MLflowWriter, "log_text", ("note", "notes.txt"), "text", "note", AnalyticsDataType.TEXT, None),
        (MLflowWriter, "set_tag", ("run", "trial"), "run", "trial", AnalyticsDataType.TAG, None),
        (MLflowWriter, "set_tags", ({"run": "trial"},), "tags", {"run": "trial"}, AnalyticsDataType.TAGS, None),
    ],
)
def test_site_tracking_writers_bind_expected_writer_and_preserve_typed_metric_records(
    kind, method, args, tag, value, dtype, step
):
    sender, engine = Mock(), Mock()
    engine.get_component.return_value = sender
    ctx = FLContext()
    ctx.set_prop(ReservedKey.ENGINE, engine)
    writer = kind()
    writer.metrics_sender_id = "metrics"
    writer.handle_event(EventType.ABOUT_TO_START_RUN, ctx)
    assert sender.writer == (LogWriterName.WANDB if kind is WandBWriter else LogWriterName.MLFLOW)
    assert writer.get_default_metric_data_type() == AnalyticsDataType.METRICS
    getattr(writer, method)(*args)
    expected = dict(tag=tag, value=value, data_type=dtype, global_step=step)
    if method == "log_text":
        expected["path"] = "notes.txt"
    sender.add.assert_called_once_with(**expected)
