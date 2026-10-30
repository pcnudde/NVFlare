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

from nvflare.apis.event_type import EventType
from nvflare.apis.fl_context import FLContext
from nvflare.app_common.abstract.fl_model import FLModel
from nvflare.app_common.aggregators.model_aggregator import ModelAggregator
from nvflare.app_common.utils.fl_model_utils import FLModelUtils


class SumAggregator(ModelAggregator):
    def __init__(self):
        super().__init__()
        self.values = []

    def accept_model(self, model):
        self.values.append(model.params["weight"])

    def aggregate_model(self):
        return FLModel(params={"weight": sum(self.values)})

    def reset_stats(self):
        self.values.clear()


def test_model_aggregator_adapter_round_trips_envelopes_and_resets_under_current_context(monkeypatch):
    aggregator, context = SumAggregator(), FLContext()
    aggregator.handle_event(EventType.START_RUN, context)
    assert aggregator.fl_ctx is context
    for value in (2, 3):
        assert aggregator.accept(FLModelUtils.to_shareable(FLModel(params={"weight": value})), context)
    assert FLModelUtils.from_shareable(aggregator.aggregate(context)).params == {"weight": 5}
    assert aggregator.fl_ctx is context
    aggregator.reset(context)
    assert aggregator.values == []
    log = Mock()
    monkeypatch.setattr(aggregator, "log_exception", log)
    aggregator.exception("aggregation failed")
    log.assert_called_once_with(fl_ctx=context, msg="aggregation failed")
