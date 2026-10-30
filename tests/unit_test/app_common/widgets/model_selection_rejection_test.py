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

from nvflare.apis.dxo import DXO, DataKind, MetaKey
from nvflare.apis.event_type import EventType
from nvflare.apis.fl_constant import FLContextKey, ReservedKey
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.app_common.app_constant import AppConstants
from nvflare.app_common.app_event_type import AppEventType
from nvflare.app_common.ccwf.comps.simple_intime_model_selector import SimpleIntimeModelSelector
from nvflare.app_common.widgets.intime_model_selector import IntimeModelSelector


@pytest.mark.parametrize("selector_type", [SimpleIntimeModelSelector, IntimeModelSelector])
@pytest.mark.parametrize(
    "failure",
    ["invalid-dxo", "wrong-kind", "initial-round", "stale-round", "missing-metric", "missing-key", "not-number"],
)
def test_unusable_selection_metrics_never_contribute_to_best_model(selector_type, failure):
    selector = selector_type(key_metric="accuracy")
    selector.fire_event = Mock()
    ctx, peer = FLContext(), FLContext()
    peer.set_prop(ReservedKey.IDENTITY_NAME, "one", private=False, sticky=True)
    ctx.set_peer_context(peer)
    ctx.set_prop(AppConstants.CURRENT_ROUND, 0 if failure == "initial-round" else 1, private=True, sticky=True)
    metric = (
        {"other": 0.5} if failure == "missing-key" else (object() if failure == "not-number" else {"accuracy": 0.5})
    )
    meta = {} if failure == "missing-metric" else {MetaKey.INITIAL_METRICS: metric}
    data = (
        Shareable()
        if failure == "invalid-dxo"
        else DXO(DataKind.METRICS if failure == "wrong-kind" else DataKind.WEIGHTS, {"w": [1]}, meta).to_shareable()
    )
    data.add_cookie(
        AppConstants.CONTRIBUTION_ROUND, 0 if failure == "stale-round" else (0 if failure == "initial-round" else 1)
    )
    ctx.set_prop(AppConstants.TRAINING_RESULT, data, private=True, sticky=False)
    peer.set_prop(FLContextKey.SHAREABLE, data, private=True, sticky=False)
    selector.handle_event(EventType.START_RUN, ctx)
    assert selector._before_accept(ctx) is False
    assert selector.validation_metric_sum_of_weights == 0
    assert selector._before_aggregate(ctx) is False
    assert AppEventType.GLOBAL_BEST_MODEL_AVAILABLE not in [
        c.args[0] if c.args else c.kwargs.get("event_type") for c in selector.fire_event.call_args_list
    ]
