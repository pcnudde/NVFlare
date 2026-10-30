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
from nvflare.apis.fl_constant import ReservedKey
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.app_common.abstract.aggregator import Aggregator
from nvflare.edge.updaters.aggr import AggrUpdater


@pytest.mark.parametrize("value", [0, -1, "2", 1.5])
def test_rejects_invalid_acceptance_threshold(value):
    with pytest.raises((TypeError, ValueError), match="min_accepted"):
        AggrUpdater("aggr", value)


def test_rejected_updates_do_not_count_and_threshold_reset_starts_a_new_batch():
    ctx = FLContext()
    aggr = Mock(spec=Aggregator)
    engine = Mock()
    engine.get_component.return_value = aggr
    ctx.set_prop(ReservedKey.ENGINE, engine)
    updater = AggrUpdater("aggr", min_accepted=2)
    updater._handle_start_run(EventType.START_RUN, ctx)
    aggr.accept.side_effect = [True, False, True, True]
    aggr.aggregate.return_value = Shareable({"mean": 3.0})

    for expected in [(True, None), (False, None)]:
        assert updater.process_child_update(Shareable({"w": 2}), ctx) == expected
    assert updater.num_accepted == 1
    assert updater.prepare_update_for_parent(ctx) is None
    aggr.aggregate.assert_not_called()

    updater.process_child_update(Shareable({"w": 4}), ctx)
    result = updater.prepare_update_for_parent(ctx)
    assert result["mean"] == 3.0
    aggr.aggregate.assert_called_once_with(ctx)
    aggr.reset.assert_called_once_with(ctx)
    assert updater.num_accepted == 0
    updater.process_child_update(Shareable({"w": 6}), ctx)
    assert updater.prepare_update_for_parent(ctx) is None
    assert updater.num_accepted == 1
    updater.process_parent_update_reply(Shareable({"ack": True}), ctx)
    assert updater.num_accepted == 1


def test_invalid_aggregator_panics_without_binding_component():
    ctx = FLContext()
    engine = Mock()
    engine.get_component.return_value = object()
    ctx.set_prop(ReservedKey.ENGINE, engine)
    updater = AggrUpdater("aggr")
    updater.system_panic = Mock()

    updater._handle_start_run(EventType.START_RUN, ctx)

    assert updater.aggregator is None
    assert "Aggregator" in updater.system_panic.call_args.args[0]


def test_aggregation_failure_preserves_count_for_retry():
    updater = AggrUpdater("aggr", min_accepted=2)
    updater.aggregator = Mock(spec=Aggregator)
    updater.num_accepted = 2
    updater.aggregator.aggregate.side_effect = RuntimeError("cannot combine")

    with pytest.raises(RuntimeError, match="cannot combine"):
        updater.prepare_update_for_parent(FLContext())

    assert updater.num_accepted == 2
    updater.aggregator.reset.assert_not_called()
