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
from nvflare.app_common.abstract.learnable_persistor import LearnablePersistor
from nvflare.app_common.abstract.model import make_model_learnable
from nvflare.app_common.abstract.shareable_generator import ShareableGenerator
from nvflare.app_common.app_constant import AppConstants
from nvflare.app_common.app_event_type import AppEventType
from nvflare.app_common.shareablegenerators.passthru import PassthroughShareableGenerator
from nvflare.edge.assessor import Assessment
from nvflare.edge.assessors.sgap import SGAPAssessor


@pytest.fixture
def context():
    ctx = FLContext()
    ctx.set_prop(ReservedKey.ENGINE, Mock())
    ctx.set_prop(AppConstants.CURRENT_ROUND, 3)
    return ctx


@pytest.mark.parametrize("field", ["shareable_generator_id", "aggregator_id", "persistor_id"])
def test_rejects_non_string_component_identifiers(field):
    values = dict(shareable_generator_id="", aggregator_id="aggr", persistor_id="")
    values[field] = 1
    with pytest.raises(TypeError, match=field):
        SGAPAssessor(**values)


def test_loads_initial_model_and_announces_it(context):
    aggr = Mock(spec=Aggregator)
    generator = Mock(spec=ShareableGenerator)
    persistor = Mock(spec=LearnablePersistor)
    model = make_model_learnable({"weight": [1, 2]}, {"version": 2})
    persistor.load.return_value = model
    components = {"aggr": aggr, "gen": generator, "persistor": persistor}
    context.get_engine().get_component.side_effect = components.get
    assessor = SGAPAssessor("gen", "aggr", "persistor")
    assessor.fire_event = Mock()

    assessor._handle_start_run(EventType.START_RUN, context)

    assert assessor.aggregator is aggr
    assert assessor.shareable_gen is generator
    assert assessor.persistor is persistor
    assert assessor._global_weights is model
    assert context.get_prop(AppConstants.GLOBAL_MODEL) is model
    persistor.load.assert_called_once_with(context)
    assessor.fire_event.assert_called_once_with(AppEventType.INITIAL_MODEL_LOADED, context)


@pytest.mark.parametrize("bad_component", ["aggr", "gen", "persistor", "model"])
def test_invalid_component_or_loaded_model_panics_without_publishing(bad_component, context):
    components = {
        "aggr": Mock(spec=Aggregator),
        "gen": Mock(spec=ShareableGenerator),
        "persistor": Mock(spec=LearnablePersistor),
    }
    components["persistor"].load.return_value = make_model_learnable({}, {})
    if bad_component == "model":
        components["persistor"].load.return_value = {"weights": {}}
    else:
        components[bad_component] = object()
    context.get_engine().get_component.side_effect = components.get
    assessor = SGAPAssessor("gen", "aggr", "persistor")
    assessor.system_panic = Mock()
    assessor.fire_event = Mock()

    assessor._handle_start_run(EventType.START_RUN, context)

    assessor.system_panic.assert_called_once()
    assert context.get_prop(AppConstants.GLOBAL_MODEL) is None
    assessor.fire_event.assert_not_called()


def test_default_generator_serializes_task_and_publishes_aggregated_weights(context):
    aggr = Mock(spec=Aggregator)
    context.get_engine().get_component.return_value = aggr
    assessor = SGAPAssessor("", "aggr", "")
    assessor._handle_start_run(EventType.START_RUN, context)
    assessor.fire_event = Mock()
    assessor.fire_event_with_data = Mock()
    assessor._global_weights = make_model_learnable({"weight": [1.0]}, {})
    aggr.aggregate.return_value = Shareable({"weights": {"weight": [4.0]}, "meta": {"round": 3}})

    task = assessor.start_task(context)
    assessor.end_task(context)

    assert isinstance(assessor.shareable_gen, PassthroughShareableGenerator)
    assert task["weights"] == {"weight": [1.0]}
    assert context.get_prop(AppConstants.GLOBAL_MODEL)["weights"] == {"weight": [4.0]}
    assert assessor._global_weights["meta"] == {"round": 3}
    aggr.aggregate.assert_called_once_with(context)
    aggr.reset.assert_called_once_with(context)
    assert [c.args[0] for c in assessor.fire_event.call_args_list] == [
        AppEventType.BEFORE_AGGREGATION,
        AppEventType.BEFORE_SHAREABLE_TO_LEARNABLE,
        AppEventType.AFTER_SHAREABLE_TO_LEARNABLE,
    ]
    assessor.fire_event_with_data.assert_called_once_with(
        AppEventType.AFTER_AGGREGATION, context, AppConstants.AGGREGATION_RESULT, aggr.aggregate.return_value
    )
    assert assessor.assess(context) == Assessment.CONTINUE


def test_persists_converted_model_after_successful_aggregation(context):
    assessor = SGAPAssessor("", "aggr", "persistor")
    assessor.aggregator = Mock(spec=Aggregator)
    assessor.shareable_gen = PassthroughShareableGenerator()
    assessor.persistor = Mock(spec=LearnablePersistor)
    assessor.aggregator.aggregate.return_value = Shareable({"weights": {"w": 7}})
    assessor.fire_event = Mock()
    assessor.fire_event_with_data = Mock()

    assessor.end_task(context)

    saved, saved_ctx = assessor.persistor.save.call_args.args
    assert saved is assessor._global_weights
    assert saved["weights"] == {"w": 7}
    assert saved_ctx is context
    assert [c.args[0] for c in assessor.fire_event.call_args_list][-2:] == [
        AppEventType.BEFORE_LEARNABLE_PERSIST,
        AppEventType.AFTER_LEARNABLE_PERSIST,
    ]


@pytest.mark.parametrize("failing_operation", ["aggregate", "reset"])
def test_failed_aggregation_preserves_model_and_never_saves(failing_operation, context):
    assessor = SGAPAssessor("", "aggr", "persistor")
    assessor.aggregator = Mock(spec=Aggregator)
    getattr(assessor.aggregator, failing_operation).side_effect = ValueError("broken update")
    assessor.persistor = Mock(spec=LearnablePersistor)
    assessor.system_panic = Mock()
    original = assessor._global_weights

    assessor.end_task(context)

    assert assessor._global_weights is original
    assert "broken update" in assessor.system_panic.call_args.args[0]
    assessor.persistor.save.assert_not_called()
    if failing_operation == "aggregate":
        assessor.aggregator.reset.assert_not_called()


@pytest.mark.parametrize("accepted", [True, False])
def test_child_update_acceptance_is_reported_without_reply(accepted, context):
    assessor = SGAPAssessor("", "aggr", "")
    assessor.aggregator = Mock(spec=Aggregator)
    assessor.aggregator.accept.return_value = accepted
    update = Shareable({"weights": {"w": 5}})

    assert assessor.process_child_update(update, context) == (accepted, None)
    assessor.aggregator.accept.assert_called_once_with(update, context)
