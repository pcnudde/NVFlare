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

from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.apis.signal import Signal
from nvflare.app_common.abstract.aggregator import Aggregator
from nvflare.app_common.abstract.metric_comparator import MetricComparator
from nvflare.app_common.ccwf import cse_client_ctl, swarm_client_ctl
from nvflare.app_common.ccwf.client_ctl import ClientSideController
from nvflare.app_common.ccwf.common import Constant


@pytest.mark.parametrize("component", ["invalid-aggregator", "invalid-comparator", "valid-comparator"])
def test_swarm_validates_both_components_before_starting_background_aggregation(monkeypatch, component):
    monkeypatch.setattr(ClientSideController, "start_run", Mock())
    ctl = swarm_client_ctl.SwarmClientController(metric_comparator_id="metric", enable_tensor_disk_offload=True)
    ctl.engine = Mock()
    ctl.engine.get_cell.return_value = None
    aggregator, comparator = Mock(spec=Aggregator), Mock(spec=MetricComparator)
    ctl.engine.get_component.side_effect = [
        object() if component == "invalid-aggregator" else aggregator,
        object() if component == "invalid-comparator" else comparator,
    ]
    panic, warning = Mock(), Mock()
    monkeypatch.setattr(ctl, "system_panic", panic)
    monkeypatch.setattr(ctl, "log_warning", warning)
    thread = Mock()
    monkeypatch.setattr(swarm_client_ctl, "threading", SimpleNamespace(Thread=Mock(return_value=thread)))
    ctl.start_run(FLContext())
    warning.assert_called_once()
    assert panic.call_count == int(component != "valid-comparator")
    assert thread.start.call_count == int(component == "valid-comparator")
    if component == "valid-comparator":
        assert ctl.metric_comparator is comparator and ctl.aggregator is aggregator


def test_swarm_defaults_trainer_and_aggregator_membership_to_configured_clients():
    ctl = swarm_client_ctl.SwarmClientController()
    ctl.config = {Constant.CLIENTS: ["one", "two"]}
    ctl.engine, ctl.me = Mock(), "one"
    ctl.process_config(FLContext())
    assert ctl.trainers == ctl.aggrs == ["one", "two"]
    assert ctl.is_trainer and ctl.is_aggr
    assert ctl.engine.register_aux_message_handler.call_count == 2


def test_aggregation_exception_publishes_failure_status_without_starting_another_round(monkeypatch):
    ctl = swarm_client_ctl.SwarmClientController()
    gatherer = Mock()
    gatherer.fl_ctx = FLContext()
    gatherer.aggregate.side_effect = ValueError("invalid weights")
    status = Mock()
    monkeypatch.setattr(ctl, "update_status", status)
    ctl._end_gather(gatherer)
    status.assert_called_once_with(action="aggregate", error=swarm_client_ctl.ReturnCode.EXECUTION_EXCEPTION)


@pytest.mark.parametrize("route", ["eval", "prepare", "fetch", "parent"])
def test_cross_site_controller_dispatch_preserves_exact_request_and_abort_signal(monkeypatch, route):
    ctl = cse_client_ctl.CrossSiteEvalClientController()
    request, ctx, abort = Shareable(), FLContext(), Signal()
    result = Shareable({"accepted": route})
    boundaries = {"eval": "_do_eval", "prepare": "_prepare_model", "fetch": "_process_get_model_request"}
    handlers = {}
    for name in boundaries.values():
        handlers[name] = Mock(return_value=result)
        monkeypatch.setattr(ctl, name, handlers[name])
    parent = Mock(return_value=result)
    monkeypatch.setattr(ClientSideController, "execute", parent)
    collect = Mock()
    monkeypatch.setattr(cse_client_ctl, "gc", SimpleNamespace(collect=collect))
    task = {
        "eval": ctl.eval_task_name,
        "prepare": ctl.prep_model_task_name,
        "fetch": ctl.ask_for_model_task_name,
        "parent": "unknown",
    }[route]
    assert ctl.execute(task, request, ctx, abort) is result
    if route == "parent":
        parent.assert_called_once_with(task, request, ctx, abort)
    elif route == "fetch":
        handlers[boundaries[route]].assert_called_once_with(request, ctx)
    else:
        handlers[boundaries[route]].assert_called_once_with(request, ctx, abort)
    assert collect.call_count == int(route == "eval")
