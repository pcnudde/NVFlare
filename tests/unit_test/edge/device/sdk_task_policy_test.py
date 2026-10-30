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

from nvflare.apis.dxo import DXO, from_dict
from nvflare.edge.device import sdk_spec as subject
from nvflare.edge.device.config import TrainConfig
from nvflare.edge.device.defs import ContextKey, DataSource, EventHandler, EventType, Executor, Filter
from nvflare.edge.device.sdk_spec import FlareRunner


class NumberFilter(Filter):
    def __init__(self, label, amount, trace):
        self.label, self.amount, self.trace = label, amount, trace

    def filter(self, data, ctx, abort_signal):
        self.trace.append(self.label)
        return DXO("number", {"value": data.data["value"] + self.amount})


def task(number=2, task_id="first"):
    return {
        "task_name": "train",
        "task_id": task_id,
        "cookie": {"task": task_id},
        "task_data": DXO("number", {"value": number}).to_dict(),
    }


@pytest.fixture
def runtime(monkeypatch):
    trace = []
    runner = FlareRunner(
        "requested",
        Mock(spec=DataSource),
        {"id": "device"},
        {"id": "user"},
        3,
        in_filters=[NumberFilter("app-in", 1, trace)],
        out_filters=[NumberFilter("app-out", 4, trace)],
        resolver_registry={"native": int},
    )
    executor = Mock(spec=Executor)
    executor.execute.side_effect = lambda data, ctx, signal: DXO("number", {"value": data.data["value"] + 10})
    handler = Mock(spec=EventHandler)
    config = TrainConfig(
        {"native": "component"},
        [NumberFilter("job-in", 2, trace)],
        [NumberFilter("job-out", 5, trace)],
        [handler],
        {"train": executor},
    )
    process = Mock(return_value=config)
    monkeypatch.setattr(subject, "process_train_config", process)
    runner._get_job = Mock(return_value={"job_name": "assigned", "job_id": "job-1", "job_data": {"config": 1}})
    runner._get_task = Mock(side_effect=[(task(), False), (None, False)])
    runner._report_result = Mock(return_value=False)
    return runner, executor, config, handler, trace, process


def test_task_filters_run_in_app_then_job_order_and_each_task_gets_isolated_context(runtime):
    runner, executor, config, handler, trace, process = runtime
    runner._get_task.side_effect = [(task(), False), (task(3, "second"), False)]
    runner._report_result.side_effect = [False, True]
    contexts = []

    def train(data, ctx, signal):
        assert "scratch" not in ctx
        ctx["scratch"] = ctx[ContextKey.TASK_ID]
        contexts.append(ctx)
        return DXO("number", {"value": data.data["value"] + 10})

    executor.execute.side_effect = train
    assert runner._do_one_job() is True
    assert trace == ["app-in", "job-in", "app-out", "job-out"] * 2
    results = [from_dict(c.args[0]).data for c in runner._report_result.call_args_list]
    assert results == [{"value": 24}, {"value": 25}]
    assert contexts[0] is not contexts[1]
    assert [c[ContextKey.TASK_ID] for c in contexts] == ["first", "second"]
    assert all(c[ContextKey.RUNNER] is runner and c[ContextKey.DATA_SOURCE] is runner.data_source for c in contexts)
    assert all(c[ContextKey.EXECUTOR] is executor and c[ContextKey.COMPONENTS] is config.objects for c in contexts)
    assert [c.args[0] for c in handler.handle_event.call_args_list] == [
        EventType.BEFORE_TRAIN,
        EventType.AFTER_TRAIN,
    ] * 2
    assert runner.job_id == "job-1" and runner.job_name == "assigned" and runner.cookie == {"task": "second"}
    process.assert_called_once_with({"config": 1}, {"native": int})


def test_no_job_or_no_task_preserves_session_completion_semantics(runtime):
    runner, executor, _, _, _, process = runtime
    runner._get_job.return_value = None
    assert runner._do_one_job() is True
    process.assert_not_called()
    runner._get_job.return_value = {"job_id": "job", "job_data": {}}
    runner._get_task.side_effect = [(None, False)]
    assert runner._do_one_job() is False
    executor.execute.assert_not_called()
    runner._report_result.assert_not_called()


@pytest.mark.parametrize("failure", ["missing_executor", "invalid_executor", "input", "output", "output_filter"])
def test_invalid_executor_and_non_dxo_boundaries_fail_before_reporting(runtime, failure):
    runner, executor, config, _, _, _ = runtime
    if failure == "missing_executor":
        config.executors = {"other": executor}
    elif failure == "invalid_executor":
        config.executors = {"train": object()}
    elif failure == "input":
        broken = Mock(spec=Filter)
        broken.filter.return_value = None
        runner.app_in_filters = [broken]
        config.in_filters = []
    elif failure == "output":
        executor.execute.side_effect = None
        executor.execute.return_value = None
    else:
        broken = Mock(spec=Filter)
        broken.filter.return_value = None
        runner.app_out_filters, config.out_filters = [broken], []
    with pytest.raises(RuntimeError, match="executor|valid DXO"):
        runner._do_one_job()
    runner._report_result.assert_not_called()
    if failure in ("missing_executor", "invalid_executor", "input"):
        executor.execute.assert_not_called()


@pytest.mark.parametrize("stage", ["task", "input", "train", "output", "report"])
def test_cancellation_at_each_boundary_prevents_following_pipeline_stage(runtime, stage):
    runner, executor, config, _, _, _ = runtime
    runner.app_in_filters, config.in_filters = [], []
    runner.app_out_filters, config.out_filters = [], []

    def abort_filter(data, ctx, signal):
        signal.trigger(True)
        return data

    stopper = Mock(spec=Filter)
    stopper.filter.side_effect = abort_filter
    if stage == "task":

        def get_task(ctx, signal):
            signal.trigger(True)
            return task(), False

        runner._get_task.side_effect = get_task
    elif stage == "input":
        runner.app_in_filters = [stopper]
    elif stage == "train":
        executor.execute.side_effect = abort_filter
    elif stage == "output":
        runner.app_out_filters = [stopper]
    else:

        def report(data, ctx, signal):
            signal.trigger(True)
            return False

        runner._report_result.side_effect = report
    assert runner._do_one_job() is True
    assert runner.abort_signal.triggered
    assert runner._report_result.called is (stage == "report")
    assert executor.execute.called is (stage in ("train", "output", "report"))


def test_run_repeats_jobs_until_session_done_and_stop_sets_abort_signal():
    runner = FlareRunner("job", Mock(spec=DataSource), {}, {}, 3)
    runner._do_one_job = Mock(side_effect=[False, False, True])
    runner.run()
    assert runner._do_one_job.call_count == 3
    runner.stop()
    assert runner.abort_signal.triggered
    with pytest.raises(ValueError, match="resolver_registry must be dict"):
        FlareRunner("job", Mock(spec=DataSource), {}, {}, 3, resolver_registry=["invalid"])
