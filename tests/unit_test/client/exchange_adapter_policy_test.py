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

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from nvflare.apis.analytix import AnalyticsDataType, LogWriterName
from nvflare.app_common.abstract.fl_model import FLModel
from nvflare.app_common.app_constant import AppConstants
from nvflare.client import api, decorator, tracking
from nvflare.client.in_process.adapters import TASK_RESULT_STOP_REASON, TaskInProcessAdapter
from nvflare.fuel.utils import deprecated as deprecation_module


@pytest.fixture
def exchange(monkeypatch):
    monkeypatch.setattr(deprecation_module, "_WARNED_DEPRECATION_MESSAGES", set())
    incoming = FLModel(params={"w": [1, 2]})
    received, sent = Mock(return_value=incoming), Mock()
    monkeypatch.setattr(decorator, "receive", received)
    monkeypatch.setattr(decorator, "send", sent)
    monkeypatch.setattr(decorator, "object_holder", decorator.ObjectHolder())
    return incoming, received, sent


@pytest.mark.parametrize("factory_style", [False, True])
def test_evaluation_metrics_are_consumed_once_by_training_without_changing_other_arguments(
    exchange, monkeypatch, factory_style
):
    incoming, received, sent = exchange
    monkeypatch.setattr(decorator, "is_train", lambda: True)

    def evaluate(model, scale):
        assert model is incoming
        return {"loss": 4 / scale}

    def train(model, steps):
        assert model is incoming
        return FLModel(params={"w": [steps]})

    eval_fn = decorator.evaluate()(evaluate) if factory_style else decorator.evaluate(evaluate)
    train_fn = decorator.train()(train) if factory_style else decorator.train(train)
    assert eval_fn(scale=2) == {"loss": 2}
    sent.assert_not_called()
    result = train_fn(steps=3)
    assert result.params == {"w": [3]}
    assert result.metrics == {"loss": 2}
    assert sent.call_args.kwargs["model"] is result
    assert decorator.object_holder.metrics is None
    assert train_fn(steps=5).metrics is None
    assert received.call_count == 3
    assert train_fn.__name__ == "train"


@pytest.mark.parametrize("value,match", [(None, "return value is None"), (3, "needs to be an FLModel")])
def test_train_rejects_invalid_user_result_before_publishing(exchange, value, match):
    wrapped = decorator.train(lambda model: value)
    with pytest.raises(RuntimeError, match=match):
        wrapped()
    exchange[2].assert_not_called()


def test_evaluate_only_task_publishes_metric_model_and_rejects_missing_result(exchange, monkeypatch):
    monkeypatch.setattr(decorator, "is_train", lambda: False)
    evaluate = decorator.evaluate(lambda model: {"accuracy": 0.75})
    assert evaluate() == {"accuracy": 0.75}
    result = exchange[2].call_args.kwargs["model"]
    assert result.params is None and result.metrics == {"accuracy": 0.75}
    assert decorator.object_holder.metrics is None
    missing = decorator.evaluate(lambda model: None)
    with pytest.raises(RuntimeError, match="return value is None"):
        missing()
    assert exchange[2].call_count == 1


def test_task_adapter_keeps_only_frozen_result_reference_and_rejects_second_publish():
    written, logs = [], []
    reference = object()

    def save(result):
        written.append(dict(result))
        return reference

    adapter = TaskInProcessAdapter(on_result=save, log_handler=logs.append)
    engine = SimpleNamespace(_set_received_shareable=Mock(), stop=False, stop_reason=None)
    adapter.bind(engine, None, None)
    from nvflare.apis.shareable import Shareable

    incoming = Shareable({"input": [2]})
    incoming.set_header(AppConstants.CURRENT_ROUND, 7)
    adapter.stage_input(incoming)
    engine._set_received_shareable.assert_called_once_with(incoming)
    result = Shareable({"weights": [3]})
    adapter.publish_result(result)
    assert result.get_header(AppConstants.CURRENT_ROUND) == 7
    assert written[0]["weights"] == [3]
    assert adapter.get_result() is reference
    assert engine.stop and engine.stop_reason == TASK_RESULT_STOP_REASON
    adapter.publish_log({"loss": 2})
    assert logs == [{"loss": 2}]
    with pytest.raises(RuntimeError, match="exactly one result"):
        adapter.publish_result(Shareable())
    assert len(written) == 1
    adapter.close()


def test_task_adapter_requires_writer_and_completed_result_without_stopping_api():
    adapter = TaskInProcessAdapter()
    engine = SimpleNamespace(stop=False)
    adapter.bind(engine, None, None)
    from nvflare.apis.shareable import Shareable

    with pytest.raises(RuntimeError, match="without sending a result"):
        adapter.get_result()
    with pytest.raises(RuntimeError, match="writer is not configured"):
        adapter.publish_result(Shareable())
    assert not engine.stop
    adapter.publish_log("ignored without a handler")


@pytest.mark.parametrize(
    "method",
    [
        "get_job_id",
        "get_site_name",
        "get_task_name",
        "is_running",
        "is_train",
        "is_evaluate",
        "is_submit_model",
        "clear",
        "shutdown",
    ],
)
def test_public_client_methods_select_explicit_context_over_default_without_touching_other_runtime(monkeypatch, method):
    explicit, default = SimpleNamespace(api=Mock()), SimpleNamespace(api=Mock())
    monkeypatch.setattr(api, "default_context", default)
    getattr(api, method)(ctx=explicit)
    getattr(explicit.api, method).assert_called_once_with()
    assert not default.api.mock_calls


def test_public_client_receive_and_send_preserve_timeout_and_cache_policy(monkeypatch):
    ctx = SimpleNamespace(api=Mock())
    monkeypatch.setattr(api, "default_context", ctx)
    api.receive(timeout=2.5)
    ctx.api.receive.assert_called_once_with(2.5)
    model = FLModel(params={"x": [1]})
    api.send(model, clear_cache=False)
    ctx.api.send.assert_called_once_with(model, False)
    with pytest.raises(TypeError, match="instance of FLModel"):
        api.send({"params": {}})
    assert ctx.api.send.call_count == 1


def test_tracking_shutdown_clears_only_owned_analytics_context_and_delegates_other_context(monkeypatch):
    sender = Mock(spec=tracking.MetricsSender)
    monkeypatch.setattr(tracking, "_tracking_context", sender)
    assert tracking.get_context() is sender
    tracking.shutdown()
    assert tracking._tracking_context is None
    sender.shutdown.assert_called_once_with()
    tracking.shutdown()
    other = object()
    shutdown = Mock()
    monkeypatch.setattr(api, "shutdown", shutdown)
    tracking.shutdown(other)
    shutdown.assert_called_once_with(other)
    monkeypatch.setattr(api, "get_context", Mock(return_value=other))
    assert tracking.get_context() is other


@pytest.mark.parametrize(
    "writer,method,args,expected",
    [
        (
            tracking.SummaryWriter,
            "add_scalars",
            ("train", {"loss": 2}, 3),
            {
                "key": "train",
                "value": {"loss": 2},
                "data_type": AnalyticsDataType.SCALARS,
                "global_step": 3,
                "writer": LogWriterName.TORCH_TB,
            },
        ),
        (
            tracking.WandBWriter,
            "log",
            ({"loss": 2}, 3),
            {
                "key": "metrics",
                "value": {"loss": 2},
                "data_type": AnalyticsDataType.METRICS,
                "global_step": 3,
                "writer": LogWriterName.WANDB,
            },
        ),
        (
            tracking.MLflowWriter,
            "log_param",
            ("lr", 0.1),
            {"key": "lr", "value": 0.1, "data_type": AnalyticsDataType.PARAMETER, "writer": LogWriterName.MLFLOW},
        ),
        (
            tracking.MLflowWriter,
            "log_params",
            ({"lr": 0.1},),
            {
                "key": "params",
                "value": {"lr": 0.1},
                "data_type": AnalyticsDataType.PARAMETERS,
                "writer": LogWriterName.MLFLOW,
            },
        ),
        (
            tracking.MLflowWriter,
            "log_metric",
            ("loss", 2, 3),
            {
                "key": "loss",
                "value": 2,
                "data_type": AnalyticsDataType.METRIC,
                "global_step": 3,
                "writer": LogWriterName.MLFLOW,
            },
        ),
        (
            tracking.MLflowWriter,
            "log_metrics",
            ({"loss": 2}, 3),
            {
                "key": "metrics",
                "value": {"loss": 2},
                "data_type": AnalyticsDataType.METRICS,
                "global_step": 3,
                "writer": LogWriterName.MLFLOW,
            },
        ),
        (
            tracking.MLflowWriter,
            "log_text",
            ("hello", "note.txt"),
            {
                "key": "text",
                "value": "hello",
                "data_type": AnalyticsDataType.TEXT,
                "path": "note.txt",
                "writer": LogWriterName.MLFLOW,
            },
        ),
        (
            tracking.MLflowWriter,
            "set_tag",
            ("run", "trial"),
            {"key": "run", "value": "trial", "data_type": AnalyticsDataType.TAG, "writer": LogWriterName.MLFLOW},
        ),
        (
            tracking.MLflowWriter,
            "set_tags",
            ({"run": "trial"},),
            {
                "key": "tags",
                "value": {"run": "trial"},
                "data_type": AnalyticsDataType.TAGS,
                "writer": LogWriterName.MLFLOW,
            },
        ),
    ],
)
def test_tracking_writers_preserve_typed_wire_metadata_without_live_analytics_transport(
    monkeypatch, writer, method, args, expected
):
    publish = Mock()
    monkeypatch.setattr(tracking, "log", publish)
    context = object()
    getattr(writer(ctx=context), method)(*args)
    assert publish.call_args.kwargs == dict(expected, ctx=context)
