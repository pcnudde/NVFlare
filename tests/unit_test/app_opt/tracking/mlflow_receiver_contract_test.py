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
from unittest.mock import MagicMock

import pytest

from nvflare.apis.analytix import AnalyticsData, AnalyticsDataType, LogWriterName
from nvflare.apis.fl_constant import ProcessType
from nvflare.apis.job_def import JobMetaKey
from nvflare.app_opt.tracking.mlflow import mlflow_receiver as subject


@pytest.fixture
def receiver():
    r = subject.MLflowReceiver(tracking_uri="http://tracking", buffer_flush_time=0)
    r.mlflow_clients["site"] = MagicMock()
    r.run_ids["site"] = "run-id"
    r._init_buffer(["site"])
    return r


@pytest.mark.parametrize(
    "kind,value,target,expected",
    [
        (AnalyticsDataType.METRIC, 1.5, AnalyticsDataType.METRICS, {"value": 1.5}),
        (AnalyticsDataType.METRICS, {"a": 1.0, "b": 2.0}, AnalyticsDataType.METRICS, {"a": 1.0, "b": 2.0}),
        (AnalyticsDataType.PARAMETER, 3, AnalyticsDataType.PARAMETERS, {"value": "3"}),
        (AnalyticsDataType.PARAMETERS, {"a": 3}, AnalyticsDataType.PARAMETERS, {"a": "3"}),
        (AnalyticsDataType.TAG, "tag", AnalyticsDataType.TAGS, {"value": "tag"}),
        (AnalyticsDataType.TAGS, {"a": "x"}, AnalyticsDataType.TAGS, {"a": "x"}),
    ],
)
def test_buffer_preserves_metric_step_and_converts_parameters_and_tags(
    receiver, monkeypatch, kind, value, target, expected
):
    monkeypatch.setattr(subject, "time", SimpleNamespace(time=lambda: 2.5))
    data = AnalyticsData("value", value, kind, sender=LogWriterName.MLFLOW, global_step=4)
    receiver.buffer_data(data, "site")
    items = list(receiver.buffer["site"][target])
    assert {item.key: item.value for item in items} == expected
    if target == AnalyticsDataType.METRICS:
        assert all(item.step == 4 and item.timestamp == 2500 for item in items)
    receiver.flush_buffers("site")
    kwargs = receiver.mlflow_clients["site"].log_batch.call_args.kwargs
    assert kwargs["run_id"] == "run-id"
    assert (
        kwargs[
            {
                AnalyticsDataType.METRICS: "metrics",
                AnalyticsDataType.PARAMETERS: "params",
                AnalyticsDataType.TAGS: "tags",
            }[target]
        ]
        == items
    )
    assert all(not b for b in receiver.buffer["site"].values())
    assert receiver.time_start == receiver.time_since_flush == 0


@pytest.mark.parametrize("missing", ["client", "run"])
def test_flush_requires_registered_client_and_run(receiver, missing):
    if missing == "client":
        receiver.mlflow_clients.clear()
    else:
        receiver.run_ids.clear()
    with pytest.raises(RuntimeError):
        receiver.flush_buffers("site")


def test_save_text_uses_artifact_path_and_finalize_flushes_metrics(receiver):
    ctx = MagicMock()
    ctx.get_peer_context.return_value = None
    text = AnalyticsData("note", "hello", AnalyticsDataType.TEXT, sender=LogWriterName.MLFLOW, path="notes/a.txt")
    receiver.save(ctx, text.to_dxo().to_shareable(), "site")
    receiver.mlflow_clients["site"].log_text.assert_called_once_with(
        run_id="run-id", text="hello", artifact_file="notes/a.txt"
    )
    metric = AnalyticsData("loss", 0.25, AnalyticsDataType.METRIC, sender=LogWriterName.MLFLOW)
    receiver.save(ctx, metric.to_dxo().to_shareable(), "site")
    batch = receiver.mlflow_clients["site"].log_batch.call_args.kwargs
    assert batch["metrics"][0].key == "loss" and batch["metrics"][0].value == 0.25
    receiver.finalize(ctx)
    receiver.mlflow_clients["site"].set_terminated.assert_called_once_with("run-id")


@pytest.mark.parametrize("server", [False, True])
def test_initialize_creates_experiment_and_per_site_run(monkeypatch, tmp_path, server):
    client = MagicMock()
    client.get_experiment_by_name.return_value = None
    client.create_experiment.return_value = "experiment"
    client.create_run.return_value.info.run_id = "created-run"
    monkeypatch.setattr(subject, "MlflowClient", MagicMock(return_value=client))
    monkeypatch.setattr(subject.mlflow, "set_tracking_uri", MagicMock())
    monkeypatch.setattr(subject, "time", SimpleNamespace(time=lambda: 123456))
    ctx = MagicMock()
    ctx.get_peer_context.return_value = None
    ctx.get_job_id.return_value = "simulate_job"
    ctx.get_prop.return_value = {JobMetaKey.JOB_NAME: "job"}
    ctx.get_process_type.return_value = ProcessType.SERVER_JOB if server else "client"
    ctx.get_engine.return_value.get_clients.return_value = [SimpleNamespace(name="site")]
    ctx.get_identity_name.return_value = "site"
    ctx.get_engine.return_value.get_workspace.return_value.get_run_dir.return_value = str(tmp_path)
    receiver = subject.MLflowReceiver(tracking_uri="http://tracking")
    receiver.initialize(ctx)
    assert receiver.run_ids == {"site": "created-run"}
    assert receiver.experiment_id == "experiment"
    assert client.create_experiment.call_args.kwargs["artifact_location"] == (tmp_path / "artifacts").as_uri()
    assert client.create_run.call_args.kwargs["run_name"] == "site-123456-job-FLARE FL Run"
    assert client.create_run.call_args.kwargs["tags"]["job_id"] == "123456"
    assert set(receiver.buffer["site"]) == {
        AnalyticsDataType.METRICS,
        AnalyticsDataType.PARAMETERS,
        AnalyticsDataType.TAGS,
    }
    receiver.initialize(ctx)
    assert client.create_run.call_count == 1


def test_existing_experiment_is_reused_and_options_are_validated(receiver, tmp_path):
    client = MagicMock()
    client.get_experiment_by_name.return_value.experiment_id = "existing"
    assert receiver._get_or_create_experiment(client, "experiment", str(tmp_path)) == "existing"
    client.create_experiment.assert_not_called()
    with pytest.raises(ValueError):
        subject.MLflowReceiver(tracking_uri=3)
    with pytest.raises(ValueError):
        receiver._get_tags("tags", {"tags": "bad"})
    assert subject._get_job_name_from_fl_ctx(SimpleNamespace(get_prop=lambda _: None), "fallback") == "fallback"
    assert receiver.get_run_id("missing") is None and receiver.get_mlflow_client("missing") is None


@pytest.mark.parametrize("server", [False, True])
def test_initialize_rejects_missing_participants(monkeypatch, tmp_path, server):
    monkeypatch.setattr(subject.mlflow, "set_tracking_uri", MagicMock())
    ctx = MagicMock()
    ctx.get_peer_context.return_value = None
    ctx.get_engine.return_value.get_workspace.return_value.get_run_dir.return_value = str(tmp_path)
    ctx.get_process_type.return_value = ProcessType.SERVER_JOB if server else "client"
    ctx.get_engine.return_value.get_clients.return_value = []
    ctx.get_identity_name.return_value = None
    with pytest.raises(RuntimeError):
        subject.MLflowReceiver(tracking_uri="http://tracking").initialize(ctx)
