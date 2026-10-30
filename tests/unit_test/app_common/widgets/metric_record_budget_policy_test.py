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

from nvflare.apis.fl_constant import FLMetaKey
from nvflare.apis.fl_context import FLContext
from nvflare.app_common.abstract.fl_model import FLModel
from nvflare.app_common.app_constant import AppConstants
from nvflare.app_common.utils.fl_model_utils import FLModelUtils
from nvflare.app_common.widgets import metrics_artifact_writer as subject


def test_metric_budget_partially_retains_site_and_records_omission_before_next_site():
    writer = subject.MetricsArtifactWriter(limits={"max_site_metric_records_per_round": 2})
    skipped = []
    sites = writer._normalize_sites(
        [
            {"name": "one", "metrics": {"a": 1}, "progress_metrics": {"b": 2, "c": 3}},
            {"name": "two", "metrics": {"d": 4}},
        ],
        skipped,
    )
    assert sites == [
        {"name": "one", "metrics": [{"name": "a", "value": 1}], "progress_metrics": [{"name": "b", "value": 2}]}
    ]
    assert [item["reason"] for item in skipped] == ["too_many_metrics", "too_many_metrics"]


def test_live_contributions_share_round_metric_budget_and_round_completion_flushes_state(monkeypatch):
    writer = subject.MetricsArtifactWriter(write_artifacts=False, limits={"max_site_metric_records_per_round": 3})
    context = FLContext()
    context.set_prop(AppConstants.CURRENT_ROUND, 0)
    for site in ("one", "two"):
        model = FLModel(
            params={"w": 1},
            current_round=0,
            metrics={"loss": 0.2},
            meta={FLMetaKey.SITE_NAME: site, AppConstants.PROGRESS_METRICS: {"accuracy": 0.8}},
        )
        context.set_prop(AppConstants.TRAINING_RESULT, FLModelUtils.to_shareable(model))
        writer._handle_after_contribution_accept(context)
    assert writer._round_site_metric_counts[0] == 3
    assert len(writer._round_sites[0]) == 2
    assert writer._round_sites[0][1]["metrics"] == [{"name": "loss", "value": 0.2}]
    assert "progress_metrics" not in writer._round_sites[0][1]
    append = Mock()
    monkeypatch.setattr(writer, "_append_round_record", append)
    progress = Mock()
    monkeypatch.setattr(subject, "log_progress", progress)
    writer._handle_round_done(context)
    assert writer._round_sites == writer._round_site_metric_counts == {}
    assert append.call_count == 1
    assert any("2 client updates" in call.args[1] for call in progress.call_args_list)
    assert writer._has_metrics and writer._final_round == 0


def test_unsupported_aggregation_metadata_cannot_drop_valid_metrics():
    writer, context = subject.MetricsArtifactWriter(write_artifacts=False), FLContext()
    context.set_prop(
        AppConstants.AGGREGATION_RESULT,
        FLModel(params={"w": 1}, metrics={"loss": 0.2}, meta={AppConstants.METRICS_AGGREGATION_INFO: "invalid"}),
    )
    writer._handle_after_aggregation(context)
    assert writer._final_aggregated_metrics == [{"name": "loss", "value": 0.2}]
    assert writer._get_job_name_from_meta({"unrelated": "value"}) is None
