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

import json
from unittest.mock import Mock

import pytest

from nvflare.apis.fl_constant import FLContextKey, FLMetaKey, ReservedKey
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.app_common.abstract.fl_model import FLModel
from nvflare.app_common.app_constant import AppConstants
from nvflare.app_common.utils.fl_model_utils import FLModelUtils
from nvflare.app_common.widgets.metrics_artifact_writer import MetricsArtifactWriter


def context():
    ctx = FLContext()
    ctx.set_prop(AppConstants.CURRENT_ROUND, 4, private=True, sticky=True)
    return ctx


def test_selection_list_metrics_filter_invalid_entries_and_obey_record_limit():
    writer = MetricsArtifactWriter(limits={"max_metrics_per_site_per_round": 2})
    metrics = [
        None,
        {},
        {"name": 1, "value": 2},
        {"name": "\x00", "value": 2},
        {"name": "nan", "value": float("nan")},
        {"name": "loss", "value": 0.5},
        {"name": "accuracy", "value": 0.75},
        {"name": "extra", "value": 9},
    ]
    expected = [{"name": "loss", "value": 0.5}, {"name": "accuracy", "value": 0.75}]
    assert writer._normalize_metric_collection(metrics, for_aggregation=True) == expected
    result = writer._normalize_selection_info(
        {
            "best_round": 3,
            "best_metrics": metrics,
            "best_aggregated_metrics": {"loss": 0.5},
            "source": "validation",
            "metric_source": "client",
            "key_metric": {"name": "loss", "mode": "min", "mode_source": "config"},
        }
    )
    assert result["best_metrics"] == expected
    assert result["best_aggregated_metrics"] == [{"name": "loss", "value": 0.5}]
    assert result["best_metric_source"] == "validation"
    assert result["best_metric_detail_source"] == "client"
    assert result["key_metric"] == {"name": "loss", "mode": "min", "mode_source": "config"}
    assert writer._normalize_metric_collection(object()) == []
    assert writer._normalize_selection_info({}) is None


def test_site_and_weight_normalization_rejects_empty_entries_and_bounds_clients():
    writer = MetricsArtifactWriter(limits={"max_sites_per_round": 1, "max_site_metric_records_per_round": 1})
    skipped = []
    sites = writer._normalize_sites(
        [
            None,
            {"name": "empty", "metrics": {}},
            {"name": "one", "metrics": {"loss": 1, "extra": 2}, "progress_metrics": {"post": 3}},
            {"name": "two", "metrics": {"loss": 4}},
        ],
        skipped,
    )
    assert sites == [{"name": "one", "metrics": [{"name": "loss", "value": 1}]}]
    assert {record["reason"] for record in skipped} == {"too_many_metrics", "too_many_sites"}
    skipped = []
    assert writer._normalize_site_weights(
        [None, {"name": ""}, {"name": "one", "weight": 2}, {"name": "two", "weight": 3}], skipped
    ) == {"one": {"weight": 2}}
    assert skipped == [{"name": "", "reason": "too_many_sites", "site": "two"}]


@pytest.mark.parametrize(
    "limit,reason",
    [("max_sites_per_round", "too_many_sites"), ("max_site_metric_records_per_round", "too_many_metrics")],
)
def test_accepted_contributions_record_budget_exhaustion_without_growing_state(limit, reason):
    writer = MetricsArtifactWriter(write_artifacts=False, limits={limit: 1})
    ctx = context()
    for name in ["one", "two"]:
        result = FLModelUtils.to_shareable(
            FLModel(
                params={"w": 1},
                metrics={"loss": 0.5},
                meta={FLMetaKey.SITE_NAME: name, FLMetaKey.NUM_STEPS_CURRENT_ROUND: 5},
            )
        )
        ctx.set_prop(AppConstants.TRAINING_RESULT, result, private=True, sticky=False)
        writer._handle_after_contribution_accept(ctx)
    assert len(writer._round_sites[4]) == 1
    assert writer._round_sites[4][0]["weight"] == 5
    assert writer._round_skipped[4][-1]["reason"] == reason
    assert writer._round_site_metric_counts[4] == 1


def test_contribution_reads_peer_shareable_and_survives_missing_or_invalid_payloads():
    writer, ctx = MetricsArtifactWriter(write_artifacts=False), context()
    writer._handle_after_contribution_accept(ctx)
    assert writer._round_contribution_count == 0
    peer = FLContext()
    peer.set_prop(ReservedKey.IDENTITY_NAME, "peer", private=False, sticky=True)
    ctx.set_peer_context(peer)
    peer.set_prop(FLContextKey.SHAREABLE, Shareable(), private=True, sticky=False)
    writer._handle_after_contribution_accept(ctx)
    assert writer._round_contribution_count == 0
    peer.set_prop(
        FLContextKey.SHAREABLE, FLModelUtils.to_shareable(FLModel(params={"w": 1})), private=True, sticky=False
    )
    writer._handle_after_contribution_accept(ctx)
    assert writer._round_contribution_count == 1
    assert writer._round_sites == {}
    assert writer._get_site_name(FLModel(), ctx) == "peer"
    assert writer._get_current_round(FLModel(meta={"current_round": 2}), ctx) == 2
    assert writer._get_current_round(FLModel(), ctx) == 4


def test_artifact_write_failures_and_byte_limits_leave_no_invalid_json(tmp_path, monkeypatch):
    writer = MetricsArtifactWriter(limits={"max_round_record_bytes": 250})
    writer._round_file_path = str(tmp_path / "round.jsonl")
    writer._summary_file_path = str(tmp_path / "summary.json")
    record = {
        "round": 1,
        "aggregated_metrics": [{"name": "loss", "value": 1}],
        "sites": [object(), {"name": "x" * 500}],
        "skipped_metrics": [],
    }
    writer._append_round_record(FLContext(), record)
    written = json.loads((tmp_path / "round.jsonl").read_text())
    assert written["round"] == 1
    assert written["sites"] == []
    assert written["skipped_metrics"][0]["reason"] == "sites_truncated"
    assert writer._fit_json_list([object(), {"loss": 1}], 50) == [{"loss": 1}]
    writer.max_round_record_bytes = 1
    before = (tmp_path / "round.jsonl").read_bytes()
    writer._append_round_record(FLContext(), record)
    assert (tmp_path / "round.jsonl").read_bytes() == before
    writer.max_round_record_bytes = 250
    monkeypatch.setattr(writer, "_safe_json_dumps", Mock(return_value=None))
    writer._append_round_record(FLContext(), record)
    assert (tmp_path / "round.jsonl").read_bytes() == before


def test_summary_reports_many_missing_custom_rounds_and_obeys_size_budget(tmp_path):
    writer = MetricsArtifactWriter(limits={"max_summary_bytes": 1})
    writer._summary_file_path = str(tmp_path / "summary.json")
    writer._round_file_path = str(tmp_path / "round.jsonl")
    writer._custom_aggregator_no_metric_rounds = list(range(11))
    writer._has_metrics = True
    writer.log_warning = Mock()
    ctx = FLContext()
    writer._write_summary_if_needed(ctx)
    assert not (tmp_path / "summary.json").exists()
    assert "11 rounds" in writer.log_warning.call_args.args[1]
