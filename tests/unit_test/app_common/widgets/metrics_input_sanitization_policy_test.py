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

from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.app_common.abstract.fl_model import FLModel
from nvflare.app_common.app_constant import AppConstants
from nvflare.app_common.widgets import metrics_artifact_writer as subject


def test_metric_normalization_rejects_empty_names_and_oversized_strings_with_bounded_findings():
    writer = subject.MetricsArtifactWriter(limits={"max_string_value_length": 4, "max_skipped_metrics_per_round": 2})
    skipped = []
    assert writer._normalize_metrics({"\x00": 1, "verbose": "too long", "ok": 2}, "site", skipped) == [
        {"name": "ok", "value": 2}
    ]
    assert [record["reason"] for record in skipped] == ["empty_metric_name", "string_too_long"]
    writer._add_skipped(skipped, "site", "extra", "full")
    assert len(skipped) == 2
    target = [{"reason": "first"}]
    writer._merge_skipped(target, [{"reason": "second"}, {"reason": "third"}])
    assert target == [{"reason": "first"}, {"reason": "second"}]


def test_json_sanitization_drops_invalid_children_without_exposing_non_string_keys():
    writer = subject.MetricsArtifactWriter()
    actual = writer._sanitize_json_object(
        {1: "secret", "list": [1, object(), "a\x00b", float("inf")], "nested": {"good": True}}
    )
    assert actual == {"list": [1, "ab"], "nested": {"good": True}}
    assert writer._sanitize_name(42) == ""
    assert writer._normalize_key_metric({"name": None}) is None
    assert writer._normalize_selection_info([]) is None


def test_malformed_shareable_and_untyped_values_do_not_become_metrics_models():
    writer = subject.MetricsArtifactWriter()
    assert writer._to_fl_model(Shareable({"bad": "DXO"})) is None
    assert writer._to_fl_model(object()) is None
    assert writer._get_current_round(FLModel(params={}), FLContext()) is None
    assert writer._get_job_name_from_meta([]) is None
    with pytest.raises(TypeError, match="must be str"):
        writer._validate_file_name(42, "summary")


def test_progress_site_display_emits_one_omission_notice_after_row_limit(monkeypatch):
    writer = subject.MetricsArtifactWriter(write_artifacts=False)
    monkeypatch.setattr(subject, "MAX_PROGRESS_CLIENT_ROWS", 1)
    progress = Mock()
    monkeypatch.setattr(subject, "log_progress", progress)
    writer._log_site_metric_phase(
        "Metrics", [{"name": str(i), "metrics": [{"name": "loss", "value": i}]} for i in range(3)], "metrics"
    )
    lines = [c.args[1] for c in progress.call_args_list]
    assert len(lines) == 3
    assert sum("omitted" in line for line in lines) == 1
    writer._handle_after_aggregation(FLContext())
    writer._handle_global_best_model_available(FLContext())
    assert writer._best_selection is None


def test_rejected_or_missing_contribution_does_not_create_site_metric_records():
    writer = subject.MetricsArtifactWriter()
    ctx = FLContext()
    writer._handle_after_contribution_accept(ctx)
    ctx.set_prop(AppConstants.AGGREGATION_ACCEPTED, False)
    writer._handle_after_contribution_accept(ctx)
    assert writer._round_sites == {}
