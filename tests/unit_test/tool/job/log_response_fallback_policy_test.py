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

import datetime

from nvflare.tool.job import job_cli as subject


def test_log_response_fallback_accepts_only_mapping_payloads_and_preserves_primary_text():
    assert subject._select_job_logs_for_output({"logs": []}, True) == {}
    assert subject._job_log_result_needs_fallback([], "all")
    assert subject._job_log_result_needs_fallback({"logs": []}, "all")
    assert subject._merge_job_log_fallback_result(None, None) == {"logs": {}}
    fallback = {"logs": {"site": "recovered"}}
    assert subject._merge_job_log_fallback_result({"logs": [], "unavailable": []}, fallback) == fallback
    primary = {"logs": {"site": "canonical"}, "unavailable": {"other": "offline"}}
    result = subject._merge_job_log_fallback_result(primary, {"logs": {"site": "replacement"}})
    assert result["logs"]["site"] == "canonical" and result["unavailable"] == {"other": "offline"}
    assert primary == {"logs": {"site": "canonical"}, "unavailable": {"other": "offline"}}


def test_nontext_site_logs_remain_available_without_inventing_line_counts():
    bounded, sites, truncated, since_applied = subject._apply_job_log_bounds({"site": None}, 10, None, 100)
    assert bounded == {"site": None}
    assert sites == {"site": {"available": True, "lines": None, "logs_truncated": False}}
    assert not truncated and not since_applied


def test_empty_log_filters_and_zero_tail_have_explicit_empty_contracts():
    stamp = datetime.datetime(2030, 1, 1)
    assert subject._filter_log_since("", stamp) == ("", False, False)
    assert subject._filter_log_json_since("", stamp) == ("", False, False)
    assert subject._tail_log_text("line\n", 0) == ("", True)
    assert subject._truncate_log_bytes("line", None) == ("line", False)


def test_custom_monitor_metric_names_are_recognized_without_nested_stats():
    aliases = subject._build_monitor_key_aliases(["custom"])
    assert aliases["custom"] == ["custom"]
    assert subject._extract_monitor_metrics(None, aliases) == {}
    assert subject._summarize_monitor_meta(None, object()) == {}
