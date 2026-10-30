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

import pytest

from nvflare.tool.job import job_cli as subject


@pytest.mark.parametrize(
    "meta,expected",
    [
        ({}, "-"),
        ({"submitter_name": "alice"}, "alice"),
        ({"submitter_name": "alice", "submitter_role": "lead"}, "alice (lead)"),
        ({"submitter_name": "alice", "submitter_org": "org"}, "alice (org)"),
        ({"submitter_name": "alice", "submitter_org": "org", "submitter_role": "lead"}, "alice (lead @ org)"),
    ],
)
def test_submitter_display_keeps_available_identity_fields(meta, expected):
    assert subject._format_submitter(meta) == expected


@pytest.mark.parametrize(
    "value,expected",
    [
        (None, None),
        ("", None),
        (3661.2, "1h 1m 1s"),
        ("00:02:03.8", "2m 3s"),
        ("12.8", "13s"),
        ("invalid", "invalid"),
        (0, "0s"),
    ],
)
def test_job_duration_display_accepts_stored_numeric_and_hms_formats(value, expected):
    assert subject._format_job_meta_duration(value) == expected


@pytest.mark.parametrize(
    "value,expected",
    [
        ("N/A", None),
        (None, None),
        ([], None),
        (1, 1.0),
        ("01:02:03.5", 3723.5),
        ("02:03.5", 123.5),
        ("3.25", 3.25),
        ("1:2:3:4", None),
        ("bad", None),
    ],
)
def test_monitor_duration_normalization_preserves_fractional_seconds(value, expected):
    assert subject._parse_monitor_duration_seconds(value) == expected


def test_json_logs_render_records_while_preserving_unstructured_lines_and_filtering_timestamps():
    text = '\n{"timestamp":"2026-10-30T00:00:00", "name":"trainer", "level":"INFO", "msg":"old"}\n{"timestamp":"bad", "time":"2026-10-30T00:01:00", "message":"new"}\nnot-json\n[1,2]\n'
    rendered = subject._render_log_json_as_text(text)
    assert "trainer - INFO - old" in rendered
    assert "not-json\n[1,2]\n" in rendered
    assert subject._render_log_json_as_text("\n") == ""
    timestamp = subject._parse_log_json_line_timestamp('{"asctime":"bad", "time":"2026-10-30T00:01:00"}')
    assert timestamp == datetime.datetime(2026, 10, 30, 0, 1)
    for line in ["bad", "[]", "{}"]:
        assert subject._parse_log_json_line_timestamp(line) is None


@pytest.mark.parametrize(
    "relative,expected",
    [
        ("server/log.txt", None),
        ("app_server/log.txt", None),
        ("log.txt", None),
        ("app_site-one/log.txt", "site-one"),
        ("logs/site-two/log.txt", "site-two"),
        ("workspace/log.txt", None),
        ("custom-client/log.txt", "custom-client"),
    ],
)
def test_downloaded_log_site_identity_is_derived_from_client_directory(tmp_path, relative, expected):
    assert subject._site_name_from_log_path(str(tmp_path), str(tmp_path / relative)) == expected
