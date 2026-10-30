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

from nvflare.apis.job_def import JobMetaKey
from nvflare.tool import cli_output
from nvflare.tool.job import job_cli as subject
from tests.timing_utils import isolate_time


def test_failed_stats_refresh_clears_stale_metrics_and_raw_payload():
    session = Mock()
    session.show_stats.side_effect = OSError("offline")
    state = {"last_stats": {"accuracy": 1}, "last_stats_raw": {"stale": True}}
    subject._refresh_monitor_stats(session, "job", state, "server", {})
    assert state == {"last_stats": None, "last_stats_raw": None}
    session.show_stats.assert_called_once_with("job", "server", None)


def test_progress_message_includes_initial_identity_and_metric_values_once(monkeypatch, capsys):
    monkeypatch.setattr(cli_output, "_output_format", "human")
    state = subject._make_monitor_state()
    state["last_stats"] = {"accuracy": 0.9}
    meta = {
        JobMetaKey.JOB_NAME.value: "model",
        JobMetaKey.SUBMIT_TIME_ISO.value: "2026-10-30T00:00:00Z",
        "status": "RUNNING",
    }
    subject._emit_monitor_progress("job", meta, state, 10, 0, 2)
    first = capsys.readouterr().out
    assert "job_id: job" in first and "name: model" in first and "submit_time:" in first
    assert "accuracy=0.9" in first and "elapsed_s: 8" in first
    subject._emit_monitor_progress("job", meta, state, 20, 0, 2)
    assert "job_id:" not in capsys.readouterr().out
    assert state["last_status"] == "RUNNING" and state["last_emit_ts"] == 20


def test_timeout_envelope_retains_last_metadata_and_metrics(monkeypatch):
    clock = isolate_time(monkeypatch, subject)
    monkeypatch.setattr(clock, "time", lambda: 30)
    event = subject._build_monitor_timeout_event(
        "job",
        10,
        0,
        2,
        {"last_meta": {"status": "RUNNING", JobMetaKey.JOB_NAME.value: "model"}, "last_stats": {"round": 3}},
    )
    assert event["status"] == "TIMEOUT" and event["job_status"] == "RUNNING"
    assert event["metrics"] == {"round": 3} and event["job_meta"]["job_name"] == "model"
    assert event["elapsed_s"] == 28 and event["terminal"] is True


def test_monitor_result_printer_renders_metrics_and_elapsed_time():
    output = Mock()
    subject._print_monitor_result_human(
        {
            "job_id": "job",
            "status": "FINISHED:ABORTED",
            "duration_s": 3,
            "job_meta": {"job_name": "model", "study": "trial"},
            "last_stats": {"accuracy": 0.75},
        },
        output,
    )
    assert [c.args[0] for c in output.call_args_list] == [
        "Job job status: ABORTED (FINISHED:ABORTED)",
        "  Duration: 3s",
        "  Name: model",
        "  Study: trial",
        "  Metrics: accuracy=0.75",
    ]


@pytest.mark.parametrize("status,expected", [(None, "UNKNOWN"), ("ABORTED", "ABORTED"), ("ABANDONED", "ABANDONED")])
def test_terminal_status_normalization_preserves_cancellation_states(status, expected):
    assert subject._normalize_monitor_event_status(status) == expected


@pytest.mark.parametrize(
    "subcommand,expected",
    [
        ("list_templates", subject.CMD_LIST_TEMPLATES),
        ("show_variables", subject.CMD_SHOW_VARIABLES),
        (None, None),
        ("unknown", None),
    ],
)
def test_job_dispatch_normalizes_legacy_names_and_rejects_missing_or_unknown_handler(monkeypatch, subcommand, expected):
    handler = Mock()
    monkeypatch.setattr(subject, "job_sub_cmd_handlers", {expected: handler} if expected else {})
    args = SimpleNamespace(job_sub_cmd=subcommand)
    if expected:
        subject.handle_job_cli_cmd(args)
        assert args.job_sub_cmd == expected
        handler.assert_called_once_with(args)
    else:
        with pytest.raises(subject.CLIUnknownCmdException):
            subject.handle_job_cli_cmd(args)
