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
import json
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from nvflare.fuel.flare_api.api_spec import NoConnection
from nvflare.tool import cli_output
from nvflare.tool.job import job_cli as subject


@pytest.fixture(autouse=True)
def command_output_and_parsers(monkeypatch):
    monkeypatch.setattr(cli_output, "_output_format", "json")
    monkeypatch.setattr(
        subject,
        "job_sub_cmd_parser",
        {
            key: None
            for key in (
                subject.CMD_JOB_LIST,
                subject.CMD_JOB_META,
                subject.CMD_JOB_STATS,
                subject.CMD_JOB_LOGS,
                subject.CMD_JOB_WAIT,
            )
        },
    )


@pytest.mark.parametrize("command", ["list", "meta", "stats", "logs", "wait"])
def test_lifecycle_command_connection_failure_has_exact_json_code_and_no_partial_success(monkeypatch, capsys, command):
    create = Mock(side_effect=NoConnection("admin offline"))
    monkeypatch.setattr(subject, "_job_session_for_args", create)
    with pytest.raises(SystemExit) as error:
        getattr(subject, "cmd_job_" + command)(SimpleNamespace(job_id="job", study="trial"))
    assert error.value.code == 2
    output = json.loads(capsys.readouterr().out)
    assert output["error_code"] == "CONNECTION_FAILED" and "admin offline" in output["message"]
    assert output["status"] == "error"
    create.assert_called_once()


@pytest.mark.parametrize("command", ["logs", "wait"])
def test_lifecycle_command_unexpected_backend_exception_is_reported_without_success(monkeypatch, capsys, command):
    monkeypatch.setattr(subject, "_job_session_for_args", Mock(side_effect=ValueError("bad protocol response")))
    with pytest.raises(SystemExit) as error:
        getattr(subject, "cmd_job_" + command)(SimpleNamespace(job_id="job", study="trial"))
    assert error.value.code == 5
    result = json.loads(capsys.readouterr().out)
    assert result["error_code"] == "INTERNAL_ERROR" and "bad protocol response" in result["message"]


@pytest.mark.parametrize("site,target,targets", [("server", "server", None), ("site-1", "client", ["site-1"])])
def test_job_stats_routes_server_and_named_site_without_rewriting_payload(monkeypatch, capsys, site, target, targets):
    session = Mock()
    session.show_stats.return_value = {"round": 3}
    monkeypatch.setattr(subject, "_job_session_for_args", lambda args, study: nullcontext(session))
    subject.cmd_job_stats(SimpleNamespace(job_id="job", study="trial", site=site))
    session.show_stats.assert_called_once_with("job", target, targets)
    assert json.loads(capsys.readouterr().out)["data"] == {"job_id": "job", "stats": {"round": 3}}


def test_untimestamped_log_content_is_preserved_without_claiming_since_filter_was_applied():
    since = datetime.datetime(2030, 1, 1)
    text = "traceback without timestamp\n"
    assert subject._filter_log_since(text, since) == (text, False, False)
    assert subject._filter_log_json_since(text, since) == (text, False, False)
    assert subject._parse_log_line_timestamp("2030-99-99 10:00:00 invalid date") is None
    assert subject._looks_like_json_log("\n   \n") is False
    assert subject._local_download_path(None) is None
    assert list(subject._iter_files_under(None)) == []
