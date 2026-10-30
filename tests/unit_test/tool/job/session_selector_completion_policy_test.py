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
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from nvflare.tool import cli_session
from nvflare.tool.job import job_cli as subject


@pytest.mark.parametrize("explicit", ["username", "directory"])
def test_partial_session_selectors_fill_only_missing_value(monkeypatch, explicit):
    resolve = Mock(return_value=("resolved-user", "/resolved/kit"))
    monkeypatch.setattr(cli_session, "resolve_admin_user_and_dir_for_args", resolve)
    create = Mock(return_value=object())
    monkeypatch.setattr(subject, "new_cli_session", create)
    args = SimpleNamespace()
    kwargs = {"username": "explicit-user"} if explicit == "username" else {"admin_user_dir": "/explicit/kit"}
    assert subject._get_session(args, study="study", **kwargs) is create.return_value
    resolve.assert_called_once_with(args)
    assert create.call_args.kwargs["username"] == ("explicit-user" if explicit == "username" else "resolved-user")
    assert create.call_args.kwargs["startup_kit_location"] == (
        "/explicit/kit" if explicit == "directory" else "/resolved/kit"
    )
    assert create.call_args.kwargs["study"] == "study"


@pytest.mark.parametrize(
    "value,expected", [(None, "-"), (True, "yes"), (False, "no"), ({"w": 2}, '{"w": 2}'), ([1], "[1]")]
)
def test_metadata_presentation_preserves_boolean_and_structured_values(value, expected):
    assert subject._format_job_meta_value(value) == expected


def test_metadata_and_log_time_fallbacks_are_timezone_explicit():
    assert subject._format_deploy_map({"app": "site"}) == "app -> site"
    assert subject._format_deploy_map([]) == "-"
    assert subject._format_job_meta_timestamp("invalid") == "invalid"
    assert subject._format_job_meta_timestamp(None) is None
    assert subject._normalize_log_timestamp("2030-01-02T03:00:00+03:00") == datetime.datetime(2030, 1, 2)
    assert subject._parse_monitor_start_ts({"start": 7}, "start", "submit") == 7.0
    with pytest.raises(ValueError, match="non-empty"):
        subject._normalize_log_timestamp("")
    with pytest.raises(ValueError, match="invalid"):
        subject._normalize_log_timestamp("invalid")
