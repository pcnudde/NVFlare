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

import io
import json
from unittest.mock import Mock

import pytest

from nvflare.fuel.hci.tools import authz_preview as subject
from nvflare.fuel.sec.authz import Person, parse_policy_config


def policy():
    result, error = parse_policy_config(
        {"format_version": "1.0", "permissions": {"lead": {"submit_job": "any"}, "member": {"submit_job": "none"}}}, {}
    )
    assert not error
    return result


def test_preview_tables_and_policy_config_describe_real_authorization_rules():
    commander = subject.Commander(policy())
    commander.stdout = io.StringIO()
    commander.do_show_rights("")
    commander.do_show_roles("")
    commander.do_show_role_rights("")
    output = commander.stdout.getvalue()
    assert "submit_job" in output
    assert "lead" in output
    assert "member" in output
    commander.stdout = io.StringIO()
    commander.do_show_config("")
    assert json.loads(commander.stdout.getvalue())["format_version"] == "1.0"
    assert commander.do_bye("") is True
    assert commander.emptyline() is None


@pytest.mark.parametrize("role,expected", [("lead", "True"), ("member", "False")])
def test_preview_evaluates_right_for_normalized_user_and_submitter_context(role, expected):
    commander = subject.Commander(policy())
    commander.stdout = io.StringIO()
    commander.do_eval_right(f"org submit_job user:org:{role} submitter:org:member")
    assert commander.stdout.getvalue().strip() == expected
    assert isinstance(commander._parse_person("name:org:role"), Person)


@pytest.mark.parametrize(
    "arguments,expected",
    [
        ("", "Usage:"),
        ("org submit_job bad-user", "bad user spec"),
        ("org submit_job user:org:lead bad-submitter", "bad submitter spec"),
    ],
)
def test_preview_reports_usage_and_malformed_person_spec_without_evaluating_policy(arguments, expected):
    fake_policy = Mock()
    commander = subject.Commander(fake_policy)
    commander.stdout = io.StringIO()
    commander.do_eval_right(arguments)
    assert expected in commander.stdout.getvalue()
    fake_policy.evaluate.assert_not_called()


@pytest.mark.parametrize(
    "result,error,expected", [(None, "", "undetermined"), (None, "policy error", "Error: policy error")]
)
def test_preview_distinguishes_indeterminate_policy_from_policy_failure(result, error, expected):
    fake_policy = Mock()
    fake_policy.evaluate.return_value = (result, error)
    commander = subject.Commander(fake_policy)
    commander.stdout = io.StringIO()
    commander.do_eval_right("org right user:org:lead")
    assert commander.stdout.getvalue().strip() == expected
    assert fake_policy.evaluate.call_args.kwargs["site_org"] == "org"


def test_policy_file_loading_rejects_invalid_policy_and_accepts_valid_configuration(tmp_path, capsys):
    path = tmp_path / "policy.json"
    path.write_text(json.dumps({"format_version": "1.0", "permissions": {"lead": "any"}}))
    assert subject.load_policy(str(path)).get_roles() == ["lead"]
    path.write_text("{}")
    with pytest.raises(SystemExit) as raised:
        subject.load_policy(str(path))
    assert raised.value.code == 1
    assert "Policy config error" in capsys.readouterr().out
