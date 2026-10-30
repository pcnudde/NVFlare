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

from subprocess import TimeoutExpired
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from nvflare.tool.package_checker import package_checker as subject
from nvflare.tool.package_checker.check_rule import CHECK_PASSED, CheckResult, CheckRule
from nvflare.tool.package_checker.package_checker import CheckStatus, PackageChecker


class Checker(PackageChecker):
    def init_rules(self, package_path):
        self.rules = []

    def should_be_checked(self):
        return True

    def get_dry_run_command(self):
        return "python -m test_only_command"


class Rule(CheckRule):
    def __init__(self, name, result, required=True):
        super().__init__(name, required)
        self.result = result
        self.inputs = []

    def __call__(self, package_path, data):
        self.inputs.append((package_path, data))
        return self.result


@pytest.fixture
def checker(tmp_path):
    instance = Checker()
    instance.init(str(tmp_path))
    return instance


def test_missing_package_fails_without_initializing_rules(tmp_path):
    checker = Checker()
    checker.init_rules = Mock()
    with pytest.raises(RuntimeError, match="does not exist"):
        checker.init(str(tmp_path / "missing"))
    checker.init_rules.assert_not_called()


@pytest.mark.parametrize("required", [False, True])
def test_required_rule_failure_blocks_dry_run_but_optional_rule_failure_is_reported(checker, required):
    rule = Rule("permission", CheckResult("denied", "fix permissions"), required)
    checker.rules = [rule]
    checker.check_dry_run = Mock(return_value=CheckStatus.PASS_WITH_CLEANUP)
    assert checker.check() == (CheckStatus.FAIL if required else CheckStatus.PASS_WITH_CLEANUP)
    assert checker.check_dry_run.called is not required
    assert rule.inputs == [(checker.package_path, None)]
    assert checker.report[checker.package_path] == [("permission", "denied", "fix permissions")]
    assert checker.check_len >= len("permission") and checker.fix_len >= len("fix permissions")


def test_ordered_rules_forward_prior_data_and_stop_at_first_required_failure(checker):
    first = Rule("first", CheckResult(CHECK_PASSED, "N/A", data={"token": 3}))
    second = Rule("second", CheckResult("invalid token", "retry"))
    never = Rule("never", CheckResult(CHECK_PASSED))
    independent = Rule("separate", CheckResult(CHECK_PASSED))
    checker.rules = [[first, second, never], independent]
    checker.check_dry_run = Mock()
    assert checker.check() == CheckStatus.FAIL
    assert first.inputs == [(checker.package_path, None)]
    assert second.inputs == [(checker.package_path, {"token": 3})]
    assert never.inputs == [] and independent.inputs == [(checker.package_path, None)]
    checker.check_dry_run.assert_not_called()
    assert [row[0] for row in checker.report[checker.package_path]] == ["first", "second", "separate"]


def test_unexpected_rule_error_is_reported_as_package_error(checker):
    rule = Mock(spec=CheckRule)
    rule.side_effect = ValueError("invalid layout")
    checker.rules = [rule]
    checker.check_dry_run = Mock()
    assert checker.check() == CheckStatus.FAIL
    name, problem, fix = checker.report[checker.package_path][0]
    assert name == "Package Error" and "invalid layout" in problem and "new package" in fix
    checker.check_dry_run.assert_not_called()


@pytest.mark.parametrize(
    "mode,status",
    [
        ("success", CheckStatus.PASS_WITH_CLEANUP),
        ("nonzero", CheckStatus.FAIL_WITH_CLEANUP),
        ("timeout", CheckStatus.PASS_WITH_CLEANUP),
        ("spawn_error", CheckStatus.FAIL),
    ],
)
@pytest.mark.parametrize("input_data", [None, "answers\n"])
def test_dry_run_status_and_cleanup_follow_mocked_process_outcome(checker, monkeypatch, mode, status, input_data):
    process = Mock(returncode=1 if mode == "nonzero" else 0, pid=71)
    process.communicate.return_value = ("process output", None)
    if mode == "timeout":
        process.communicate.side_effect = TimeoutExpired("command", 5)
    launch = Mock(return_value=process)
    if mode == "spawn_error":
        launch.side_effect = OSError("cannot launch")
    monkeypatch.setattr(subject, "run_command_in_subprocess", launch)
    kill = Mock()
    monkeypatch.setattr(subject, "os", SimpleNamespace(killpg=kill))
    checker.get_dry_run_inputs = Mock(return_value=input_data)
    assert checker.check_dry_run() == status
    launch.assert_called_once_with("python -m test_only_command")
    if mode != "spawn_error":
        expected = {"timeout": 5}
        if input_data is not None:
            expected["input"] = input_data
        process.communicate.assert_called_once_with(**expected)
    assert kill.called is (mode == "timeout")
    if mode == "timeout":
        kill.assert_called_once_with(71, subject.signal.SIGTERM)
    report = checker.report[checker.package_path][0]
    assert report[0] == "Check dry run"
    assert (report[1] == CHECK_PASSED) is (mode in ("success", "timeout"))


def test_report_printing_wraps_long_problems_and_handles_empty_package_results(checker, capsys):
    checker.problem_len = 5
    checker.add_report("storage", "123456789", "permissions")
    checker.report["empty"] = []
    checker.print_report()
    text = capsys.readouterr().out
    assert "storage" in text and "12345" in text and "6789" in text and "permissions" in text
    assert "Checking Package: empty" in text and "Passed" in text


def test_stop_uses_controlled_subprocess_boundary_without_executing_kill_command(checker, monkeypatch):
    process = Mock()
    process.communicate.return_value = ("stopped", "")
    launch = Mock(return_value=process)
    monkeypatch.setattr(subject, "run_command_in_subprocess", launch)
    checker.stop_dry_run()
    assert launch.call_args.args[0] == "pkill -9 -f 'python -m test_only_command'"
    process.communicate.assert_called_once_with()
