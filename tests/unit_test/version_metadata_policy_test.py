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

import errno
from unittest.mock import Mock

import pytest

from nvflare import _version as subject


def pieces(tag="2.0", distance=3, dirty=False, branch="master"):
    return {
        "closest-tag": tag,
        "distance": distance,
        "dirty": dirty,
        "branch": branch,
        "short": "abc1234",
        "long": "abc1234ffff",
        "error": None,
        "date": "2026-10-30T12:00:00+0000",
    }


@pytest.mark.parametrize(
    "style, expected",
    [
        ("pep440", "2.0+3.gabc1234"),
        ("pep440-branch", "2.0.dev0+3.gabc1234"),
        ("pep440-pre", "2.0.post0.dev3"),
        ("pep440-post", "2.0.post3+gabc1234"),
        ("pep440-post-branch", "2.0.post3.dev0+gabc1234"),
        ("pep440-old", "2.0.post3"),
        ("git-describe", "2.0-3-gabc1234"),
        ("git-describe-long", "2.0-3-gabc1234"),
        (None, "2.0+3.gabc1234"),
        ("default", "2.0+3.gabc1234"),
    ],
)
def test_version_style_preserves_commit_and_date_metadata(style, expected):
    result = subject.render(pieces(branch="feature"), style)
    assert result == {
        "version": expected,
        "full-revisionid": "abc1234ffff",
        "dirty": False,
        "error": None,
        "date": "2026-10-30T12:00:00+0000",
    }


@pytest.mark.parametrize(
    "style, expected",
    [
        ("pep440", "0+untagged.3.gabc1234.dirty"),
        ("pep440-branch", "0.dev0+untagged.3.gabc1234.dirty"),
        ("pep440-pre", "0.post0.dev3"),
        ("pep440-post", "0.post3.dev0+gabc1234"),
        ("pep440-post-branch", "0.post3.dev0+gabc1234.dirty"),
        ("pep440-old", "0.post3.dev0"),
        ("git-describe", "abc1234-dirty"),
        ("git-describe-long", "abc1234-dirty"),
    ],
)
def test_untagged_dirty_builds_have_explicit_development_identity(style, expected):
    assert subject.render(pieces(tag=None, dirty=True, branch="feature"), style)["version"] == expected


@pytest.mark.parametrize(
    "style",
    ["pep440", "pep440-branch", "pep440-pre", "pep440-post", "pep440-post-branch", "pep440-old", "git-describe"],
)
def test_clean_tag_is_released_without_development_suffix(style):
    assert subject.render(pieces(distance=0), style)["version"] == "2.0"


@pytest.mark.parametrize(
    "style, expected",
    [
        ("pep440", "2.0+3.gabc1234.dirty"),
        ("pep440-branch", "2.0+3.gabc1234.dirty"),
        ("pep440-post-branch", "2.0.post3+gabc1234.dirty"),
        ("pep440-old", "2.0.post3.dev0"),
    ],
)
def test_dirty_master_builds_keep_dirty_marker(style, expected):
    assert subject.render(pieces(dirty=True), style)["version"] == expected


def test_post_release_increment_and_existing_local_version_separator():
    assert subject.render(pieces(tag="2.0.post4"), "pep440-pre")["version"] == "2.0.post5.dev3"
    assert subject.render(pieces(tag="2.0+cuda"), "pep440")["version"] == "2.0+cuda.3.gabc1234"
    assert subject.pep440_split_post("2.0.post") == ("2.0", 0)
    assert subject.render(pieces(distance=0), "git-describe-long")["version"] == "2.0-0-gabc1234"


def test_unrecognized_style_and_parse_failure_are_explicit():
    with pytest.raises(ValueError, match="unknown style"):
        subject.render(pieces(), "unsupported")
    metadata = pieces()
    metadata["error"] = "invalid git output"
    assert subject.render(metadata, "pep440") == {
        "version": "unknown",
        "full-revisionid": "abc1234ffff",
        "dirty": None,
        "error": "invalid git output",
        "date": None,
    }


@pytest.mark.parametrize(
    "refs, expected",
    [
        ("(HEAD, tag: v2.0, tag: v2.0rc1)", "2.0"),
        ("(master, v1.2)", "1.2"),
        ("(tag: v, tag: vrelease, tag: unrelated3)", "0+unknown"),
    ],
)
def test_archive_metadata_prefers_release_tags_and_normalizes_signed_date(refs, expected):
    result = subject.git_versions_from_keywords(
        {"refnames": refs, "full": " abc1234 ", "date": "GPG signature\n2026-10-30 12:00:00 +0000"}, "v", True
    )
    assert result["version"] == expected
    assert result["full-revisionid"] == "abc1234"
    assert result["dirty"] is False
    assert result["date"] == (None if expected == "0+unknown" else "2026-10-30T12:00:00+0000")


@pytest.mark.parametrize("keywords", [{}, {"refnames": "$Format:%d$"}])
def test_missing_or_unexpanded_archive_keywords_request_another_method(keywords):
    with pytest.raises(subject.NotThisMethod):
        subject.git_versions_from_keywords(keywords, "v", True)


def test_release_parent_directory_search_is_bounded_to_three_levels(tmp_path):
    root = tmp_path / "nvflare-3.0" / "source" / "nvflare"
    assert subject.versions_from_parentdir("nvflare-", str(root), False)["version"] == "3.0"
    with pytest.raises(subject.NotThisMethod):
        subject.versions_from_parentdir("nvflare-", str(root / "too-deep"), True)


@pytest.mark.parametrize(
    "description, branch, extra, expected_tag, expected_distance, expected_branch",
    [
        ("v2.0-3-gabc1234-dirty", "feature", [], "2.0", 3, "feature"),
        ("abc1234", "master", [("12", 0)], None, 12, "master"),
        ("v2.0-0-gabc1234", "HEAD", [("* (HEAD detached)\n  feature\n  master", 0)], "2.0", 0, "master"),
        ("v2.0-0-gabc1234", "HEAD", [("* (HEAD detached)", 0)], "2.0", 0, None),
        ("v2.0-0-gabc1234", "HEAD", [("* (HEAD detached)\n  feature", 0)], "2.0", 0, "feature"),
    ],
)
def test_git_description_parses_tag_distance_dirty_and_detached_branch(
    description, branch, extra, expected_tag, expected_distance, expected_branch
):
    runner = Mock(
        side_effect=[(".git", 0), (description, 0), ("abc1234ffff", 0), (branch, 0)]
        + extra
        + [("GPG signature\n2026-10-30 12:00:00 +0000", 0)]
    )
    result = subject.git_pieces_from_vcs("v", "/unused", False, runner)
    assert result["closest-tag"] == expected_tag
    assert result["distance"] == expected_distance
    assert result["branch"] == expected_branch
    assert result["dirty"] == description.endswith("-dirty")
    assert result["date"] == "2026-10-30T12:00:00+0000"
    runner.assert_any_call(["git"], ["rev-parse", "--git-dir"], cwd="/unused", hide_stderr=True)


@pytest.mark.parametrize(
    "responses",
    [
        [(None, 1)],
        [(".git", 0), (None, 1)],
        [(".git", 0), ("v2.0-0-gabc1234", 0), (None, 1)],
        [(".git", 0), ("v2.0-0-gabc1234", 0), ("hash", 0), (None, 1)],
        [(".git", 0), ("v2.0-0-gabc1234", 0), ("hash", 0), ("HEAD", 0), (None, 1)],
    ],
)
def test_git_command_failure_requests_another_version_method(responses):
    with pytest.raises(subject.NotThisMethod):
        subject.git_pieces_from_vcs("v", "/unused", True, Mock(side_effect=responses))


@pytest.mark.parametrize(
    "description, message",
    [("broken-description", "unable to parse"), ("release2.0-3-gabc1234", "doesn't start with prefix")],
)
def test_unexpected_git_output_reports_error_instead_of_wrong_release(description, message):
    runner = Mock(side_effect=[(".git", 0), (description, 0), ("abc1234ffff", 0), ("master", 0)])
    assert message in subject.git_pieces_from_vcs("v", "/unused", True, runner)["error"]


def test_command_fallback_decodes_output_and_only_removes_empty_worktree_head(tmp_path, monkeypatch):
    head = tmp_path / "HEAD"
    head.write_bytes(b"")
    process = Mock(returncode=0)
    process.communicate.return_value = (b"  2.0\n", b"")
    popen = Mock(side_effect=[OSError(errno.ENOENT, "not installed"), process])
    monkeypatch.setattr(subject.subprocess, "Popen", popen)
    assert subject.run_command(["missing", "git"], ["describe"], str(tmp_path), hide_stderr=True) == ("2.0", 0)
    assert not head.exists()
    assert popen.call_args.kwargs["shell"] is False
    head.write_text("do not remove")
    popen.side_effect = None
    popen.return_value = process
    process.returncode = 2
    assert subject.run_command(["git"], [], str(tmp_path), verbose=True) == (None, 2)
    assert head.read_text() == "do not remove"


@pytest.mark.parametrize("error", [errno.ENOENT, errno.EACCES])
def test_unavailable_command_returns_unknown_without_executing_shell(monkeypatch, error):
    monkeypatch.setattr(subject.subprocess, "Popen", Mock(side_effect=OSError(error, "blocked")))
    assert subject.run_command(["git"], [], verbose=True) == (None, None)


def test_version_discovery_falls_back_from_git_to_release_directory(monkeypatch):
    monkeypatch.setattr(subject, "git_versions_from_keywords", Mock(side_effect=subject.NotThisMethod()))
    monkeypatch.setattr(subject, "git_pieces_from_vcs", Mock(side_effect=subject.NotThisMethod()))
    monkeypatch.setattr(subject, "__file__", "/tmp/nvflare-4.0/nvflare/_version.py")
    config = subject.get_config()
    config.parentdir_prefix = "nvflare-"
    monkeypatch.setattr(subject, "get_config", lambda: config)
    assert subject.get_versions()["version"] == "4.0"
    monkeypatch.setattr(subject, "__file__", "/tmp/unversioned/nvflare/_version.py")
    assert subject.get_versions()["error"] == "unable to compute version"
    monkeypatch.delattr(subject, "__file__")
    assert subject.get_versions()["error"] == "unable to find root of source tree"
