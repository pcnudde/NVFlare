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

from nvflare import _version as subject


def test_archive_keyword_parser_reads_quoted_values_without_executing_module(tmp_path):
    source = tmp_path / "version.py"
    source.write_text(
        'raise AssertionError("must not execute")\n  git_refnames = "(tag: v3.0)"\ngit_full = "abc1234"\ngit_date = "2026-10-30 00:00:00 +0000"\n'
    )
    keywords = subject.git_get_keywords(str(source))
    assert keywords == {"refnames": "(tag: v3.0)", "full": "abc1234", "date": "2026-10-30 00:00:00 +0000"}
    assert subject.git_versions_from_keywords(keywords, "v", False)["version"] == "3.0"


def test_missing_and_unquoted_archive_keywords_are_not_treated_as_expanded_metadata(tmp_path):
    assert subject.git_get_keywords(str(tmp_path / "missing")) == {}
    source = tmp_path / "version.py"
    source.write_text("git_refnames = None\ngit_full = None\ngit_date = None\n")
    assert subject.git_get_keywords(str(source)) == {}
