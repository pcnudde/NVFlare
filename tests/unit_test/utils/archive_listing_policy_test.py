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

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
from zipfile import ZipFile

from nvflare.utils import zip_utils as subject


def test_directory_listing_preserves_exclusion_as_single_process_argument(tmp_path, monkeypatch):
    run = Mock()
    monkeypatch.setattr(subject, "subprocess", SimpleNamespace(run=run))
    subject.print_tree(tmp_path, "__pycache__|*.pyc")
    run.assert_called_once_with(["tree", "-a", "--dirsfirst", "-I", "__pycache__|*.pyc", str(tmp_path)], check=False)


def test_missing_tree_program_falls_back_to_sorted_recursive_listing(tmp_path, monkeypatch, capsys):
    (tmp_path / "nested").mkdir()
    (tmp_path / "nested" / "file.txt").write_text("value")
    (tmp_path / "a.txt").write_text("value")
    monkeypatch.setattr(subject, "subprocess", SimpleNamespace(run=Mock(side_effect=FileNotFoundError())))
    subject.print_tree(tmp_path)
    text = capsys.readouterr().out
    assert "'tree' command not found" in text
    assert text.index("a.txt") < text.index("nested") < text.index("file.txt")
    assert "    ├── file.txt" in text


def test_zip_listing_owns_extracted_directory_and_passes_exclusion(tmp_path, monkeypatch):
    archive = tmp_path / "job.zip"
    with ZipFile(archive, "w") as output:
        output.writestr("config/settings.json", '{"value": 1}')
    extracted = []

    def display(path, excluded):
        assert excluded == "*.pyc"
        assert (path / "config" / "settings.json").read_text() == '{"value": 1}'
        extracted.append(Path(path))

    monkeypatch.setattr(subject, "print_tree", display)
    subject.print_zip_tree(archive, "*.pyc")
    assert len(extracted) == 1 and not extracted[0].exists()
