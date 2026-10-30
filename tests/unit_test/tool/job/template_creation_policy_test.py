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

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from pyhocon import ConfigFactory

from nvflare.tool.job import job_cli as subject


def make_template(root, name="training"):
    template = root / name
    config = template / "app" / "config"
    config.mkdir(parents=True)
    (config / "config_fed_server.conf").write_text("{workflows: []}", encoding="utf-8")
    (template / "info.conf").write_text(
        'description="Averaging models"\ncontroller_type="fedavg"\nexecution_api_type="client"', encoding="utf-8"
    )
    return template


def test_template_discovery_and_variable_tables_show_available_metadata_and_component_values(
    tmp_path, monkeypatch, capsys
):
    template = make_template(tmp_path)
    (tmp_path / "ignored.txt").write_text("ignored")
    index = subject.build_job_template_indices(str(tmp_path))
    assert index.get("templates.training.description") == "Averaging models"
    subject.display_available_templates(index)
    subject.display_template_variables(
        "/jobs/example",
        {
            "custom_app": {
                "/tmp/config_fed_server.conf": {
                    "rounds": SimpleNamespace(value=5, component_name="trainer"),
                    "rate": SimpleNamespace(value=0.1, component_name=None),
                }
            }
        },
    )
    output = capsys.readouterr().out
    for expected in ["training", "Averaging models", "fedavg", "client", "custom_app", "rounds", "trainer", "0.1"]:
        assert expected in output
    monkeypatch.setattr(subject, "find_job_templates_location", Mock(return_value=str(tmp_path)))
    assert subject.get_src_template_by_name(SimpleNamespace(template="training")) == str(template)
    assert subject.get_src_template(SimpleNamespace(template=str(template))) == str(template)
    assert subject.get_src_template(SimpleNamespace(template=str(tmp_path))) is None
    with pytest.raises(ValueError, match="Invalid template name"):
        subject.get_src_template_by_name(SimpleNamespace(template="absent"))


@pytest.mark.parametrize("force", [False, True])
def test_prepare_job_folder_preserves_existing_app_unless_force_is_selected(tmp_path, force, capsys):
    job = tmp_path / "job"
    (job / "app").mkdir(parents=True)
    (job / "app" / "old.py").write_text("old")
    (job / "notes").write_text("retain")
    subject.prepare_job_folder(SimpleNamespace(job_folder=str(job), force=force))
    assert job.is_dir()
    assert (job / "app" / "old.py").exists() is (not force)
    assert (job / "notes").exists() is (not force)
    if not force:
        assert "overwrite" in capsys.readouterr().out


def test_job_folder_creation_and_invalid_existing_file_are_reported(tmp_path):
    destination = tmp_path / "new"
    subject.prepare_job_folder(SimpleNamespace(job_folder=str(destination), force=False))
    assert destination.is_dir()
    file = tmp_path / "file"
    file.write_bytes(b"unrelated")
    with pytest.raises(ValueError, match="not directory"):
        subject.prepare_job_folder(SimpleNamespace(job_folder=str(file), force=False))
    assert file.read_bytes() == b"unrelated"


def test_app_layout_and_script_copy_remove_cache_without_modifying_script_source(tmp_path):
    job = tmp_path / "job"
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    (scripts / "train.py").write_text("print('training')")
    (scripts / "__pycache__").mkdir()
    (scripts / "__pycache__" / "train.pyc").write_bytes(b"cache")
    custom_dirs = subject.prepare_app_dirs(str(job), ["first", "second"])
    subject.prepare_app_scripts(str(job), custom_dirs, SimpleNamespace(script_dir=str(scripts)))
    for directory in custom_dirs:
        assert (Path(directory) / "train.py").read_text() == "print('training')"
        assert not (Path(directory) / "__pycache__").exists()
    assert (scripts / "__pycache__" / "train.pyc").exists()
    assert subject.get_config_dirs(str(job), ["first", "second"]) == [
        str(job / "first" / "config"),
        str(job / "second" / "config"),
    ]
    assert subject.get_config_dirs(str(job), []) == [str(job / "app" / "config")]
    assert subject.dst_config_path(str(job), "config.json", "first") == str(job / "first" / "config" / "config.json")
    assert len(subject.prepare_app_dirs(str(tmp_path / "default"), [])) == 1


@pytest.mark.parametrize("case", ["missing", "same", "parent"])
def test_script_copy_rejects_missing_sources_and_recursive_directory_relationships(tmp_path, case):
    root = tmp_path / "root"
    root.mkdir()
    script = tmp_path / "missing" if case == "missing" else root
    destination = root / "job" if case == "parent" else root
    with pytest.raises(ValueError, match="doesn't exists" if case == "missing" else "same or sub"):
        subject.prepare_app_scripts(
            str(destination), [str(tmp_path / "custom")], SimpleNamespace(script_dir=str(script))
        )
    assert not (tmp_path / "custom").exists()


@pytest.mark.parametrize("force", [False, True])
def test_meta_configuration_preserves_user_file_or_forces_template_and_cleans_app_local_copies(tmp_path, force):
    job, template = tmp_path / "my_job", tmp_path / "template"
    config = job / "app" / "config"
    config.mkdir(parents=True)
    template.mkdir()
    (job / "meta.conf").write_text('name="original"\nmin_clients=7')
    (config / "meta.json").write_text('{"obsolete": true}')
    (template / "meta.conf").write_text('name="template"\nmin_clients=2')
    subject.prepare_meta_config(SimpleNamespace(job_folder=str(job) + "/", force=force), str(template), [])
    saved = ConfigFactory.parse_file(str(job / "meta.conf"))
    assert saved.get("name") == ("my_job" if force else "original")
    assert saved.get("min_clients") == (2 if force else 7)
    assert not (config / "meta.json").exists()


def test_meta_configuration_uses_default_template_when_no_source_meta_exists(tmp_path):
    job, template = tmp_path / "job", tmp_path / "template"
    job.mkdir()
    template.mkdir()
    subject.prepare_meta_config(SimpleNamespace(job_folder=str(job), force=False), str(template), ["custom"])
    assert ConfigFactory.parse_file(str(job / "meta.conf")).get("name") == "job"


def test_template_directory_update_preserves_existing_hidden_settings_and_handles_missing_config(tmp_path, monkeypatch):
    (tmp_path / "templates").mkdir()
    save = Mock()
    monkeypatch.setattr(subject, "save_config", save)
    monkeypatch.setattr(
        subject, "load_hidden_config_state", Mock(return_value=(str(tmp_path / "config.conf"), None, False))
    )
    subject.update_job_templates_dir(str(tmp_path / "templates"))
    assert save.call_args.args[0].get("job_template.path") == str(tmp_path / "templates")
    assert save.call_args.args[1] == str(tmp_path / "config.conf")
    monkeypatch.setattr(
        subject,
        "load_hidden_config_state",
        Mock(return_value=(str(tmp_path / "config.conf"), ConfigFactory.parse_string("other=7"), False)),
    )
    subject.update_job_templates_dir(str(tmp_path / "templates"))
    assert save.call_args.args[0].get("other") == 7


def test_removing_template_metadata_keeps_runtime_configuration(tmp_path):
    (tmp_path / "info.conf").write_text("metadata")
    (tmp_path / "info.md").write_text("docs")
    (tmp_path / "__init__.py").write_text("")
    (tmp_path / "__pycache__").mkdir()
    (tmp_path / "config_fed_server.conf").write_text("runtime")
    subject.remove_extra_files(str(tmp_path))
    assert sorted(p.name for p in tmp_path.iterdir()) == ["config_fed_server.conf"]
