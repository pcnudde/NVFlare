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

import pytest

from nvflare.apis.fl_constant import WorkspaceConstants as Keys
from nvflare.apis.workspace import Workspace


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    for name in (Keys.ENV_VAR_RESULT_ROOT, Keys.ENV_VAR_AUDIT_ROOT, Keys.ENV_VAR_LOG_ROOT):
        monkeypatch.delenv(name, raising=False)
    root = tmp_path / "workspace"
    (root / "startup").mkdir(parents=True)
    (root / "local").mkdir()
    return root


@pytest.mark.parametrize("selected", [Keys.ENV_VAR_RESULT_ROOT, Keys.ENV_VAR_AUDIT_ROOT, Keys.ENV_VAR_LOG_ROOT])
def test_output_roots_share_default_log_audit_location_and_isolate_site(workspace, tmp_path, monkeypatch, selected):
    output = tmp_path / "output"
    monkeypatch.setenv(selected, str(output))
    ws = Workspace(str(workspace), site_name="site1")
    assert ws.get_log_root("job1") == str(output / "site1" / "job1")
    assert ws.get_audit_root("job1") == str(output / "site1" / "job1")
    assert Path(ws.get_log_root("job1")).is_dir()
    assert ws.get_log_root() == str(output / "site1")
    if selected == Keys.ENV_VAR_RESULT_ROOT:
        assert ws.get_result_root("job1") == str(output / "site1" / "job1")
    else:
        assert ws.get_result_root("job1") == ws.get_run_dir("job1")
    assert Path(ws.get_audit_file_path("job1")).parent == output / "site1" / "job1"


@pytest.mark.parametrize("server, job", [(True, True), (True, False), (False, True), (False, False)])
def test_startup_resources_include_only_process_specific_overlays_in_precedence_order(workspace, server, job):
    local, startup = workspace / "local", workspace / "startup"
    (local / Keys.DEFAULT_RESOURCES_CONFIG).write_text("{}")
    (local / Keys.JOB_RESOURCES_CONFIG).write_text("{}")
    for folder in (startup, local):
        for name in ("shared__resources.json", "parent__p_resources.json", "job__j_resources.json"):
            (folder / name).write_text("{}")
    ws = Workspace(str(workspace))
    files = ws.get_config_files_for_startup(server, job)
    assert files[0] == str(startup / (Keys.SERVER_STARTUP_CONFIG if server else Keys.CLIENT_STARTUP_CONFIG))
    assert files[1] == str(local / Keys.DEFAULT_RESOURCES_CONFIG)
    assert (str(local / Keys.JOB_RESOURCES_CONFIG) in files) is job
    included, excluded = (
        ("job__j_resources.json", "parent__p_resources.json")
        if job
        else ("parent__p_resources.json", "job__j_resources.json")
    )
    overlays = files[3 if job else 2 :]
    assert overlays == [
        str(folder / name) for folder in (startup, local) for name in ("shared__resources.json", included)
    ]
    assert not any(path.endswith(excluded) for path in files)


def test_site_configuration_fallback_chooses_first_existing_file(workspace):
    local = workspace / "local"
    ws = Workspace(str(workspace))
    assert ws.get_authorization_file_path() is None
    assert ws.get_log_config_file_path() is None
    (local / Keys.DEFAULT_AUTHORIZATION_CONFIG).write_text("{}")
    assert ws.get_authorization_file_path() == str(local / Keys.DEFAULT_AUTHORIZATION_CONFIG)
    (local / Keys.AUTHORIZATION_CONFIG).write_text("{}")
    assert ws.get_authorization_file_path() == str(local / Keys.AUTHORIZATION_CONFIG)
    assert ws.get_file_path_in_site_config(["absent", Keys.AUTHORIZATION_CONFIG]) == str(
        local / Keys.AUTHORIZATION_CONFIG
    )
    with pytest.raises(ValueError, match="file_basename"):
        ws.get_file_path_in_site_config(42)
    assert ws.get_log_root("job1") == ws.get_run_dir("job1")
    assert ws.get_audit_root() == str(workspace)
    assert ws.get_log_root() == str(workspace)
    assert ws.get_audit_root("job1") == ws.get_run_dir("job1")


def test_stats_paths_preserve_prefix_and_custom_configuration_directory(workspace):
    ws = Workspace(str(workspace), site_name="site1", config_folder="custom-config")
    run = Path(ws.get_run_dir("job1"))
    assert ws.get_app_config_dir("job1") == str(run / "app_site1" / "custom-config")
    assert ws.get_app_custom_dir("job1") == str(run / "app_site1" / "custom")
    assert ws.get_stats_pool_summary_path("job1", "worker") == str(
        run / ("worker." + Keys.STATS_POOL_SUMMARY_FILE_NAME)
    )
    assert ws.get_stats_pool_records_path("job1", "worker") == str(
        run / ("worker." + Keys.STATS_POOL_RECORDS_FILE_NAME)
    )
    assert ws.get_stats_pool_summary_path("job1") == str(run / Keys.STATS_POOL_SUMMARY_FILE_NAME)
    assert ws.get_stats_pool_records_path("job1") == str(run / Keys.STATS_POOL_RECORDS_FILE_NAME)
