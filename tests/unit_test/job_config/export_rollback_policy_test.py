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

import json
import os
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from nvflare.job_config import fed_job_config as subject
from nvflare.job_config.api import FedJob
from nvflare.job_config.fed_job_config import FedJobConfig


def test_statistics_job_export_wires_privacy_cleanser_chain_for_each_client(tmp_path):
    pytest.importorskip("pandas")
    from nvflare.app_opt.statistics.df.df_core_statistics import DFStatisticsCore
    from nvflare.job_config.stats_job import StatsJob

    job = StatsJob(
        "statistics",
        {"count": {}},
        DFStatisticsCore(),
        "stats/output.json",
        min_count=5,
        min_noise_level=0.2,
        max_noise_level=0.4,
        max_bins_percent=8,
    )
    job.setup_clients(["site-a", "site-b"])
    job.export_job(str(tmp_path))
    root = tmp_path / "statistics"
    server_paths = list(root.rglob("config_fed_server.json"))
    client_paths = list(root.rglob("config_fed_client.json"))
    assert len(server_paths) == 1 and len(client_paths) == 2
    server = json.loads(server_paths[0].read_text())
    workflow = server["workflows"][0]
    assert workflow["path"].endswith("StatisticsController")
    assert workflow["args"]["writer_id"] == "stats_writer"
    for path in client_paths:
        data = json.loads(path.read_text())
        components = {item["id"]: item for item in data["components"]}
        assert components["min_count_cleanser"]["args"]["min_count"] == 5
        assert components["min_max_noise_cleanser"]["args"]["min_noise_level"] == 0.2
        assert components["hist_bins_cleanser"]["args"]["max_bins_percent"] == 8
        assert set(data["executors"][0]["tasks"]) == {"fed_stats", "fed_stats_pre_run"}
        privacy = data["task_result_filters"][0]
        assert privacy["tasks"] == ["fed_stats"]
        assert privacy["filters"][0]["args"]["result_cleanser_ids"] == [
            "min_count_cleanser",
            "min_max_noise_cleanser",
            "hist_bins_cleanser",
        ]


def test_failed_export_replacement_restores_old_owned_directory_and_removes_partial_new_tree(tmp_path, monkeypatch):
    config = FedJobConfig("atomic", 1, mandatory_clients=["site"])
    config.generate_job_config(str(tmp_path))
    target = tmp_path / "atomic"
    (target / "old-result").write_text("preserve")
    original_meta = (target / "meta.json").read_text()
    config.meta_props = {"description": "new"}
    isolated_os = SimpleNamespace(**vars(os))
    replace = os.replace

    def publish(source, destination):
        if os.path.basename(source).startswith(".atomic.") and os.fspath(destination) == str(target):
            raise OSError("publish failed")
        return replace(source, destination)

    isolated_os.replace = publish
    monkeypatch.setattr(subject, "os", isolated_os)
    with pytest.raises(OSError, match="publish failed"):
        config.generate_job_config(str(tmp_path))
    assert (target / "old-result").read_text() == "preserve"
    assert (target / "meta.json").read_text() == original_meta
    assert json.loads(original_meta)["mandatory_clients"] == ["site"]
    assert not list(tmp_path.glob(".atomic.*"))


def test_atomic_metadata_failure_removes_temp_file_and_preserves_previous_contents(tmp_path, monkeypatch):
    config = FedJobConfig("atomic", 1)
    existing = tmp_path / "meta.json"
    existing.write_text("old metadata")
    isolated_os = SimpleNamespace(**vars(os))
    isolated_os.replace = Mock(side_effect=OSError("filesystem refused replace"))
    monkeypatch.setattr(subject, "os", isolated_os)
    with pytest.raises(OSError, match="refused replace"):
        config._generate_meta(str(tmp_path), "new metadata")
    assert existing.read_text() == "old metadata"
    assert not (tmp_path / "meta.json.tmp").exists()


def test_simulator_interrupt_invokes_process_cleanup_boundary_and_returns_aborted_code(tmp_path, monkeypatch):
    config = FedJobConfig("sim", 1)
    process = Mock()
    process.wait.side_effect = KeyboardInterrupt()
    start, cleanup = Mock(return_value=process), Mock()
    monkeypatch.setattr(subject, "popen_in_new_session", start)
    monkeypatch.setattr(subject, "kill_child_processes", cleanup)
    assert config.simulator_run(str(tmp_path), n_clients=2, threads=1) == -9
    cleanup.assert_called_once_with(os.getpid())
    assert start.call_args.kwargs["shell"] is False
    assert "-n" in start.call_args.args[0]


@pytest.mark.parametrize(
    "kind,method,args,match",
    [
        ("config", "add_fed_app", ("app", object()), "FedAppConfig"),
        ("config", "set_site_app", ("site", "absent"), "does not exist"),
        ("config", "add_resource_spec", ("site", []), "must be a dict"),
        ("job", "to_server", (None,), "empty object"),
        ("job", "set_execution_lifetime", ("task", "server"), "applies to client apps"),
    ],
)
def test_invalid_job_configuration_is_rejected_before_mutating_deployment(kind, method, args, match):
    obj = FedJobConfig("valid", 1) if kind == "config" else FedJob("valid")
    with pytest.raises((ValueError, RuntimeError), match=match):
        getattr(obj, method)(*args)
    assert not (obj.deploy_map if kind == "config" else obj._deploy_map)


def test_job_metadata_packages_and_keyword_contract_validate_without_losing_prior_settings():
    config = FedJobConfig("valid", 1)
    config.set_app_packages(None)
    assert config.app_packages == []
    config.set_app_packages(["numpy"])
    assert config.app_packages == ["numpy"]
    config.add_resource_spec("site", {"num_of_cpus": 2})
    with pytest.raises(RuntimeError, match="already exist"):
        config.add_resource_spec("site", {"num_of_cpus": 3})
    assert config.resource_specs["site"] == {"num_of_cpus": 2}
    with pytest.raises(ValueError, match="no args expected"):
        FedJob.check_kwargs({"unknown": 1}, {})
    with pytest.raises(ValueError, match="Missing required arg"):
        FedJob.check_kwargs({}, {"required": True})
