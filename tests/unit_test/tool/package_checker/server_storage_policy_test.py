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
from unittest.mock import Mock

import pytest

from nvflare.tool.package_checker import check_rule
from nvflare.tool.package_checker import server_package_checker as subject
from nvflare.tool.package_checker.check_rule import CHECK_PASSED, CheckAddressBinding, CheckWriting
from nvflare.tool.package_checker.package_checker import PackageChecker


def config(tmp_path, **overrides):
    startup = tmp_path / "startup"
    startup.mkdir(exist_ok=True)
    data = {
        "servers": [{"service": {"target": "external:8002", "scheme": "tcp"}, "admin_port": 8003}],
        "snapshot_persistor": {
            "path": "nvflare.app_common.state_persistors.storage_state_persistor.StorageStatePersistor",
            "args": {
                "storage": {
                    "path": "nvflare.app_common.storages.filesystem_storage.FilesystemStorage",
                    "args": {"root_dir": str(tmp_path / "snapshots")},
                }
            },
        },
        "components": [
            {
                "path": "nvflare.apis.impl.job_def_manager.SimpleJobDefManager",
                "args": {"uri_root": str(tmp_path / "jobs")},
            }
        ],
    }
    data.update(overrides)
    (startup / "fed_server.json").write_text(json.dumps(data))
    return data


def test_server_checker_builds_storage_and_port_rules_and_cleans_owned_dry_run_directories(tmp_path, monkeypatch):
    config(tmp_path)
    checker = subject.ServerPackageChecker()
    checker.init(str(tmp_path))
    assert checker.should_be_checked() and checker.dry_run_timeout == 3
    assert len(checker.rules) == 4
    assert subject._get_fl_host_and_port(str(tmp_path)) == ("localhost", 8002)
    assert subject._get_admin_host_and_port(str(tmp_path)) == ("localhost", 8003)
    command = checker.get_dry_run_command()
    assert "nvflare.private.fed.app.server.server_train" in command and "secure_train=true" in command
    assert checker.snapshot_storage_root == str(tmp_path / "snapshots")
    assert checker.job_storage_root == str(tmp_path / "jobs")
    (tmp_path / "snapshots").mkdir()
    (tmp_path / "jobs").mkdir()
    stop = Mock()
    monkeypatch.setattr(PackageChecker, "stop_dry_run", stop)
    checker.stop_dry_run(force=False)
    stop.assert_called_once_with(force=False)
    assert not (tmp_path / "snapshots").exists() and not (tmp_path / "jobs").exists()


def test_server_detection_and_nonstandard_storage_do_not_invent_paths(tmp_path):
    checker = subject.ServerPackageChecker()
    checker.package_path = str(tmp_path)
    assert not checker.should_be_checked()
    config(tmp_path, snapshot_persistor={}, components=[])
    assert subject._get_snapshot_storage_root(str(tmp_path)) == ""
    assert subject._get_job_storage_root(str(tmp_path)) == ""
    data = config(tmp_path)
    data["servers"][0]["service"]["scheme"] = "unsupported"
    (tmp_path / "startup/fed_server.json").write_text(json.dumps(data))
    with pytest.raises(RuntimeError, match="not supported"):
        checker.init(str(tmp_path))


@pytest.mark.parametrize("error", [None, PermissionError("denied")])
def test_address_and_storage_rules_translate_probe_results_without_real_socket(monkeypatch, error):
    bind = Mock(return_value=error)
    write = Mock(return_value=error)
    monkeypatch.setattr(check_rule, "try_bind_address", bind)
    monkeypatch.setattr(check_rule, "try_write_dir", write)
    address = CheckAddressBinding("port", lambda package: ("localhost", 8))
    storage = CheckWriting("storage", lambda package: "/output")
    first, second = address("/kit"), storage("/kit")
    bind.assert_called_once_with("localhost", 8)
    write.assert_called_once_with("/output")
    assert (first.problem == CHECK_PASSED) is (error is None)
    assert (second.problem == CHECK_PASSED) is (error is None)
    if error:
        assert "denied" in first.problem and "denied" in second.problem
    write.reset_mock()
    assert CheckWriting("unused", lambda package: "")("/kit").problem == CHECK_PASSED
    write.assert_not_called()
