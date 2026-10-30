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

import os
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import yaml

from nvflare.lighter.cc_provision.impl import onprem_packager as subject
from nvflare.lighter.constants import ProvFileName
from tests.timing_utils import isolate_time


@pytest.mark.parametrize(
    "output,expected", [("building\nCVM Bundle /tmp/site-a.tgz is ready\n", "/tmp/site-a.tgz"), ("missing", None)]
)
def test_cvm_archive_discovery_extracts_only_ready_bundle_path(output, expected):
    assert subject._extract_cvm_tar_path(output) == expected
    assert subject._extract_docker_tar_path("DOCKER_ARCHIVE=/tmp/image.tar\n") == "/tmp/image.tar"
    assert subject._extract_docker_tar_path("missing") is None


@pytest.mark.parametrize(
    "yaml_path,file_path", [(None, "file"), ("config.yaml", None), (123, "file"), ("config.yaml", 123)]
)
def test_package_paths_require_nonempty_string_inputs(yaml_path, file_path):
    with pytest.raises(ValueError):
        subject.to_abs_path(yaml_path, file_path)


def test_relative_build_command_is_resolved_against_config_directory(tmp_path):
    path = tmp_path / "config" / "project.yaml"
    assert subject.to_abs_path(str(path), "../build.sh") == str(tmp_path / "build.sh")
    assert subject.to_abs_path(str(path), str(tmp_path / "absolute.sh")) == str(tmp_path / "absolute.sh")


@pytest.mark.parametrize("return_code", [0, 7])
def test_command_boundary_preserves_stdout_and_stderr_and_reports_failure(monkeypatch, return_code):
    process = Mock()
    process.stdout.readline.side_effect = ["progress\n", "completed\n", ""]
    process.stderr.readline.side_effect = ["warning\n", "", ""]
    process.poll.side_effect = [return_code]
    process.wait.return_value = return_code
    factory = Mock(return_value=process)
    subprocess_proxy = SimpleNamespace(**vars(subject.subprocess))
    subprocess_proxy.Popen = factory
    monkeypatch.setattr(subject, "subprocess", subprocess_proxy)
    if return_code:
        with pytest.raises(subject.subprocess.CalledProcessError) as raised:
            subject.run_command(["build", "config.yaml"], cwd="/working")
        assert raised.value.returncode == 7
        assert raised.value.output == "progress\ncompleted\n"
        assert raised.value.stderr == "warning\n"
    else:
        assert subject.run_command(["build", "config.yaml"], cwd="/working") == "progress\ncompleted\n"
    assert factory.call_args.kwargs["shell"] is False
    assert factory.call_args.kwargs["cwd"] == "/working"
    process.wait.assert_called_once_with()


@pytest.mark.parametrize("failure", [None, "build-error", "missing-archive"])
def test_participant_packaging_removes_temporary_config_and_replaces_kit_only_after_success(
    tmp_path, monkeypatch, failure
):
    config = tmp_path / "cc.yaml"
    config.write_text(yaml.safe_dump({"user_config": {"original": "preserved"}}))
    site = tmp_path / "site-a"
    site.mkdir()
    (site / "startup.txt").write_text("startup kit")
    archive = tmp_path / "built.tgz"
    archive.write_bytes(b"archive")
    participant = SimpleNamespace(name="site-a", get_prop=lambda key: str(config))
    context = SimpleNamespace(get_result_location=lambda: str(tmp_path))
    packager = subject.OnPremPackager()
    temporary = []

    def build(path):
        temporary.append(path)
        with open(path) as stream:
            data = yaml.safe_load(stream)
        assert data["user_config"] == {"original": "preserved", "nvflare": str(site)}
        if failure == "build-error":
            raise RuntimeError("builder failed")
        return None if failure == "missing-archive" else str(archive)

    monkeypatch.setattr(packager, "_build_cc_image", build)
    if failure:
        with pytest.raises(RuntimeError, match="builder failed" if failure == "build-error" else "CVM build failed"):
            packager._package_for_participant(participant, context)
        assert (site / "startup.txt").read_text() == "startup kit"
    else:
        packager._package_for_participant(participant, context)
        assert sorted(p.name for p in site.iterdir()) == ["site-a.tgz"]
        assert (site / "site-a.tgz").read_bytes() == b"archive"
    assert len(temporary) == 1 and not os.path.exists(temporary[0])
    assert yaml.safe_load(config.read_text()) == {"user_config": {"original": "preserved"}}


def test_cc_image_requires_executable_command_before_invoking_subprocess(tmp_path, monkeypatch):
    config = tmp_path / "config.yaml"
    config.write_text("{}")
    command = tmp_path / "build.sh"
    command.write_text("#!/bin/sh\n")
    packager = subject.OnPremPackager(build_image_cmd=str(command))
    invoke = Mock(return_value="CVM Bundle /tmp/built.tgz is ready")
    monkeypatch.setattr(subject, "run_command", invoke)
    with pytest.raises(FileNotFoundError, match="not found or is not executable"):
        packager._build_cc_image(str(config))
    invoke.assert_not_called()
    command.chmod(0o700)
    assert packager._build_cc_image(str(config)) == "/tmp/built.tgz"
    invoke.assert_called_once_with([str(command), str(config)])


def test_packager_removes_start_all_and_spaces_participant_builds_with_controlled_clock(tmp_path, monkeypatch):
    start = tmp_path / ProvFileName.START_ALL_SH
    start.write_text("obsolete")
    participants = [SimpleNamespace(name="a"), SimpleNamespace(name="b")]
    project = SimpleNamespace(get_all_participants=lambda: participants)
    context = SimpleNamespace(get_result_location=lambda: str(tmp_path))
    packager = subject.OnPremPackager()
    package = Mock()
    monkeypatch.setattr(packager, "_package_for_participant", package)
    clock = isolate_time(monkeypatch, subject)
    monkeypatch.setattr(clock, "sleep", Mock())
    packager.package(project, context)
    assert not start.exists()
    assert [c.args for c in package.call_args_list] == [(participant, context) for participant in participants]
    clock.sleep.assert_called_once_with(100.0)
