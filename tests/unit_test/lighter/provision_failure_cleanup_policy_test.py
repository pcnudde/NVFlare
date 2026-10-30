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

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from nvflare.lighter.constants import CtxKey, WorkDir
from nvflare.lighter.entity import Project
from nvflare.lighter.provisioner import Provisioner
from nvflare.lighter.spec import Builder, Packager


@pytest.mark.parametrize("case", ["empty", "wrong_builder", "wrong_packager", "bad_template"])
def test_provisioner_rejects_invalid_workflow_components_before_build(tmp_path, case):
    with pytest.raises(
        ValueError,
        match={
            "empty": "no builders",
            "wrong_builder": "Builder type",
            "wrong_packager": "Packager type",
            "bad_template": "dict",
        }[case],
    ):
        if case == "bad_template":
            Provisioner(str(tmp_path), [Builder()]).add_template("bad")
        else:
            Provisioner(
                str(tmp_path),
                [] if case == "empty" else [object()] if case == "wrong_builder" else [Builder()],
                object() if case == "wrong_packager" else None,
            )


@pytest.mark.parametrize("logger", [SimpleNamespace(), SimpleNamespace(info=1)])
def test_provisioner_requires_callable_logger_contract_before_allocating_workspace(tmp_path, logger):
    provisioner = Provisioner(str(tmp_path), [Builder()])
    with pytest.raises(ValueError, match="invalid logger"):
        provisioner.provision(object(), logger=logger)
    assert not list(tmp_path.iterdir())


def test_provisioner_failure_removes_incomplete_prod_and_skips_packaging(tmp_path):
    project = Project("project", "description")
    project.set_server("server", "org", {})
    prod = tmp_path / "incomplete-prod"
    prod.mkdir()
    calls = []

    class FailingBuilder(Builder):
        def initialize(self, project, context):
            context[WorkDir.CURRENT_PROD_DIR] = str(prod)
            calls.append("initialize")

        def build(self, project, context):
            calls.append("build")
            raise ValueError("controlled failure")

        def finalize(self, project, context):
            calls.append("finalize")

    packager = Mock(spec=Packager)
    provisioner = Provisioner(str(tmp_path), [FailingBuilder()], packager)
    logger = Mock()
    result = provisioner.provision(project, logger=logger)
    assert result[CtxKey.BUILD_ERROR]
    assert calls == ["initialize", "build"]
    assert not prod.exists()
    assert not (tmp_path / "project" / "wip").exists()
    assert "controlled failure" in result.get_errors()[0]
    logger.error.assert_called_once()
    packager.package.assert_not_called()


def test_provisioner_rejects_missing_server_and_invalid_mode(tmp_path):
    project = Project("project", "description")
    subject = Provisioner(str(tmp_path), [Builder()])
    with pytest.raises(RuntimeError, match="missing server"):
        subject.provision(project)
    project.set_server("server", "org", {})
    with pytest.raises(ValueError, match="invalid mode"):
        subject.provision(project, mode="bad")
