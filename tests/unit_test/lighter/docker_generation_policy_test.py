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
import yaml

from nvflare.lighter.constants import CtxKey, ProvFileName
from nvflare.lighter.impl.docker import DockerBuilder


@pytest.mark.parametrize("server_present,requirements_present", [(True, True), (False, False)])
def test_docker_builder_renders_participant_services_and_build_inputs_only(
    tmp_path, server_present, requirements_present
):
    requirements = tmp_path / "input-requirements.txt"
    if requirements_present:
        requirements.write_text("nvflare==2.7.0\n")
    output = tmp_path / "output"
    output.mkdir()
    templates = {
        "__flserver__": {"volumes": ["source:${WORKSPACE}"], "command": ["flserver", "org=__org_name__"]},
        "__flclient__": {"volumes": [], "command": ["flclient", "uid=__flclient__", "org=__org_name__"]},
    }
    context = Mock()
    context.yaml_load_template_section.return_value = {"version": "3", "services": templates}
    context.get_wip_dir.return_value = str(output)
    context.get.side_effect = lambda key: {CtxKey.FED_LEARN_PORT: 8002, CtxKey.ADMIN_PORT: 8003}.get(key)
    context.get_template_section.return_value = "RUN echo installed\n"
    server = SimpleNamespace(name="server-a", org="server-org") if server_present else None
    clients = [SimpleNamespace(name="site-a", org="client-org"), SimpleNamespace(name="site-b", org="other-org")]
    project = SimpleNamespace(get_server=lambda: server, get_clients=lambda: clients)
    builder = DockerBuilder(base_image="python:3.12", requirements_file=str(requirements))
    builder.build(project, context)
    compose = yaml.safe_load((output / ProvFileName.COMPOSE_YAML).read_text())
    services = compose["services"]
    assert set(services) == ({"server-a", "site-a", "site-b"} if server_present else {"site-a", "site-b"})
    assert services["site-a"]["command"] == ["site-a", "uid=site-a", "org=client-org"]
    assert services["site-b"]["command"] == ["site-b", "uid=site-b", "org=other-org"]
    assert services["site-a"]["volumes"] == ["./site-a:${WORKSPACE}"]
    if server_present:
        assert services["server-a"]["ports"] == ["8002:8002", "8003:8003"]
        assert services["server-a"]["command"] == ["server-a", "org=server-org"]
    build = output / ProvFileName.COMPOSE_BUILD_DIR
    assert (build / ProvFileName.DOCKERFILE).read_text() == "FROM python:3.12\nRUN echo installed\n"
    assert (build / ProvFileName.REQUIREMENTS_TXT).read_text() == ("nvflare==2.7.0\n" if requirements_present else "")
    assert "WORKSPACE=/workspace" in (output / ProvFileName.ENV).read_text()
