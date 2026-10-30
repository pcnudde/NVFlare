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
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from nvflare.lighter import tree_prov as subject
from nvflare.lighter.entity import ParticipantType, Project
from nvflare.lighter.spec import Builder


@pytest.fixture(autouse=True)
def reset_topology_counters(monkeypatch):
    for key in (
        "num_relays",
        "num_leaf_relays",
        "num_non_leaf_relays",
        "num_clients",
        "num_leaf_clients",
        "num_non_leaf_clients",
    ):
        monkeypatch.setattr(subject.Stats, key, 0)
    monkeypatch.setattr(subject.PortManager, "last_port_number", 9000)


def parameters(**overrides):
    result = {
        "depth": 2,
        "width": 2,
        "clients": 2,
        "root_dir": "/workspace",
        "project_name": "tree",
        "rp": 4321,
        "max_sites": 100,
    }
    result.update(overrides)
    return result


@pytest.mark.parametrize("leaf_only", [False, True])
def test_tree_provision_builds_expected_relay_and_client_hierarchies(monkeypatch, leaf_only):
    provision = Mock()
    factory = Mock(return_value=provision)
    monkeypatch.setattr(subject, "Provisioner", factory)
    project = Project("tree", "topology test")
    admin = subject._new_participant("admin@example.com", ParticipantType.ADMIN, {"role": "project_admin"})
    subject.hierachical_provision(parameters(lcp_only=leaf_only), project, [Builder()], [admin])
    assert len(project.get_relays()) == 6
    assert len(project.get_clients()) == (8 if leaf_only else 14)
    assert project.get_admins() == [admin]
    assert project.get_server().name == "server"
    assert (subject.Stats.num_leaf_relays, subject.Stats.num_non_leaf_relays) == (4, 2)
    assert subject.Stats.num_leaf_clients == 8
    packager = factory.call_args.args[2]
    assert len(packager.lcp_map) == 8
    assert len({item["port"] for item in packager.lcp_map.values()}) == 8
    assert all(item["host"] == "localhost" for item in packager.lcp_map.values())
    clients = {client.name: client for client in project.get_clients()}
    leaf = clients["C111"]
    assert leaf.get_prop("connect_to")["name"] == "R11"
    assert leaf.get_prop("parent") == (None if leaf_only else "C11")
    if not leaf_only:
        assert clients["C11"].get_prop("parent") == "C1"
    provision.provision.assert_called_once_with(project)


@pytest.mark.parametrize("options", [{"analyze": True}, {"max_sites": 1}, {"depth": 0}, {"width": 1}, {"clients": 1}])
def test_invalid_or_analysis_topology_never_invokes_provisioner(monkeypatch, options, capsys):
    factory = Mock()
    monkeypatch.setattr(subject, "Provisioner", factory)
    subject.hierachical_provision(parameters(**options), Project("tree", "test"), [], [])
    factory.assert_not_called()
    assert capsys.readouterr().out


@pytest.mark.parametrize(
    "params,project,builders,admins",
    [
        (None, None, [], []),
        (None, Project("tree", "test"), "bad", []),
        (None, Project("tree", "test"), [], None),
        (None, Project("tree", "test"), [], []),
        ({}, Project("tree", "test"), [], []),
    ],
)
def test_tree_provision_rejects_invalid_inputs_before_creating_sites(params, project, builders, admins):
    with pytest.raises(ValueError):
        subject.hierachical_provision(params, project, builders, admins)


def test_tree_packager_writes_proxy_and_simulator_scripts_and_preserves_ca(tmp_path):
    startup = tmp_path / "server" / "startup"
    startup.mkdir(parents=True)
    (startup / "rootCA.pem").write_text("trusted root")
    ctx = SimpleNamespace(get_result_location=lambda: str(tmp_path))
    packager = subject._Packager({"site-a": {"host": "localhost", "port": 9001}}, 4321)
    packager.package(Project("tree", "test"), ctx)
    scripts = tmp_path / "scripts"
    assert json.loads((scripts / subject.LCP_MAP_BASENAME).read_text()) == packager.lcp_map
    assert (scripts / "rootCA.pem").read_text() == "trusted root"
    assert "routing_proxy 4321 lcp_map.json rootCA.pem" in (scripts / "start_rp.sh").read_text()
    assert "-m lcp_map.json -c rootCA.pem" in (scripts / "simulate_lcp.sh").read_text()
    config = json.loads((scripts / subject.SIMULATION_CONFIG).read_text())
    assert config["endpoint"] == "http://localhost:4321"
    assert config["num_devices"] == 10000


def test_tree_cli_wires_requested_topology_and_provision_builders(monkeypatch):
    argv = [
        "tree_prov",
        "--root_dir",
        "/workspace",
        "--project_name",
        "edge-test",
        "--depth",
        "2",
        "--width",
        "3",
        "--clients",
        "4",
        "--analyze",
        "--lcp_only",
    ]
    monkeypatch.setattr("sys.argv", argv)
    provision = Mock()
    monkeypatch.setattr(subject, "hierachical_provision", provision)
    subject.main()
    params, project, builders, admins = provision.call_args.args
    assert params["depth"] == 2 and params["width"] == 3 and params["clients"] == 4
    assert params["analyze"] and params["lcp_only"]
    assert project.name == "edge-test"
    assert [type(builder).__name__ for builder in builders] == [
        "WorkspaceBuilder",
        "StaticFileBuilder",
        "CertBuilder",
        "SignatureBuilder",
        "EdgeBuilder",
    ]
    assert admins[0].name == "admin@nvidia.com"
