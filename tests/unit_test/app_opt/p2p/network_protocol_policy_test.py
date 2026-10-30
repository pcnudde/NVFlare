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

from unittest.mock import MagicMock

import networkx as nx
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from nvflare.apis.dxo import DataKind, from_shareable
from nvflare.apis.fl_context import FLContext
from nvflare.apis.signal import Signal
from nvflare.app_opt.p2p.controllers.dist_opt_controller import DistOptController
from nvflare.app_opt.p2p.types import Config, Neighbor, Network, Node
from nvflare.app_opt.p2p.utils import config_generator
from nvflare.app_opt.p2p.utils.topology import doubly_stochastic_adjacency, get_matrix_steady_state


def test_metropolis_hastings_weights_are_nonnegative_doubly_stochastic_and_have_uniform_stationary_vector():
    weights = doubly_stochastic_adjacency(nx.path_graph(3))
    expected = torch.tensor([[2 / 3, 1 / 3, 0], [1 / 3, 1 / 3, 1 / 3], [0, 1 / 3, 2 / 3]])
    torch.testing.assert_close(weights, expected)
    torch.testing.assert_close(weights.sum(0), torch.ones(3))
    torch.testing.assert_close(weights.sum(1), torch.ones(3))
    steady = get_matrix_steady_state(weights)
    torch.testing.assert_close(steady, torch.ones(3) / 3)
    torch.testing.assert_close(weights @ steady, steady)


def test_network_generation_retries_disconnected_graph_and_maps_weights_to_neighbors(monkeypatch):
    graph = nx.path_graph(3)
    generator = MagicMock(side_effect=[nx.empty_graph(3), graph])
    monkeypatch.setattr(config_generator.nx, "gnp_random_graph", generator)
    random_state = np.random.get_state()
    try:
        network, weights = config_generator.generate_random_network(3, seed=7, connection_probability=0.4)
    finally:
        np.random.set_state(random_state)
    assert generator.call_count == 2
    assert [node.id for node in network.nodes] == ["site-1", "site-2", "site-3"]
    assert [[neighbor.id for neighbor in node.neighbors] for node in network.nodes] == [
        ["site-2"],
        ["site-1", "site-3"],
        ["site-2"],
    ]
    for i, node in enumerate(network.nodes):
        for neighbor in node.neighbors:
            j = int(neighbor.id.split("-")[1]) - 1
            assert neighbor.weight == weights[i, j].item()


def test_controller_sends_each_neighbor_configuration_before_broadcasting_algorithm_parameters(caplog):
    nodes = [Node("site-1", [Neighbor("site-2", 0.25)]), Node("site-2", [Neighbor("site-1", 0.5)])]
    controller = DistOptController(Config(Network(nodes), extra={"stepsize": 0.1, "iterations": 3}))
    order = []
    controller.send_and_wait = MagicMock(side_effect=lambda **kwargs: order.append("configure"))
    controller.broadcast_and_wait = MagicMock(side_effect=lambda **kwargs: order.append("run"))
    context = FLContext()
    controller.control_flow(Signal(), context)
    assert order == ["configure", "configure", "run"]
    for node, call in zip(nodes, controller.send_and_wait.call_args_list):
        assert call.kwargs["targets"] == [node.id]
        task = call.kwargs["task"]
        assert task.name == "config"
        dxo = from_shareable(task.data)
        assert dxo.data_kind == DataKind.APP_DEFINED
        assert dxo.data == {"neighbors": [neighbor.__dict__ for neighbor in node.neighbors]}
    kwargs = controller.broadcast_and_wait.call_args.kwargs
    assert kwargs["targets"] == ["site-1", "site-2"]
    assert kwargs["min_responses"] == 0
    assert kwargs["task"].name == "run_algorithm"
    assert from_shareable(kwargs["task"].data).data == {"stepsize": 0.1, "iterations": 3}
    caplog.set_level("INFO")
    controller.start_controller(context)
    controller.stop_controller(context)
    assert "started" in caplog.text and "stopped" in caplog.text
