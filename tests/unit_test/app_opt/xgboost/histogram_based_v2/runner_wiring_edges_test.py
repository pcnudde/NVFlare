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

pytest.importorskip("xgboost")

from nvflare.apis.fl_context import FLContext  # noqa: E402
from nvflare.app_opt.xgboost.data_loader import XGBDataLoader  # noqa: E402
from nvflare.app_opt.xgboost.histogram_based_v2 import fed_controller, fed_executor  # noqa: E402
from nvflare.app_opt.xgboost.histogram_based_v2.defs import Constant  # noqa: E402
from nvflare.app_opt.xgboost.histogram_based_v2.mock.mock_controller import MockXGBController  # noqa: E402
from nvflare.app_opt.xgboost.histogram_based_v2.mock.mock_executor import MockXGBExecutor  # noqa: E402
from nvflare.app_opt.xgboost.histogram_based_v2.mock.mock_secure_executor import MockSecureXGBExecutor  # noqa: E402
from nvflare.app_opt.xgboost.histogram_based_v2.runners import xgb_server_runner  # noqa: E402
from nvflare.app_opt.xgboost.histogram_based_v2.sec import client_handler, server_handler  # noqa: E402


@pytest.mark.parametrize("kind", [MockXGBExecutor, MockSecureXGBExecutor])
def test_mock_xgb_executor_builds_correct_runner_and_adaptor_with_independent_cancellation(kind):
    executor = kind(in_process=False, int_server_grpc_options=[("limit", 1024)], per_msg_timeout=12, tx_timeout=45)
    adaptor = executor.get_adaptor(FLContext())
    assert not adaptor.in_process
    assert adaptor.int_server_grpc_options == [("limit", 1024)]
    assert adaptor.app_runner.__class__.__name__ == (
        "MockSecureClientRunner" if kind is MockSecureXGBExecutor else "MockClientRunner"
    )
    assert executor.per_msg_timeout == 12 and executor.tx_timeout == 45
    assert not adaptor.app_runner.asked_to_stop
    adaptor.app_runner.stop()
    assert adaptor.app_runner.asked_to_stop
    assert adaptor.starter is None and adaptor.process is None


def test_mock_xgb_controller_preserves_aggregation_deadline_and_server_cancellation():
    controller = MockXGBController(3, aggr_timeout=17, in_process=False, int_client_grpc_options=[("limit", 2048)])
    adaptor = controller.get_adaptor(FLContext())
    assert adaptor.app_runner.aggr_timeout == 17
    assert adaptor.int_client_grpc_options == [("limit", 2048)] and not adaptor.in_process
    assert adaptor.app_runner.is_stopped() == (False, 0)
    adaptor.app_runner.stop()
    assert adaptor.app_runner.is_stopped() == (True, 0)
    assert controller.num_rounds == 3


def test_federated_adaptor_factories_register_security_components_and_propagate_client_timeouts(monkeypatch):
    for module in (client_handler, server_handler):
        if module.tenseal_imported:
            monkeypatch.setattr(module, "decomposers", SimpleNamespace(register=Mock()))
    loader = Mock(spec=XGBDataLoader)
    engine = Mock()
    engine.get_component.return_value = loader
    ctx = Mock(spec=FLContext)
    ctx.get_engine.return_value = engine
    executor = fed_executor.FedXGBHistogramExecutor(
        "loader", per_msg_timeout=12, tx_timeout=45, model_file_name="weights.json"
    )
    adaptor = executor.get_adaptor(ctx)
    assert adaptor.per_msg_timeout == 12 and adaptor.tx_timeout == 45
    assert adaptor.app_runner._data_loader is loader
    assert adaptor.app_runner.model_file_name == "weights.json"
    registered = engine.add_component.call_args.args[1]
    assert isinstance(registered, client_handler.ClientSecurityHandler)
    controller = fed_controller.XGBFedController(2, 0, False, {"max_depth": 2})
    server = controller.get_adaptor(ctx)
    assert isinstance(engine.add_component.call_args.args[1], server_handler.ServerSecurityHandler)
    assert isinstance(server.app_runner, xgb_server_runner.XGBServerRunner)
    assert server.starter is None and server.process is None


def test_native_federated_server_runner_passes_world_size_and_port_without_opening_server(monkeypatch):
    start = Mock()
    monkeypatch.setattr(xgb_server_runner, "xgb_federated", SimpleNamespace(run_federated_server=start))
    runner = xgb_server_runner.XGBServerRunner()
    runner.run({Constant.RUNNER_CTX_PORT: 9000, Constant.RUNNER_CTX_WORLD_SIZE: 3})
    start.assert_called_once_with(n_workers=3, port=9000)
    assert runner.is_stopped() == (True, 0)
