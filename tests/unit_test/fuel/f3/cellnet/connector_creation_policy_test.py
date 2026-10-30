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

from unittest.mock import Mock

import pytest

from nvflare.fuel.f3.cellnet import connector_manager as subject
from nvflare.fuel.f3.cellnet.fqcn import FqcnInfo
from nvflare.fuel.f3.comm_error import CommError
from nvflare.fuel.f3.drivers.connector_info import Mode
from tests.timing_utils import isolate_time


@pytest.fixture
def manager():
    config = Mock()
    config.get_backbone_connection_generation.return_value = 2
    config.get_internal_connection_scheme.return_value = "tcp"
    config.get_adhoc_connection_scheme.return_value = "tcp"
    config.allow_adhoc_connections.return_value = True
    config.get_config.return_value = {}
    return subject.ConnectorManager(Mock(), True, config)


@pytest.mark.parametrize(
    "config",
    [{"internal": "invalid"}, {"internal": {"resources": {}}}, {"internal": {"scheme": "tcp", "resources": "bad"}}],
)
def test_connector_configuration_rejects_missing_scheme_and_wrong_shapes(config):
    with pytest.raises(subject.ConfigError):
        subject.ConnectorManager._validate_conn_config(config, "internal")


@pytest.mark.parametrize("generation", [[1, 3], 3, None])
def test_backbone_policy_respects_explicit_generation_filter(manager, generation):
    manager.comm_config = {"backbone": {"connect_generation": generation}}
    assert manager.should_connect_to_server(FqcnInfo("site-a"))
    assert manager.should_connect_to_server(FqcnInfo("site-a.job")) is (generation is None)
    assert manager.should_connect_to_server(FqcnInfo("site-a.job.rank")) is (generation is not None)


@pytest.mark.parametrize("failure", [None, "communication", "unexpected"])
def test_dynamic_listener_returns_connector_data_or_contains_creation_failure(manager, monkeypatch, failure):
    manager.communicator.start_listener.return_value = ("handle", "tcp://localhost:1234", {"param": "value"})
    if failure:
        manager.communicator.start_listener.side_effect = (
            CommError(CommError.ERROR, "listen failed") if failure == "communication" else ValueError("bad scheme")
        )
    clock = isolate_time(monkeypatch, subject)
    monkeypatch.setattr(clock, "sleep", Mock())
    result = manager.get_internal_listener()
    if failure:
        assert result is None
    else:
        assert result.handle == "handle"
        assert result.get_connection_url() == "tcp://localhost:1234"
        assert result.get_connection_params() == {"param": "value"}
        assert result.active is False
        clock.sleep.assert_called_once_with(0.5)
    manager.communicator.add_connector.assert_not_called()


@pytest.mark.parametrize("active", [False, True])
def test_explicit_external_url_uses_requested_mode_security_and_connection_resources(manager, active):
    manager.communicator.add_connector.return_value = ("handle", {"security": "tls"})
    if active:
        result = manager.get_external_connector("tcp://host:1234", False)
    else:
        result = manager.get_external_listener("tcp://host:1234", False)
    manager.communicator.add_connector.assert_called_once_with(
        "tcp://host:1234", Mode.ACTIVE if active else Mode.PASSIVE, True, None
    )
    assert result.active is active
    assert result.connect_url == "tcp://host:1234"


@pytest.mark.parametrize("active,internal,adhoc", [(True, False, False), (False, False, False), (False, True, True)])
def test_invalid_connector_topology_is_rejected_before_driver_invocation(manager, active, internal, adhoc):
    with pytest.raises(RuntimeError):
        manager._get_connector("", active, internal, adhoc, False)
    assert not manager.communicator.mock_calls


def test_disallowed_adhoc_listener_is_suppressed_without_transport_calls(manager):
    manager.adhoc_allowed = False
    assert manager.get_external_listener("", True) is None
    assert not manager.communicator.mock_calls
