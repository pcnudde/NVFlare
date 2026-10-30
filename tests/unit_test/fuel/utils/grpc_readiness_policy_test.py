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

from nvflare.fuel.utils import grpc_utils as subject


@pytest.mark.parametrize("probe_only", [True, False])
def test_readiness_probe_closes_only_probe_channel_and_forwards_budget(monkeypatch, probe_only):
    channel, ready = Mock(), Mock()
    sdk = SimpleNamespace(
        insecure_channel=Mock(return_value=channel),
        channel_ready_future=Mock(return_value=ready),
        FutureTimeoutError=subject.grpc.FutureTimeoutError,
    )
    monkeypatch.setattr(subject, "grpc", sdk)
    assert subject.create_channel("host:1234", [("limit", 3)], 0.5, probe_only) is (None if probe_only else channel)
    sdk.insecure_channel.assert_called_once_with("host:1234", options=[("limit", 3)])
    ready.result.assert_called_once_with(timeout=0.5)
    assert channel.close.call_count == int(probe_only)


def test_readiness_timeout_is_contextual_without_real_network_or_wait(monkeypatch):
    ready = Mock()
    ready.result.side_effect = subject.grpc.FutureTimeoutError()
    sdk = SimpleNamespace(
        insecure_channel=Mock(),
        channel_ready_future=Mock(return_value=ready),
        FutureTimeoutError=subject.grpc.FutureTimeoutError,
    )
    monkeypatch.setattr(subject, "grpc", sdk)
    with pytest.raises(RuntimeError, match="cannot connect to server after 0.5"):
        subject.create_channel("host:1234", [], 0.5, False)
    ready.result.assert_called_once_with(timeout=0.5)
