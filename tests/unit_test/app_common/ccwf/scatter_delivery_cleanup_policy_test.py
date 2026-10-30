# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.app_common.ccwf import swarm_client_ctl as subject
from nvflare.app_common.ccwf.common import Constant


@pytest.mark.parametrize("failure", ["local", "remote"])
def test_swarm_stops_scatter_when_local_queue_or_remote_delivery_rejects(monkeypatch, failure):
    controller = subject.SwarmClientController()
    controller.me, controller.trainers, controller.aggrs = "one", ["one"], ["two"]
    local = Mock(return_value=failure != "local")
    remote = Mock(return_value=False)
    monkeypatch.setattr(controller, "set_learn_task", local)
    monkeypatch.setattr(controller, "send_learn_task", remote)
    data = Shareable({"model": [1]})
    assert controller._scatter(data, 3, FLContext()) is False
    local.assert_called_once()
    assert local.call_args.kwargs["task_data"] is not data
    assert data.get_header(Constant.AGGREGATOR) == "two"
    if failure == "local":
        remote.assert_not_called()
    else:
        remote.assert_called_once()
        assert remote.call_args.kwargs["targets"] == ["two"]


@pytest.mark.parametrize("client", [SimpleNamespace(), SimpleNamespace(get_fqcn=lambda: None)])
def test_missing_client_fqcn_cannot_be_stamped_as_result_receiver(client):
    controller = subject.SwarmClientController()
    controller.engine = SimpleNamespace(get_client_from_name=lambda name: client)
    assert controller._get_aggregation_client_fqcn("two", FLContext()) is None


def test_offload_cleanup_still_attempts_root_removal_when_download_drain_raises(tmp_path, monkeypatch):
    controller, context = subject.SwarmClientController(), FLContext()
    root = tmp_path / "tensors"
    root.mkdir()
    (root / "tensor.bin").write_bytes(b"owned")
    controller._tensor_disk_offload_root_dir = str(root)
    monkeypatch.setattr(subject, "_cleanup_active_disk_tensor_downloads", Mock())
    monkeypatch.setattr(
        controller, "_drain_tensor_disk_offload_threads", Mock(side_effect=RuntimeError("drain failed"))
    )
    warning = Mock()
    monkeypatch.setattr(controller, "log_warning", warning)
    controller._cleanup_tensor_disk_offload(context)
    assert not root.exists() and controller._tensor_disk_offload_root_dir is None
    assert "failed to clean up" in warning.call_args.args[1]
