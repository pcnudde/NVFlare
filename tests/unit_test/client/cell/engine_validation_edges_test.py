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

import numpy as np
import pytest

from nvflare.apis.analytix import AnalyticsDataType
from nvflare.app_common.abstract.fl_model import FLModel, ParamsType
from nvflare.client.cell import api as cell
from nvflare.client.cell.bootstrap import BootstrapKey
from nvflare.client.config import ConfigKey, ExchangeFormat
from nvflare.client.in_process.adapters import TaskInProcessAdapter
from nvflare.client.in_process.api import InProcessClientAPI
from nvflare.fuel.f3.cellnet.defs import MessageHeaderKey, ReturnCode
from nvflare.fuel.f3.cellnet.utils import new_cell_message


@pytest.fixture
def local_api():
    adapter = TaskInProcessAdapter(on_result=Mock(return_value="stored"))
    api = InProcessClientAPI({ConfigKey.TASK_NAME: "train"}, adapter=adapter)
    api.init(
        rank="0", config={"custom": "kept", ConfigKey.TASK_EXCHANGE: {ConfigKey.EXCHANGE_FORMAT: ExchangeFormat.NUMPY}}
    )
    try:
        yield api
    finally:
        api.shutdown()


@pytest.mark.parametrize("name", ["get_task_name", "is_train", "is_evaluate", "is_submit_model", "log"])
def test_noncontrol_inprocess_rank_cannot_read_task_state_or_publish_analytics(local_api, name):
    local_api.rank = "1"
    with pytest.raises(RuntimeError, match="only rank 0"):
        getattr(local_api, name)("loss", 2, AnalyticsDataType.SCALAR) if name == "log" else getattr(local_api, name)()


def test_inprocess_configuration_merge_and_invalid_result_fail_before_publication(local_api):
    assert local_api.client_config.config["custom"] == "kept"
    assert local_api.client_config.config[ConfigKey.TASK_NAME] == "train"
    with pytest.raises(ValueError, match="dictionary"):
        local_api.prepare_client_config([])
    local_api.receive_called = True
    with pytest.raises(RuntimeError, match="does not have either params or metrics"):
        local_api.send(FLModel())
    local_api._adapter._on_result.assert_not_called()


@pytest.fixture
def cell_api(tmp_path):
    path = tmp_path / "profile.json"
    path.write_text(
        json.dumps(
            {
                BootstrapKey.SCHEMA_VERSION: 1,
                BootstrapKey.EXECUTION_MODE: "external_process",
                BootstrapKey.CJ_FQCN: "site.job",
                BootstrapKey.TRAINER_FQCN: "site.job.trainer",
                BootstrapKey.JOB_ID: "job",
                BootstrapKey.SITE_NAME: "site",
                BootstrapKey.CONNECT_URL: "tcp://127.0.0.1:9000",
                BootstrapKey.LAUNCH_TOKEN: "unit-token",
                BootstrapKey.SECURE_MODE: False,
            }
        )
    )
    return cell.CellClientAPI(str(path))


@pytest.mark.parametrize("engine", ["cell", "inprocess"])
def test_param_diff_requires_original_and_matching_keys_before_marking_diff(cell_api, local_api, engine):
    api = cell_api if engine == "cell" else local_api
    if engine == "cell":
        api._task_exchange = {ConfigKey.EXCHANGE_FORMAT: ExchangeFormat.RAW}
    else:
        api.client_config.config[ConfigKey.TASK_EXCHANGE] = {ConfigKey.EXCHANGE_FORMAT: ExchangeFormat.RAW}
    with pytest.raises(RuntimeError, match="no received model"):
        api._prepare_param_diff(FLModel(params={"w": np.array([2.0])}))
    original = FLModel(params={"w": np.array([1.0])})
    if engine == "cell":
        api._fl_model = original
    else:
        api.fl_model = original
    result = api._prepare_param_diff(FLModel(params={"w": np.array([4.0])}))
    assert result.params_type == ParamsType.DIFF
    np.testing.assert_array_equal(result.params["w"], [3])
    with pytest.raises(RuntimeError, match="params diff function failed"):
        api._prepare_param_diff(FLModel(params={"other": np.array([4.0])}))
    if engine == "cell":
        api._task_exchange[ConfigKey.EXCHANGE_FORMAT] = "unsupported"
    else:
        api.client_config.config[ConfigKey.TASK_EXCHANGE][ConfigKey.EXCHANGE_FORMAT] = "unsupported"
    with pytest.raises(RuntimeError, match="no default params diff"):
        api._prepare_param_diff(FLModel(params={"w": np.array([4.0])}))


def test_cell_session_security_policy_cannot_change_after_authentication(cell_api):
    with pytest.raises(cell.TrainerSessionError, match="must be a bool"):
        cell_api._install_site_auth_headers(1)
    cell_api._install_site_auth_headers(False)
    cell_api._install_site_auth_headers(False)
    with pytest.raises(cell.TrainerSessionError, match="changed after authentication"):
        cell_api._install_site_auth_headers(True)
    cell_api._session_security_configured = False
    cell_api._secure_mode = True
    with pytest.raises(cell.TrainerSessionError, match="no site auth token"):
        cell_api._install_site_auth_headers(True, "NA", "signature")
    assert not cell_api._session_security_configured


def test_cell_send_requires_admitted_task_and_nonempty_model(cell_api):
    cell_api._is_control_rank = True
    cell_api._receive_called = True
    with pytest.raises(cell.TrainerSessionError, match="no current task"):
        cell_api.send(FLModel(params={"w": [1]}))
    cell_api._current_task = {"task_id": "task"}
    with pytest.raises(RuntimeError, match="does not have either params or metrics"):
        cell_api.send(FLModel())
    cell_api._abort, cell_api._abort_reason = True, "owner failed"
    with pytest.raises(cell.TrainerSessionError, match="owner failed"):
        cell_api._check_session_alive()
    assert "owner failed" in cell_api._result_publication_end_reason()
    cell_api._abort, cell_api._closed = False, True
    assert cell_api._result_publication_end_reason() == "session closed"
    cell_api.log("ignored", 1, AnalyticsDataType.SCALAR)
    assert cell_api._normalize_result_receiver_ids("site.job") == ("site.job",)
    with pytest.raises(cell.TrainerSessionError, match="cell-level failure"):
        cell_api._check_result_accepted(new_cell_message({MessageHeaderKey.RETURN_CODE: ReturnCode.TIMEOUT}, {}))
