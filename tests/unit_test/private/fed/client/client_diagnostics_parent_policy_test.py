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

import pytest

from nvflare.apis.client import Client
from nvflare.apis.fl_context import FLContext
from nvflare.fuel.f3.cellnet.fqcn import FQCN
from nvflare.private.admin_defs import Message
from nvflare.private.defs import InfoCollectorTopic, RequestHeader
from nvflare.private.fed.client import utils
from nvflare.private.fed.client.client_engine import ClientEngine
from nvflare.private.fed.client.info_coll_cmd import ClientInfoProcessor


@pytest.mark.parametrize(
    "topic", [InfoCollectorTopic.SHOW_STATS, InfoCollectorTopic.SHOW_ERRORS, InfoCollectorTopic.RESET_ERRORS, "unknown"]
)
@pytest.mark.parametrize("valid_result", [True, False])
def test_info_processor_routes_job_id_and_normalizes_nonmapping_engine_result(topic, valid_result):
    engine = Mock(spec=ClientEngine)
    engine.get_current_run_info.return_value = {"round": 3} if valid_result else None
    engine.get_errors.return_value = {"error": "failed"} if valid_result else "bad result"
    request = Message(topic, "")
    request.set_header(RequestHeader.JOB_ID, "job")
    processor = ClientInfoProcessor()
    assert set(processor.get_topics()) == {
        InfoCollectorTopic.SHOW_STATS,
        InfoCollectorTopic.SHOW_ERRORS,
        InfoCollectorTopic.RESET_ERRORS,
    }
    response = processor.process(request, engine)
    assert response.topic == "reply_" + topic
    payload = json.loads(response.body)
    if topic == InfoCollectorTopic.SHOW_STATS:
        engine.get_current_run_info.assert_called_once_with("job")
        assert payload == ({"round": 3} if valid_result else {})
    elif topic == InfoCollectorTopic.SHOW_ERRORS:
        engine.get_errors.assert_called_once_with("job")
        assert payload == ({"error": "failed"} if valid_result else {})
    elif topic == InfoCollectorTopic.RESET_ERRORS:
        engine.reset_errors.assert_called_once_with("job")
        assert payload == {"status": "OK"}
    else:
        assert payload == {"error": "invalid topic unknown"}
    with pytest.raises(TypeError, match="ClientEngineInternalSpec"):
        processor.process(request, object())


@pytest.mark.parametrize("mode", ["root", "no_engine", "missing_parent", "wrong_parent", "valid"])
def test_parent_routing_requires_real_registered_client_only_for_hierarchical_site(mode):
    context = FLContext()
    if mode == "root":
        assert utils.determine_parent_name({}) is None
        assert utils.determine_parent_fqcn({}, context) == FQCN.ROOT_SERVER
        return
    config = {"fqsn": "parent.child"}
    if mode == "no_engine":
        with pytest.raises(ValueError, match="missing engine"):
            utils.determine_parent_fqcn(config, context)
        return
    engine = Mock()
    parent = Client("parent", "token")
    parent.set_fqcn("parent.job")
    engine.get_client_from_name.return_value = (
        None if mode == "missing_parent" else object() if mode == "wrong_parent" else parent
    )
    context.get_engine = lambda: engine
    if mode == "valid":
        assert utils.determine_parent_fqcn(config, context) == "parent.job"
    else:
        with pytest.raises(
            RuntimeError, match="cannot find parent" if mode == "missing_parent" else "expect parent_client"
        ):
            utils.determine_parent_fqcn(config, context)
    engine.get_client_from_name.assert_called_once_with("parent")


def test_parent_config_rejects_nonmapping_input():
    with pytest.raises(ValueError, match="expect client_config to be dict"):
        utils.determine_parent_name([])
