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

import pytest

pytest.importorskip("private_set_intersection.python")

from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.apis.signal import Signal
from nvflare.app_common.app_constant import PSIConst
from nvflare.app_common.psi.psi_spec import PSI
from nvflare.app_opt.psi.dh_psi.dh_psi_task_handler import DhPSITaskHandler


def _handler(items):
    local = MagicMock(spec=PSI)
    local.load_items.return_value = items
    engine = MagicMock()
    engine.get_component.return_value = local
    context = FLContext()
    context.get_engine = lambda: engine
    context.get_identity_name = lambda: "site"
    subject = DhPSITaskHandler("local")
    subject.initialize(context)
    local.initialize.assert_called_once_with(context)
    return subject, local, context


def _execute(subject, context, stage, **payload):
    message = Shareable(payload)
    message[PSIConst.TASK_KEY] = stage
    return subject.execute_task(PSIConst.TASK, message, context, Signal())


def _prepare(subject, context):
    return _execute(subject, context, PSIConst.TASK_PREPARE, **{PSIConst.BLOOM_FILTER_FPR: 1e-9})


def test_native_psi_handler_roundtrip_rotates_keys_and_saves_ordered_intersection():
    server, server_local, server_context = _handler(["a", "c", "e"])
    client, client_local, client_context = _handler(["b", "c", "e"])
    assert _prepare(server, server_context)[PSIConst.ITEMS_SIZE] == 3
    assert _prepare(client, client_context)[PSIConst.ITEMS_SIZE] == 3
    previous_client, previous_server = server.psi_client, server.psi_server
    setup = _execute(server, server_context, PSIConst.TASK_SETUP, **{PSIConst.ITEMS_SIZE: 3})
    assert server.psi_client is not previous_client and server.psi_server is not previous_server
    request = _execute(client, client_context, PSIConst.TASK_REQUEST, **{PSIConst.SETUP_MSG: setup[PSIConst.SETUP_MSG]})
    response = _execute(
        server, server_context, PSIConst.TASK_RESPONSE, **{PSIConst.REQUEST_MSG: request[PSIConst.REQUEST_MSG]}
    )
    result = _execute(
        client, client_context, PSIConst.TASK_INTERSECT, **{PSIConst.RESPONSE_MSG: response[PSIConst.RESPONSE_MSG]}
    )
    assert result[PSIConst.ITEMS_SIZE] == 2
    assert client.intersects == ["c", "e"]
    assert client.get_items() == ["c", "e"]
    client_local.save.assert_called_once_with(["c", "e"])
    client_local.load_items.assert_called_once_with()
    server_local.load_items.assert_called_once_with()


def test_setup_and_response_batches_keep_each_client_protocol_message_separate():
    server, _, context = _handler(["a", "b", "c"])
    _prepare(server, context)
    setup = _execute(server, context, PSIConst.TASK_SETUP, **{PSIConst.ITEMS_SIZE_SET: [2, 3]})[PSIConst.SETUP_MSG]
    assert set(setup) == {"2", "3"}
    assert all(isinstance(value, bytes) and value for value in setup.values())
    clients = {}
    requests = {}
    for name, items in {"first": ["b", "x"], "second": ["a", "c", "y"]}.items():
        client, local, client_context = _handler(items)
        _prepare(client, client_context)
        request = _execute(
            client, client_context, PSIConst.TASK_REQUEST, **{PSIConst.SETUP_MSG: setup[str(len(items))]}
        )
        requests[name] = request[PSIConst.REQUEST_MSG]
        clients[name] = (client, local, client_context)
    responses = _execute(server, context, PSIConst.TASK_RESPONSE, **{PSIConst.REQUEST_MSG_SET: requests})[
        PSIConst.RESPONSE_MSG
    ]
    assert set(responses) == set(requests)
    for name, expected in [("first", ["b"]), ("second", ["a", "c"])]:
        client, local, client_context = clients[name]
        client.calculate_intersection(Shareable({PSIConst.RESPONSE_MSG: responses[name]}))
        local.save.assert_called_once_with(expected)


def test_psi_handler_rejects_duplicates_empty_setup_unknown_task_and_missing_request():
    duplicate, _, _ = _handler(["a", "a"])
    with pytest.raises(ValueError, match="duplicates"):
        duplicate.get_items()
    empty, _, context = _handler([])
    with pytest.raises(RuntimeError, match="doesn't have any items"):
        empty.setup(Shareable({PSIConst.ITEMS_SIZE: 1}), "site")
    with pytest.raises(RuntimeError, match="TASK_UNKNOWN"):
        empty.execute_task("unknown", Shareable(), context, Signal())
    with pytest.raises(ValueError, match="Required PSI Message"):
        empty.process_request(Shareable())
