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

from nvflare.apis.analytix import AnalyticsDataType
from nvflare.client.cell import api as module
from nvflare.client.cell.attach_session import AttachTrainerSession
from nvflare.client.cell.bootstrap import BootstrapKey
from nvflare.fuel.f3.streaming.transfer_progress import TransferProgressState


@pytest.fixture
def api(tmp_path):
    profile = tmp_path / "profile.json"
    profile.write_text(
        json.dumps(
            {
                BootstrapKey.SCHEMA_VERSION: 1,
                BootstrapKey.EXECUTION_MODE: "external_process",
                BootstrapKey.CJ_FQCN: "site.job",
                BootstrapKey.TRAINER_FQCN: "site.job.trainer",
                BootstrapKey.JOB_ID: "job",
                BootstrapKey.SITE_NAME: "site",
                BootstrapKey.CONNECT_URL: "tcp://127.0.0.1:9000",
                BootstrapKey.LAUNCH_TOKEN: "token",
                BootstrapKey.SECURE_MODE: False,
            }
        )
    )
    return module.CellClientAPI(str(profile))


@pytest.mark.parametrize(
    "ending,reason",
    [
        ("_abort", "session aborted while serving result"),
        ("_closed", "session closed while serving result"),
        (None, "ended without a terminal outcome"),
    ],
)
def test_unresolved_result_transfer_cannot_be_mistaken_for_success_after_session_end(api, ending, reason):
    waiter = Mock(transaction_id="tx", outcome=None)
    waiter.wait.return_value = None
    waiter.done.return_value = ending is None
    if ending:
        setattr(api, ending, True)
    with pytest.raises(module.TrainerSessionError, match=reason):
        api._wait_for_result_transfers([waiter])
    waiter.wait.assert_called_once()


def test_transfer_cleanup_error_does_not_prevent_remaining_owned_transactions_from_cleanup(api, monkeypatch):
    delete = Mock(side_effect=[OSError("cleanup failed"), None])
    monkeypatch.setattr(module, "DownloadService", SimpleNamespace(delete_transaction=delete))
    api._delete_result_transfers([SimpleNamespace(transaction_id="one"), SimpleNamespace(transaction_id="two")])
    assert [call.args[0] for call in delete.call_args_list] == ["one", "two"]


def test_loop_guard_contains_session_error_and_closes_api_without_exposing_it(api, monkeypatch):
    api._is_control_rank = True
    api._rank = "0"
    monkeypatch.setattr(api, "_receive_internal", Mock(side_effect=module.TrainerSessionError("session ended")))
    close = Mock()
    monkeypatch.setattr(api, "shutdown", close)
    assert not api.is_running()
    close.assert_called_once_with()


def test_closed_receive_and_failed_analytics_transport_have_no_new_task_or_session_side_effect(api):
    api._closed = True
    assert api._await_task(timeout=None) is None
    assert not api._abort_if_cj_timed_out()
    api._closed = False
    api._is_control_rank = True
    api._rank = "0"
    api._cell = Mock()
    api._cell.fire_and_forget.side_effect = OSError("disconnected")
    api.logger = Mock()
    api.log("loss", 1.0, AnalyticsDataType.SCALAR)
    api.logger.warning.assert_called_once()
    assert not api._abort and api._current_task is None


@pytest.mark.parametrize(
    "finished,session_end,expected", [(True, None, None), (False, "session closed", "session closed")]
)
def test_ambiguous_result_resolution_uses_terminal_transfer_or_session_authority_without_waiting(
    finished, session_end, expected
):
    api = SimpleNamespace(
        _config={BootstrapKey.ATTACH_ID: "trainer", BootstrapKey.SITE_NAME: "site"},
        _result_publication_end_reason=lambda: session_end,
    )
    session = AttachTrainerSession(api)
    session._accepted_result_attempt_from_status = Mock(return_value=None)
    waiter = Mock()
    waiter.done.return_value = finished
    waiter.wait.return_value = SimpleNamespace(status=TransferProgressState.FAILED)
    if expected:
        with pytest.raises(module.TrainerSessionError, match=expected):
            session._resolve_ambiguous_result("task", "result", {"attempt": [waiter]})
    else:
        assert session._resolve_ambiguous_result("task", "result", {"attempt": [waiter]}) is None
    session._accepted_result_attempt_from_status.assert_called_once_with("task", "result")
