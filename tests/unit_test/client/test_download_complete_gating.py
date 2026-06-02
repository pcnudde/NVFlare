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

"""Tests for reverse result download completion gating.

With reverse PASS_THROUGH active, the CJ can ACK a result message before the
server has downloaded tensor payloads from the subprocess DownloadService. These
tests cover the subprocess wait contract: known download transactions must reach
terminal state before shutdown, metric-only results must not wait, and legacy
fallback paths remain bounded by download_complete_timeout.
"""

import threading
import time
from unittest.mock import MagicMock, patch

import pytest

from nvflare.client.flare_agent import FlareAgent, _TaskContext
from nvflare.fuel.f3.streaming.download_service import TransactionDoneStatus
from nvflare.fuel.utils.fobs import FOBSContextKey
from nvflare.fuel.utils.fobs.decomposers.via_downloader import (
    DownloadTransactionInfo,
    _tls,
    clear_download_initiated,
    clear_download_transactions,
)

# ---------------------------------------------------------------------------
# Module-level fixture: prevent os._exit() from killing the pytest worker.
# _do_submit_result() calls os._exit(0) after the download gate so that the
# subprocess can bypass non-daemon thread cleanup.  In unit tests we patch it
# to a no-op so the worker process survives.
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _no_os_exit(monkeypatch):
    monkeypatch.setattr("nvflare.client.flare_agent.os._exit", lambda code: None)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_cell_pipe(pass_through_on_send: bool = True):
    """Return a mock CellPipe with a trackable cell.update_fobs_context."""
    from nvflare.fuel.utils.pipe.cell_pipe import CellPipe

    pipe = MagicMock(spec=CellPipe)
    pipe.pass_through_on_send = pass_through_on_send
    pipe.closed = False
    # cell.update_fobs_context() captures the props dict for inspection
    pipe.cell = MagicMock()
    return pipe


def _make_non_cell_pipe():
    """Return a mock FilePipe (not a CellPipe subclass)."""
    from nvflare.fuel.utils.pipe.file_pipe import FilePipe

    pipe = MagicMock(spec=FilePipe)
    return pipe


def _make_agent(pipe, download_complete_timeout: float = 5.0, streaming_idle_timeout: float = 600.0):
    """Return a FlareAgent stub backed by the given pipe.

    Bypasses __init__ network setup by constructing manually.
    """
    agent = FlareAgent.__new__(FlareAgent)
    agent.logger = MagicMock()
    agent.pipe = pipe
    agent.submit_result_timeout = 30.0
    agent._download_complete_timeout = download_complete_timeout
    agent._streaming_idle_timeout = streaming_idle_timeout
    agent._close_pipe = False
    agent._close_metric_pipe = False
    agent.task_lock = threading.Lock()
    agent.asked_to_stop = False
    agent.current_task = None
    agent._launch_once = False  # direct os._exit(0) path; patched to no-op by _no_os_exit fixture

    # pipe_handler.send_to_peer returns True by default
    agent.pipe_handler = MagicMock()
    agent.pipe_handler.send_to_peer.return_value = True
    agent.pipe_handler.asked_to_stop = False

    return agent


def _make_task_ctx():
    return _TaskContext(task_id="tid-1", task_name="train", msg_id="msg-1")


# ---------------------------------------------------------------------------
# 1-7: FlareAgent._do_submit_result()
# ---------------------------------------------------------------------------


class TestDoSubmitResultGating:
    """FlareAgent._do_submit_result() gating behaviour."""

    def _patch_shareable(self, agent):
        agent.task_result_to_shareable = MagicMock(return_value=MagicMock())

    def setup_method(self):
        clear_download_initiated()
        clear_download_transactions()

    def test_download_complete_cb_registered_before_send(self):
        """DOWNLOAD_COMPLETE_CB must be in cell FOBS context before send_to_peer() is called."""
        pipe = _make_cell_pipe(pass_through_on_send=True)
        agent = _make_agent(pipe)
        self._patch_shareable(agent)

        registered_before_send = {}

        def capture_on_send(reply, timeout):
            # Inspect what was registered on the cell at send time
            for c in pipe.cell.update_fobs_context.call_args_list:
                props = c[0][0]
                if (
                    FOBSContextKey.DOWNLOAD_COMPLETE_CB in props
                    and props[FOBSContextKey.DOWNLOAD_COMPLETE_CB] is not None
                ):
                    registered_before_send["cb"] = props[FOBSContextKey.DOWNLOAD_COMPLETE_CB]
            # Simulate _finalize_download_tx() setting the flag (tensors in result)
            _tls.download_initiated = True
            # Fire the callback to unblock the wait
            if registered_before_send.get("cb"):
                registered_before_send["cb"]("tid", "FINISHED", [])
            return True

        agent.pipe_handler.send_to_peer.side_effect = capture_on_send

        agent._do_submit_result(_make_task_ctx(), None, "OK")

        assert "cb" in registered_before_send, "DOWNLOAD_COMPLETE_CB must be registered before send_to_peer()"
        assert callable(registered_before_send["cb"])

    def test_waits_for_cb_and_returns_true(self):
        """Returns True when DOWNLOAD_COMPLETE_CB fires within timeout."""
        pipe = _make_cell_pipe(pass_through_on_send=True)
        agent = _make_agent(pipe, download_complete_timeout=5.0)
        self._patch_shareable(agent)

        # Fire the callback from a background thread shortly after send_to_peer()
        registered_cb = {}

        def fire_cb_on_send(reply, timeout):
            for c in pipe.cell.update_fobs_context.call_args_list:
                props = c[0][0]
                if (
                    FOBSContextKey.DOWNLOAD_COMPLETE_CB in props
                    and props[FOBSContextKey.DOWNLOAD_COMPLETE_CB] is not None
                ):
                    registered_cb["cb"] = props[FOBSContextKey.DOWNLOAD_COMPLETE_CB]
            # Simulate _finalize_download_tx() setting the flag (tensors in result)
            _tls.download_initiated = True

            def _fire():
                if registered_cb.get("cb"):
                    registered_cb["cb"]("tid", "FINISHED", [])

            threading.Thread(target=_fire, daemon=True).start()
            return True

        agent.pipe_handler.send_to_peer.side_effect = fire_cb_on_send

        result = agent._do_submit_result(_make_task_ctx(), None, "OK")

        assert result is True

    def test_transaction_timeout_returns_false(self):
        pipe = _make_cell_pipe(pass_through_on_send=True)
        agent = _make_agent(pipe, download_complete_timeout=5.0)
        self._patch_shareable(agent)

        registered_cb = {}

        def fire_timeout_on_send(reply, timeout):
            _tls.download_initiated = True
            _tls.download_transactions = [
                DownloadTransactionInfo(tx_id="tx-timeout", ref_ids=("ref-1",), created_time=1.0)
            ]
            for c in pipe.cell.update_fobs_context.call_args_list:
                props = c[0][0]
                if (
                    FOBSContextKey.DOWNLOAD_COMPLETE_CB in props
                    and props[FOBSContextKey.DOWNLOAD_COMPLETE_CB] is not None
                ):
                    registered_cb["cb"] = props[FOBSContextKey.DOWNLOAD_COMPLETE_CB]
            if registered_cb.get("cb"):
                registered_cb["cb"]("tx-timeout", TransactionDoneStatus.TIMEOUT, [])
            return True

        agent.pipe_handler.send_to_peer.side_effect = fire_timeout_on_send

        result = agent._do_submit_result(_make_task_ctx(), None, "OK")

        assert result is False

    def test_transaction_wait_abort_deletes_known_transactions(self):
        pipe = _make_cell_pipe(pass_through_on_send=True)
        agent = _make_agent(pipe, download_complete_timeout=5.0)
        self._patch_shareable(agent)

        def send_then_stop(reply, timeout):
            _tls.download_initiated = True
            _tls.download_transactions = [
                DownloadTransactionInfo(tx_id="tx-abort", ref_ids=("ref-1",), created_time=1.0)
            ]
            agent.asked_to_stop = True
            return True

        agent.pipe_handler.send_to_peer.side_effect = send_then_stop

        with (
            patch("nvflare.client.flare_agent.DownloadService.finalize_transaction_if_finished", return_value=False),
            patch("nvflare.client.flare_agent.DownloadService.delete_transaction") as delete_tx,
        ):
            result = agent._do_submit_result(_make_task_ctx(), None, "OK")

        assert result is False
        delete_tx.assert_called_once_with("tx-abort")

    def test_transaction_wait_stop_after_finish_returns_true_without_delete(self):
        pipe = _make_cell_pipe(pass_through_on_send=True)
        agent = _make_agent(pipe, download_complete_timeout=5.0)
        self._patch_shareable(agent)

        registered_cb = {}

        def send_then_stop(reply, timeout):
            _tls.download_initiated = True
            _tls.download_transactions = [
                DownloadTransactionInfo(tx_id="tx-finished-on-stop", ref_ids=("ref-1",), created_time=1.0)
            ]
            for c in pipe.cell.update_fobs_context.call_args_list:
                props = c[0][0]
                if (
                    FOBSContextKey.DOWNLOAD_COMPLETE_CB in props
                    and props[FOBSContextKey.DOWNLOAD_COMPLETE_CB] is not None
                ):
                    registered_cb["cb"] = props[FOBSContextKey.DOWNLOAD_COMPLETE_CB]
            agent.asked_to_stop = True
            return True

        def finalize_if_finished(tx_id):
            registered_cb["cb"](tx_id, TransactionDoneStatus.FINISHED, [])
            return True

        agent.pipe_handler.send_to_peer.side_effect = send_then_stop

        with (
            patch(
                "nvflare.client.flare_agent.DownloadService.finalize_transaction_if_finished",
                side_effect=finalize_if_finished,
            ) as finalize_tx,
            patch("nvflare.client.flare_agent.DownloadService.delete_transaction") as delete_tx,
        ):
            result = agent._do_submit_result(_make_task_ctx(), None, "OK")

        assert result is True
        finalize_tx.assert_called_once_with("tx-finished-on-stop")
        delete_tx.assert_not_called()

    def test_transaction_finished_wait_returns_true(self):
        pipe = _make_cell_pipe(pass_through_on_send=True)
        agent = _make_agent(pipe, download_complete_timeout=5.0)
        self._patch_shareable(agent)

        registered_cb = {}

        def fire_finished_on_send(reply, timeout):
            _tls.download_initiated = True
            _tls.download_transactions = [
                DownloadTransactionInfo(tx_id="tx-finished", ref_ids=("ref-1",), created_time=1.0)
            ]
            for c in pipe.cell.update_fobs_context.call_args_list:
                props = c[0][0]
                if (
                    FOBSContextKey.DOWNLOAD_COMPLETE_CB in props
                    and props[FOBSContextKey.DOWNLOAD_COMPLETE_CB] is not None
                ):
                    registered_cb["cb"] = props[FOBSContextKey.DOWNLOAD_COMPLETE_CB]
            if registered_cb.get("cb"):
                registered_cb["cb"]("tx-finished", TransactionDoneStatus.FINISHED, [])
            return True

        agent.pipe_handler.send_to_peer.side_effect = fire_finished_on_send

        result = agent._do_submit_result(_make_task_ctx(), None, "OK")

        assert result is True

    def test_transaction_wait_polls_finished_transaction_without_monitor_tick(self):
        pipe = _make_cell_pipe(pass_through_on_send=True)
        agent = _make_agent(pipe, download_complete_timeout=5.0)
        self._patch_shareable(agent)

        registered_cb = {}

        def send_with_finished_transaction(reply, timeout):
            _tls.download_initiated = True
            _tls.download_transactions = [
                DownloadTransactionInfo(tx_id="tx-finished-before-monitor", ref_ids=("ref-1",), created_time=1.0)
            ]
            for c in pipe.cell.update_fobs_context.call_args_list:
                props = c[0][0]
                if (
                    FOBSContextKey.DOWNLOAD_COMPLETE_CB in props
                    and props[FOBSContextKey.DOWNLOAD_COMPLETE_CB] is not None
                ):
                    registered_cb["cb"] = props[FOBSContextKey.DOWNLOAD_COMPLETE_CB]
            return True

        def finalize_if_finished(tx_id):
            registered_cb["cb"](tx_id, TransactionDoneStatus.FINISHED, [])
            return True

        agent.pipe_handler.send_to_peer.side_effect = send_with_finished_transaction

        with patch(
            "nvflare.client.flare_agent.DownloadService.finalize_transaction_if_finished",
            side_effect=finalize_if_finished,
        ) as finalize_tx:
            result = agent._do_submit_result(_make_task_ctx(), None, "OK")

        assert result is True
        finalize_tx.assert_called_once_with("tx-finished-before-monitor")

    def test_waits_for_all_download_transactions_before_returning(self):
        pipe = _make_cell_pipe(pass_through_on_send=True)
        agent = _make_agent(pipe, download_complete_timeout=5.0)
        self._patch_shareable(agent)

        registered_cb = {}
        delayed_thread = None

        def fire_one_then_delay_sibling_on_send(reply, timeout):
            nonlocal delayed_thread
            _tls.download_initiated = True
            _tls.download_transactions = [
                DownloadTransactionInfo(tx_id="tx-1", ref_ids=("ref-1",), created_time=1.0),
                DownloadTransactionInfo(tx_id="tx-2", ref_ids=("ref-2",), created_time=1.0),
            ]
            for c in pipe.cell.update_fobs_context.call_args_list:
                props = c[0][0]
                if (
                    FOBSContextKey.DOWNLOAD_COMPLETE_CB in props
                    and props[FOBSContextKey.DOWNLOAD_COMPLETE_CB] is not None
                ):
                    registered_cb["cb"] = props[FOBSContextKey.DOWNLOAD_COMPLETE_CB]
            registered_cb["cb"]("tx-1", TransactionDoneStatus.FINISHED, [])

            def _finish_later():
                time.sleep(0.05)
                registered_cb["cb"]("tx-2", TransactionDoneStatus.FINISHED, [])

            delayed_thread = threading.Thread(target=_finish_later, daemon=True)
            delayed_thread.start()
            return True

        agent.pipe_handler.send_to_peer.side_effect = fire_one_then_delay_sibling_on_send

        start = time.monotonic()
        result = agent._do_submit_result(_make_task_ctx(), None, "OK")
        elapsed = time.monotonic() - start
        delayed_thread.join(timeout=1.0)

        assert result is True
        assert elapsed >= 0.04, "must not return after only the first sibling transaction finishes"

    def test_failed_sibling_transaction_returns_false_and_deletes_pending_transactions(self):
        pipe = _make_cell_pipe(pass_through_on_send=True)
        agent = _make_agent(pipe, download_complete_timeout=5.0)
        self._patch_shareable(agent)

        registered_cb = {}

        def fire_failed_sibling_on_send(reply, timeout):
            _tls.download_initiated = True
            _tls.download_transactions = [
                DownloadTransactionInfo(tx_id="tx-1", ref_ids=("ref-1",), created_time=1.0),
                DownloadTransactionInfo(tx_id="tx-2", ref_ids=("ref-2",), created_time=1.0),
                DownloadTransactionInfo(tx_id="tx-3", ref_ids=("ref-3",), created_time=1.0),
            ]
            for c in pipe.cell.update_fobs_context.call_args_list:
                props = c[0][0]
                if (
                    FOBSContextKey.DOWNLOAD_COMPLETE_CB in props
                    and props[FOBSContextKey.DOWNLOAD_COMPLETE_CB] is not None
                ):
                    registered_cb["cb"] = props[FOBSContextKey.DOWNLOAD_COMPLETE_CB]
            registered_cb["cb"]("tx-1", TransactionDoneStatus.FINISHED, [])
            registered_cb["cb"]("tx-2", TransactionDoneStatus.TIMEOUT, [])
            return True

        agent.pipe_handler.send_to_peer.side_effect = fire_failed_sibling_on_send

        with patch("nvflare.client.flare_agent.DownloadService.delete_transaction") as delete_tx:
            result = agent._do_submit_result(_make_task_ctx(), None, "OK")

        assert result is False
        delete_tx.assert_called_once_with("tx-3")
        warning_msgs = [call[0][0] for call in agent.logger.warning.call_args_list]
        assert any("tx-2" in msg and TransactionDoneStatus.TIMEOUT in msg for msg in warning_msgs)

    def test_ignores_unknown_transaction_callback_until_expected_transaction_finishes(self):
        pipe = _make_cell_pipe(pass_through_on_send=True)
        agent = _make_agent(pipe, download_complete_timeout=5.0)
        self._patch_shareable(agent)

        registered_cb = {}
        delayed_thread = None

        def fire_unknown_then_expected_on_send(reply, timeout):
            nonlocal delayed_thread
            _tls.download_initiated = True
            _tls.download_transactions = [
                DownloadTransactionInfo(tx_id="tx-expected", ref_ids=("ref-1",), created_time=1.0)
            ]
            for c in pipe.cell.update_fobs_context.call_args_list:
                props = c[0][0]
                if (
                    FOBSContextKey.DOWNLOAD_COMPLETE_CB in props
                    and props[FOBSContextKey.DOWNLOAD_COMPLETE_CB] is not None
                ):
                    registered_cb["cb"] = props[FOBSContextKey.DOWNLOAD_COMPLETE_CB]
            registered_cb["cb"]("tx-other", TransactionDoneStatus.FINISHED, [])

            def _finish_expected():
                time.sleep(0.05)
                registered_cb["cb"]("tx-expected", TransactionDoneStatus.FINISHED, [])

            delayed_thread = threading.Thread(target=_finish_expected, daemon=True)
            delayed_thread.start()
            return True

        agent.pipe_handler.send_to_peer.side_effect = fire_unknown_then_expected_on_send

        start = time.monotonic()
        result = agent._do_submit_result(_make_task_ctx(), None, "OK")
        elapsed = time.monotonic() - start
        delayed_thread.join(timeout=1.0)

        assert result is True
        assert elapsed >= 0.04, "unknown completion callbacks must not satisfy the expected transaction"

    def test_reverse_transaction_ttl_uses_streaming_idle_timeout(self):
        pipe = _make_cell_pipe(pass_through_on_send=True)
        agent = _make_agent(pipe, download_complete_timeout=1800.0, streaming_idle_timeout=777.0)
        self._patch_shareable(agent)

        registered_cb = {}

        def fire_finished_on_send(reply, timeout):
            assert reply._dl_ttl == 777.0
            _tls.download_initiated = True
            _tls.download_transactions = [
                DownloadTransactionInfo(tx_id="tx-finished", ref_ids=("ref-1",), created_time=1.0)
            ]
            for c in pipe.cell.update_fobs_context.call_args_list:
                props = c[0][0]
                if (
                    FOBSContextKey.DOWNLOAD_COMPLETE_CB in props
                    and props[FOBSContextKey.DOWNLOAD_COMPLETE_CB] is not None
                ):
                    registered_cb["cb"] = props[FOBSContextKey.DOWNLOAD_COMPLETE_CB]
            registered_cb["cb"]("tx-finished", TransactionDoneStatus.FINISHED, [])
            return True

        agent.pipe_handler.send_to_peer.side_effect = fire_finished_on_send

        assert agent._do_submit_result(_make_task_ctx(), None, "OK") is True

    def test_timeout_logs_warning_and_returns_true(self):
        """When DOWNLOAD_COMPLETE_CB never fires, a warning is logged and True is returned.

        Subprocess exit is non-fatal even on timeout; the server may still be
        downloading, but blocking forever would hang the entire training job.
        """
        pipe = _make_cell_pipe(pass_through_on_send=True)
        agent = _make_agent(pipe, download_complete_timeout=0.05)  # very short timeout
        self._patch_shareable(agent)

        # Simulate _finalize_download_tx() setting the flag (tensors in result)
        # so the agent enters the wait path instead of returning immediately.
        def _set_tls_on_send(reply, timeout):
            _tls.download_initiated = True
            return True

        agent.pipe_handler.send_to_peer.side_effect = _set_tls_on_send

        result = agent._do_submit_result(_make_task_ctx(), None, "OK")

        assert result is True, "Non-fatal timeout must still return True"
        agent.logger.warning.assert_called_once()
        warning_msg = agent.logger.warning.call_args[0][0]
        assert "0.05" in warning_msg or "Download completion" in warning_msg

    def test_send_fails_returns_false_without_waiting(self):
        """When send_to_peer() fails, returns False and does NOT wait for the callback."""
        pipe = _make_cell_pipe(pass_through_on_send=True)
        agent = _make_agent(pipe, download_complete_timeout=60.0)
        self._patch_shareable(agent)
        agent.pipe_handler.send_to_peer.return_value = False

        result = agent._do_submit_result(_make_task_ctx(), None, "OK")

        assert result is False

    def test_pass_through_false_uses_plain_send(self):
        """pass_through_on_send=False uses plain send_to_peer() without event gating."""
        pipe = _make_cell_pipe(pass_through_on_send=False)
        agent = _make_agent(pipe)
        self._patch_shareable(agent)
        agent.pipe_handler.send_to_peer.return_value = True

        result = agent._do_submit_result(_make_task_ctx(), None, "OK")

        assert result is True
        # DOWNLOAD_COMPLETE_CB must NOT be registered on the cell
        for c in pipe.cell.update_fobs_context.call_args_list:
            assert (
                FOBSContextKey.DOWNLOAD_COMPLETE_CB not in c[0][0]
            ), "DOWNLOAD_COMPLETE_CB must not be set when pass_through_on_send=False"

    def test_non_cell_pipe_uses_plain_send(self):
        """Non-CellPipe (e.g. FilePipe) uses plain send_to_peer(), no gating.

        FilePipe has no cell attribute; the code must not attempt to access
        pipe.cell.update_fobs_context.  We verify by confirming the result is
        True (send_to_peer succeeded) and that no AttributeError was raised.
        """
        pipe = _make_non_cell_pipe()
        agent = _make_agent(pipe)
        self._patch_shareable(agent)
        agent.pipe_handler.send_to_peer.return_value = True

        result = agent._do_submit_result(_make_task_ctx(), None, "OK")

        assert result is True
        agent.pipe_handler.send_to_peer.assert_called_once()

    def test_download_complete_cb_cleared_after_wait(self):
        """DOWNLOAD_COMPLETE_CB is set to None in FOBS context after the wait.

        Stale callbacks accumulate across rounds if not cleared and could fire
        for a later transaction, corrupting the gating Event for that round.
        """
        pipe = _make_cell_pipe(pass_through_on_send=True)
        agent = _make_agent(pipe, download_complete_timeout=0.05)
        self._patch_shareable(agent)

        # Simulate _finalize_download_tx() setting the flag so agent enters the wait path.
        def _set_tls_on_send(reply, timeout):
            _tls.download_initiated = True
            return True

        agent.pipe_handler.send_to_peer.side_effect = _set_tls_on_send

        agent._do_submit_result(_make_task_ctx(), None, "OK")

        # The last update_fobs_context call must clear DOWNLOAD_COMPLETE_CB
        last_call = pipe.cell.update_fobs_context.call_args_list[-1]
        last_props = last_call[0][0]
        assert FOBSContextKey.DOWNLOAD_COMPLETE_CB in last_props
        assert (
            last_props[FOBSContextKey.DOWNLOAD_COMPLETE_CB] is None
        ), "DOWNLOAD_COMPLETE_CB must be set to None after the wait"


# ---------------------------------------------------------------------------
# via_downloader._create_downloader() DOWNLOAD_COMPLETE_CB wiring
# ---------------------------------------------------------------------------


class TestCreateDownloaderCallback:
    """_create_downloader() must wire DOWNLOAD_COMPLETE_CB as transaction_done_cb."""

    def _make_fobs_ctx(self, cb=None):
        mock_cell = MagicMock()
        mock_cell.get_fqcn.return_value = "site1/job1"
        ctx = {
            FOBSContextKey.CELL: mock_cell,
            FOBSContextKey.NUM_RECEIVERS: 1,
        }
        if cb is not None:
            ctx[FOBSContextKey.DOWNLOAD_COMPLETE_CB] = cb
        return ctx

    def _make_decomposer(self):
        """Return the simplest concrete ViaDownloaderDecomposer used in these tests."""
        from nvflare.app_common.decomposers.numpy_decomposers import NumpyArrayDecomposer

        return NumpyArrayDecomposer()

    def test_download_complete_cb_passed_as_transaction_done_cb(self):
        """When DOWNLOAD_COMPLETE_CB is in fobs_ctx, it is wired as transaction_done_cb."""
        sentinel_cb = MagicMock()
        fobs_ctx = self._make_fobs_ctx(cb=sentinel_cb)

        with (patch("nvflare.fuel.utils.fobs.decomposers.via_downloader.ObjectDownloader") as MockOD,):
            MockOD.return_value = MagicMock()
            self._make_decomposer()._create_downloader(fobs_ctx)

        MockOD.assert_called_once()
        _, kwargs = MockOD.call_args
        assert (
            kwargs.get("transaction_done_cb") is sentinel_cb
        ), "DOWNLOAD_COMPLETE_CB must be wired as transaction_done_cb"

    def test_no_download_complete_cb_gives_none_transaction_done_cb(self):
        """Without DOWNLOAD_COMPLETE_CB in fobs_ctx, transaction_done_cb=None."""
        fobs_ctx = self._make_fobs_ctx(cb=None)

        with (patch("nvflare.fuel.utils.fobs.decomposers.via_downloader.ObjectDownloader") as MockOD,):
            MockOD.return_value = MagicMock()
            self._make_decomposer()._create_downloader(fobs_ctx)

        _, kwargs = MockOD.call_args
        assert (
            kwargs.get("transaction_done_cb") is None
        ), "transaction_done_cb must be None when DOWNLOAD_COMPLETE_CB is absent"

    def test_gc_callback_removed(self):
        """_on_tx_done (GC transaction_done_cb) must no longer exist in via_downloader."""
        import nvflare.fuel.utils.fobs.decomposers.via_downloader as vd

        assert not hasattr(vd, "_on_tx_done"), (
            "_on_tx_done must remain absent; object cleanup is handled synchronously "
            "and transaction completion belongs to DownloadService"
        )


# ---------------------------------------------------------------------------
# 11-12: ClientConfig.get_download_complete_timeout()
# ---------------------------------------------------------------------------


class TestClientConfigDownloadCompleteTimeout:
    """ClientConfig.get_download_complete_timeout() returns the configured value."""

    def test_returns_configured_value(self):
        """get_download_complete_timeout() returns the value from TASK_EXCHANGE section."""
        from nvflare.client.config import ClientConfig, ConfigKey

        cfg = ClientConfig(config={ConfigKey.TASK_EXCHANGE: {ConfigKey.DOWNLOAD_COMPLETE_TIMEOUT: 3600.0}})
        assert cfg.get_download_complete_timeout() == 3600.0

    def test_returns_default_when_not_set(self):
        """get_download_complete_timeout() returns 1800.0 when not configured."""
        from nvflare.client.config import ClientConfig

        cfg = ClientConfig(config={})
        assert cfg.get_download_complete_timeout() == 1800.0


# ---------------------------------------------------------------------------
# ClientConfig.get_max_resends() negative value clamping
# ---------------------------------------------------------------------------


class TestClientConfigMaxResends:
    """get_max_resends() clamps negative values to 0 with a warning.

    A negative max_resends (e.g. a typo of -1 in YAML) would otherwise behave
    like max_resends=0 silently, causing send_to_peer() to abort after the first
    failure without any indication to the user.
    """

    def test_positive_value_returned_as_is(self):
        """Positive max_resends is returned unchanged."""
        from nvflare.client.config import ClientConfig, ConfigKey

        cfg = ClientConfig(config={ConfigKey.TASK_EXCHANGE: {ConfigKey.MAX_RESENDS: 5}})
        assert cfg.get_max_resends() == 5

    def test_zero_is_valid(self):
        """max_resends=0 is valid (one attempt, no retries)."""
        from nvflare.client.config import ClientConfig, ConfigKey

        cfg = ClientConfig(config={ConfigKey.TASK_EXCHANGE: {ConfigKey.MAX_RESENDS: 0}})
        assert cfg.get_max_resends() == 0

    def test_none_returns_none(self):
        """max_resends=None means unlimited retries and is returned as None."""
        from nvflare.client.config import ClientConfig, ConfigKey

        cfg = ClientConfig(config={ConfigKey.TASK_EXCHANGE: {ConfigKey.MAX_RESENDS: None}})
        assert cfg.get_max_resends() is None

    def test_negative_clamped_to_zero(self):
        """Negative max_resends is clamped to 0."""
        from nvflare.client.config import ClientConfig, ConfigKey

        cfg = ClientConfig(config={ConfigKey.TASK_EXCHANGE: {ConfigKey.MAX_RESENDS: -1}})
        result = cfg.get_max_resends()
        assert result == 0, f"Expected 0 for max_resends=-1, got {result}"

    def test_negative_clamped_logs_warning(self):
        """Negative max_resends logs a warning so the user knows it was corrected."""
        from nvflare.client.config import ClientConfig, ConfigKey

        cfg = ClientConfig(config={ConfigKey.TASK_EXCHANGE: {ConfigKey.MAX_RESENDS: -3}})
        with patch.object(cfg.logger, "warning") as mock_warn:
            cfg.get_max_resends()
            mock_warn.assert_called_once()
            msg = mock_warn.call_args[0][0]
            assert "-3" in msg or "negative" in msg.lower(), f"Warning should mention -3 or 'negative': {msg}"

    def test_default_is_3(self):
        """Default max_resends when not configured is 3."""
        from nvflare.client.config import ClientConfig

        cfg = ClientConfig(config={})
        assert cfg.get_max_resends() == 3


# ---------------------------------------------------------------------------
# Status-based download logging
# ---------------------------------------------------------------------------


class TestDownloadStatusLogging:
    """_do_submit_result() logs info for FINISHED and warning for terminal failures."""

    def _patch_shareable(self, agent):
        agent.task_result_to_shareable = MagicMock(return_value=MagicMock())

    def setup_method(self):
        clear_download_initiated()

    def _run_with_status(self, status: str, timeout: float = 5.0):
        """Helper: run _do_submit_result and fire callback with given status."""
        pipe = _make_cell_pipe(pass_through_on_send=True)
        agent = _make_agent(pipe, download_complete_timeout=timeout)
        self._patch_shareable(agent)

        registered_cb = {}

        def fire_cb(reply, t):
            for c in pipe.cell.update_fobs_context.call_args_list:
                props = c[0][0]
                if (
                    FOBSContextKey.DOWNLOAD_COMPLETE_CB in props
                    and props[FOBSContextKey.DOWNLOAD_COMPLETE_CB] is not None
                ):
                    registered_cb["cb"] = props[FOBSContextKey.DOWNLOAD_COMPLETE_CB]
            # Simulate _finalize_download_tx() setting the flag (tensors in result)
            _tls.download_initiated = True
            if registered_cb.get("cb"):
                registered_cb["cb"]("tid", status, [])
            return True

        agent.pipe_handler.send_to_peer.side_effect = fire_cb
        agent._do_submit_result(_make_task_ctx(), None, "OK")
        return agent

    def test_finished_logs_info(self):
        """FINISHED status logs info, not a download warning."""
        from nvflare.fuel.f3.streaming.download_service import TransactionDoneStatus

        agent = self._run_with_status(TransactionDoneStatus.FINISHED)
        agent.logger.info.assert_called()
        # Must not log a warning for FINISHED
        for call in agent.logger.warning.call_args_list:
            msg = call[0][0]
            assert "download transaction" not in msg, f"FINISHED must not trigger download warning: {msg}"

    def test_timeout_logs_warning(self):
        """TIMEOUT status logs warning with status in the message."""
        from nvflare.fuel.f3.streaming.download_service import TransactionDoneStatus

        agent = self._run_with_status(TransactionDoneStatus.TIMEOUT)
        agent.logger.warning.assert_called()
        # At least one warning must mention the status
        msgs = [c[0][0] for c in agent.logger.warning.call_args_list]
        assert any(
            TransactionDoneStatus.TIMEOUT in m for m in msgs
        ), f"Warning must include status={TransactionDoneStatus.TIMEOUT!r}. Got: {msgs}"

    def test_deleted_logs_warning(self):
        """DELETED status logs warning with status in the message."""
        from nvflare.fuel.f3.streaming.download_service import TransactionDoneStatus

        agent = self._run_with_status(TransactionDoneStatus.DELETED)
        agent.logger.warning.assert_called()
        msgs = [c[0][0] for c in agent.logger.warning.call_args_list]
        assert any(
            TransactionDoneStatus.DELETED in m for m in msgs
        ), f"Warning must include status={TransactionDoneStatus.DELETED!r}. Got: {msgs}"


# ---------------------------------------------------------------------------
# FlareAgentWithCellPipe default values
# ---------------------------------------------------------------------------


class TestFlareAgentWithCellPipeDefaults:
    """FlareAgentWithCellPipe forwards timeout defaults to the base class."""

    def _make_cellpipe_cls(self):
        """Patch CellPipe so FlareAgentWithCellPipe.__init__ doesn't open sockets."""
        from nvflare.fuel.utils.pipe.cell_pipe import CellPipe

        return MagicMock(spec=CellPipe)

    def test_submit_result_timeout_default_is_60(self):
        """submit_result_timeout default must match the base-class default."""
        import inspect

        from nvflare.client.flare_agent import FlareAgentWithCellPipe

        sig = inspect.signature(FlareAgentWithCellPipe.__init__)
        default = sig.parameters["submit_result_timeout"].default
        assert (
            default == 60.0
        ), f"submit_result_timeout default must be 60.0 (aligned with FlareAgent base). Got {default}"

    def test_heartbeat_timeout_default_is_60(self):
        """heartbeat_timeout default must match the base-class default."""
        import inspect

        from nvflare.client.flare_agent import FlareAgentWithCellPipe

        sig = inspect.signature(FlareAgentWithCellPipe.__init__)
        default = sig.parameters["heartbeat_timeout"].default
        assert default == 60.0, f"heartbeat_timeout default must be 60.0 (aligned with FlareAgent base). Got {default}"

    def test_download_complete_timeout_param_exists(self):
        """download_complete_timeout parameter must exist with default 1800.0."""
        import inspect

        from nvflare.client.flare_agent import FlareAgentWithCellPipe

        sig = inspect.signature(FlareAgentWithCellPipe.__init__)
        assert (
            "download_complete_timeout" in sig.parameters
        ), "download_complete_timeout parameter must exist on FlareAgentWithCellPipe"
        default = sig.parameters["download_complete_timeout"].default
        assert default == 1800.0, f"download_complete_timeout default must be 1800.0. Got {default}"
