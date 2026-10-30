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

from nvflare.fuel.common.excepts import ComponentNotAuthorized
from nvflare.fuel.common.exit_codes import ProcessExitCode
from nvflare.fuel.f3 import mpm as module


@pytest.fixture
def monitor(monkeypatch):
    subject = module.MainProcessMonitor
    for key, value in {
        "_cleanup_cbs": [],
        "_stopping": False,
        "_logger": Mock(),
        "_aio_ctx": None,
        "name": "unit-monitor",
    }.items():
        monkeypatch.setattr(subject, key, value)
    return subject


@pytest.mark.parametrize("value", [None, "", 7])
def test_monitor_name_requires_nonempty_string(monitor, value):
    with pytest.raises(ValueError, match="name must"):
        monitor.set_name(value)
    assert monitor.name == "unit-monitor"


def test_aio_context_is_lazily_acquired_once_and_reused(monkeypatch, monitor):
    context = object()
    factory = Mock(return_value=context)
    monkeypatch.setattr(module, "AioContext", SimpleNamespace(get_global_context=factory))
    assert monitor.get_aio_context() is context
    assert monitor.get_aio_context() is context
    factory.assert_called_once_with()


def test_monitor_rejects_invalid_cleanup_and_contains_callback_failure(monitor):
    with pytest.raises(ValueError, match="not callable"):
        monitor.add_cleanup_cb(7)

    def bad_callback():
        raise ValueError("cleanup failed")

    assert monitor._call_cb((bad_callback, (), {})) is None
    assert "cleanup failed" in monitor.logger().error.call_args.args[0]
    monitor._cleanup_one_round([()])
    assert monitor.logger().warning.call_count == 1


def test_recursive_cleanup_has_finite_round_budget_and_signals_waiter(monitor):
    calls = []

    def cleanup():
        calls.append("called")
        if len(calls) <= 10:
            monitor.add_cleanup_cb(cleanup)

    monitor.add_cleanup_cb(cleanup)
    waiter = Mock()
    monitor._do_cleanup(waiter)
    assert len(calls) == 10
    assert len(monitor._cleanup_cbs) == 1
    waiter.set.assert_called_once_with()
    assert "after 10 rounds" in monitor.logger().warning.call_args.args[0]


def test_main_monitor_rejects_noncallable_and_nonmain_thread_without_running_callback(monkeypatch, monitor):
    with pytest.raises(ValueError, match="must be runnable"):
        monitor.run(None)
    callback = Mock()
    monkeypatch.setattr(module, "threading", SimpleNamespace(current_thread=lambda: SimpleNamespace(name="worker")))
    with pytest.raises(RuntimeError, match="MainThread"):
        monitor.run(callback)
    callback.assert_not_called()


def test_main_monitor_maps_unauthorized_component_and_cleans_previous_rc_file(monkeypatch, tmp_path, monitor):
    monkeypatch.chdir(tmp_path)
    marker = tmp_path / module.FLMetaKey.PROCESS_RC_FILE
    marker.write_text("previous")
    main_thread = SimpleNamespace(name="MainThread", daemon=False)
    monkeypatch.setattr(
        module, "threading", SimpleNamespace(current_thread=lambda: main_thread, enumerate=lambda: [main_thread])
    )
    shutdown, close = Mock(), Mock()
    monkeypatch.setattr(monitor, "_start_shutdown", shutdown)
    monkeypatch.setattr(module, "AioContext", SimpleNamespace(close_global_context=close))
    callback = Mock(side_effect=ComponentNotAuthorized("untrusted"))
    assert monitor.run(callback, shutdown_grace_time=0, cleanup_grace_time=1) == ProcessExitCode.UNSAFE_COMPONENT
    assert not marker.exists()
    shutdown.assert_called_once_with(0, 1)
    close.assert_called_once_with()
    assert monitor.is_stopping()
