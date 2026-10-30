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

import json
import threading
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from nvflare.apis.fl_constant import FLContextKey, ReservedKey, StreamCtxKey
from nvflare.apis.fl_context import FLContext
from nvflare.app_common.logging import constants, job_log_streamer, site_log_streamer
from tests.timing_utils import isolate_time


@pytest.mark.parametrize("configured", [False, True])
def test_explicit_site_setting_takes_precedence_without_reading_workspace(monkeypatch, configured):
    monkeypatch.setattr(constants.ConfigService, "get_bool_var", Mock(return_value=configured))
    workspace = Mock()
    monkeypatch.setattr(constants, "Workspace", workspace)
    assert constants.is_log_streaming_allowed(FLContext()) is configured
    workspace.assert_not_called()


@pytest.mark.parametrize(
    "content, expected", [("{}", True), ('{"allow_log_streaming": false}', False), ("invalid", True)]
)
def test_workspace_resources_fallback_defaults_to_allow_but_honors_explicit_disable(
    tmp_path, monkeypatch, content, expected
):
    monkeypatch.setattr(constants.ConfigService, "get_bool_var", Mock(return_value=None))
    (tmp_path / "startup").mkdir()
    (tmp_path / "local").mkdir()
    (tmp_path / "local" / "resources.json").write_text(content)
    ctx = FLContext()
    ctx.set_prop(FLContextKey.WORKSPACE_ROOT, str(tmp_path))
    ctx.set_prop(ReservedKey.IDENTITY_NAME, "site")
    assert constants.is_log_streaming_allowed(ctx) is expected
    (tmp_path / "local" / "resources.json").unlink()
    assert constants.is_log_streaming_allowed(ctx) is True


def test_missing_workspace_context_uses_default_allow_without_creating_workspace(monkeypatch):
    monkeypatch.setattr(constants.ConfigService, "get_bool_var", Mock(return_value=None))
    factory = Mock()
    monkeypatch.setattr(constants, "Workspace", factory)
    assert constants.is_log_streaming_allowed() is True
    assert constants.is_log_streaming_allowed(FLContext()) is True
    factory.assert_not_called()


@pytest.mark.parametrize("stopped", [False, True])
def test_job_stream_waits_for_file_once_or_honors_stop_before_opening(tmp_path, monkeypatch, stopped):
    widget = job_log_streamer.JobLogStreamer(poll_interval=0.25)
    event = threading.Event()
    if stopped:
        event.set()
    widget._stop_event = event
    path = tmp_path / "log.txt"
    clock = isolate_time(monkeypatch, job_log_streamer)
    sleeps = []

    def sleep(delay):
        sleeps.append(delay)
        assert len(sleeps) == 1, "streamer did not notice the newly created log file"
        path.write_text("ready")

    monkeypatch.setattr(clock, "sleep", sleep)
    stream = Mock()
    monkeypatch.setattr(job_log_streamer.LogStreamer, "stream_log", stream)
    ctx = FLContext()
    widget._do_stream(str(path), "site", "job", ctx, ["server"])
    assert sleeps == ([] if stopped else [0.25])
    assert stream.call_count == int(not stopped)
    if not stopped:
        call = stream.call_args.kwargs
        assert call["stop_event"] is event and call["fl_ctx"] is ctx
        assert call["stream_ctx"] == {StreamCtxKey.CLIENT_NAME: "site", StreamCtxKey.JOB_ID: "job"}


def test_job_stream_without_stop_event_is_rejected_and_transport_failure_is_logged(tmp_path, monkeypatch):
    widget = job_log_streamer.JobLogStreamer()
    warning, exception = Mock(), Mock()
    monkeypatch.setattr(widget, "log_warning", warning)
    monkeypatch.setattr(widget, "log_exception", exception)
    stream = Mock(side_effect=OSError("receiver offline"))
    monkeypatch.setattr(job_log_streamer.LogStreamer, "stream_log", stream)
    ctx = FLContext()
    path = tmp_path / "log.txt"
    path.write_text("ready")
    widget._do_stream(str(path), "site", "job", ctx, ["server"])
    stream.assert_not_called()
    warning.assert_called_once()
    widget._stop_event = threading.Event()
    widget._do_stream(str(path), "site", "job", ctx, ["server"])
    stream.assert_called_once()
    exception.assert_called_once()


@pytest.mark.parametrize("failure", ["invalid-json", "write-denied", "none-components"])
def test_site_injection_handles_configuration_failures_and_normalizes_empty_components(tmp_path, monkeypatch, failure):
    path = tmp_path / "config.json"
    path.write_text("broken" if failure == "invalid-json" else json.dumps({"components": None}))
    monkeypatch.setattr(
        site_log_streamer, "Workspace", lambda **kwargs: SimpleNamespace(get_app_config_dir=lambda job: str(tmp_path))
    )
    monkeypatch.setattr(site_log_streamer.JobConstants, "CLIENT_JOB_CONFIG", "config.json")
    monkeypatch.setattr(site_log_streamer, "is_log_streaming_allowed", lambda ctx: True)
    widget = site_log_streamer.SiteLogStreamer(log_file_name="custom.log", liveness_interval=4, poll_interval=0.2)
    exception = Mock()
    monkeypatch.setattr(widget, "log_exception", exception)
    if failure == "write-denied":
        monkeypatch.setattr(widget, "_atomic_write_json", Mock(side_effect=PermissionError("read only")))
    ctx = FLContext()
    ctx.set_prop(FLContextKey.JOB_META, {"job_id": "job"})
    ctx.set_prop(FLContextKey.WORKSPACE_ROOT, str(tmp_path))
    ctx.set_prop(ReservedKey.IDENTITY_NAME, "site")
    widget._on_before_job_launch("launch", ctx)
    assert exception.call_count == int(failure != "none-components")
    if failure == "none-components":
        component = json.loads(path.read_text())["components"][0]
        assert component["args"] == {
            "target_parent_server": True,
            "log_file_name": "custom.log",
            "liveness_interval": 4,
            "poll_interval": 0.2,
        }
    elif failure == "write-denied":
        assert json.loads(path.read_text()) == {"components": None}
