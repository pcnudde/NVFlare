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

import logging
from unittest.mock import Mock

import pytest

from nvflare.apis.event_type import EventType
from nvflare.apis.fl_constant import FLContextKey, ReservedKey
from nvflare.apis.fl_context import FLContext
from nvflare.app_common.widgets import event_recorder as subject


@pytest.mark.parametrize(
    "scenario,missing,none,dtype,attrs",
    [
        ("missing", 1, 0, 0, 0),
        ("none", 0, 1, 1, 0),
        ("wrong-type", 0, 0, 1, 0),
        ("public", 0, 0, 0, 1),
        ("private", 0, 0, 0, 1),
        ("sticky", 0, 0, 0, 1),
        ("non-sticky", 0, 0, 0, 1),
        ("optional", 0, 0, 0, 0),
        ("valid", 0, 0, 0, 0),
    ],
)
def test_property_validation_classifies_missing_values_types_and_visibility(scenario, missing, none, dtype, attrs):
    recorder = subject.EventRecorder()
    stats = subject._EventStats()
    ctx = FLContext()
    ctx.set_prop(recorder._KEY_EVENT_STATS, stats, private=True, sticky=False)
    request = subject._CtxPropReq(str, True, True)
    if scenario != "missing":
        value = None if scenario in ("none", "optional") else 5 if scenario == "wrong-type" else "value"
        private = scenario != "public"
        sticky = scenario != "non-sticky"
        if scenario == "private":
            request.is_private = False
        if scenario == "sticky":
            request.is_sticky = False
        if scenario == "optional":
            request = subject._CtxPropReq(None, None, None, allow_none=True)
        ctx.set_prop("required", value, private=private, sticky=sticky)
    recorder.validate_prop("required", request, ctx)
    assert (stats.prop_missing, stats.prop_none_value, stats.prop_dtype_mismatch, stats.prop_attr_mismatch) == (
        missing,
        none,
        dtype,
        attrs,
    )


@pytest.fixture(params=[subject.EventRecorder, subject.ServerEventRecorder, subject.ClientEventRecorder])
def recorder_runtime(request, tmp_path):
    recorder = request.param()
    logger = recorder.logger
    old_handlers = set(logger.handlers)
    old_level = logger.level
    logger.setLevel(logging.INFO)
    engine = Mock()
    engine.get_workspace.return_value.get_app_dir.return_value = str(tmp_path)
    ctx = FLContext()
    ctx.set_prop(ReservedKey.ENGINE, engine, private=True, sticky=True)
    ctx.set_prop(ReservedKey.RUN_NUM, "job", private=False, sticky=True)
    ctx.set_prop(ReservedKey.IDENTITY_NAME, "site", private=False, sticky=True)
    try:
        yield recorder, ctx, tmp_path
    finally:
        for handler in set(logger.handlers) - old_handlers:
            logger.removeHandler(handler)
            handler.close()
        logger.setLevel(old_level)


def test_event_recording_has_owned_file_handler_counts_calls_and_seeds_site_context(recorder_runtime):
    recorder, ctx, root = recorder_runtime
    ctx.set_prop(FLContextKey.EVENT_ID, "evt-1", private=True, sticky=False)
    recorder.handle_event(EventType.START_RUN, ctx)
    recorder.handle_event(EventType.END_RUN, ctx)
    assert recorder.event_stats[EventType.START_RUN].call_count == 1
    assert recorder.event_stats[EventType.END_RUN].call_count == 1
    assert recorder.event_stats[EventType.START_RUN].prop_missing == 0
    for handler in recorder.logger.handlers:
        handler.flush()
    log = (root / recorder.log_file_name).read_text()
    assert "Got event" in log and "evt-1" in log
    assert sum(isinstance(handler, logging.FileHandler) for handler in recorder.logger.handlers) == 1
    if isinstance(recorder, subject.ServerEventRecorder):
        assert ctx.get_prop_detail("testPrivateServerSticky")["private"]
        assert not ctx.get_prop_detail("testPublicServerNonSticky")["sticky"]
    if isinstance(recorder, subject.ClientEventRecorder):
        assert ctx.get_prop_detail("testPrivateClientSticky")["private"]
        assert not ctx.get_prop_detail("testPublicClientNonSticky")["sticky"]


def test_custom_event_requirements_validate_peer_properties_and_block_lists(recorder_runtime):
    recorder, ctx, _ = recorder_runtime
    required = subject._CtxPropReq(str, False, False)
    recorder.event_reqs["custom"] = subject._EventReq(
        {"required": required}, {"remote": required}, ["forbidden"], ["peer_forbidden"]
    )
    ctx.set_prop("required", "ok", private=False, sticky=False)
    ctx.set_prop("forbidden", 1, private=True, sticky=False)
    recorder.handle_event("custom", ctx)
    assert recorder.event_stats["custom"].peer_ctx_missing == 1
    peer = FLContext()
    peer.set_prop("remote", "remote-value", private=False, sticky=False)
    peer.set_prop("peer_forbidden", 1, private=True, sticky=False)
    ctx.set_peer_context(peer)
    recorder.handle_event("custom", ctx)
    stats = recorder.event_stats["custom"]
    assert stats.call_count == 2
    assert stats.prop_block_list_violation == 3
    assert peer.get_prop(recorder._KEY_EVENT_STATS) is stats
    assert peer.get_prop(recorder._KEY_CTX_TYPE) == "peer_ctx"
    assert peer.get_prop(recorder._KEY_EVENT_TYPE) == "custom"
    assert recorder.event_tag(FLContext()) == "[?]"
