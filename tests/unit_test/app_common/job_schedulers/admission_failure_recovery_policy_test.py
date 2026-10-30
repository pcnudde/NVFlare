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

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from nvflare.apis.fl_constant import ReservedKey
from nvflare.apis.fl_context import FLContext
from nvflare.apis.job_def import JobMetaKey
from nvflare.app_common.job_schedulers import job_scheduler as subject


@pytest.mark.parametrize("method", ["check", "cancel"])
def test_scheduler_rejects_context_without_server_engine(method):
    scheduler, context = subject.DefaultJobScheduler(), FLContext()
    context.set_prop(ReservedKey.ENGINE, object())
    with pytest.raises(RuntimeError, match="ServerEngineSpec"):
        if method == "check":
            scheduler._check_client_resources(Mock(), {}, context)
        else:
            scheduler._cancel_resources({}, {}, context)


def test_restored_jobs_are_unique_and_concurrency_limit_prevents_admission(monkeypatch):
    scheduler, context = subject.DefaultJobScheduler(max_jobs=1), FLContext()
    scheduler.restore_scheduled_job("one")
    scheduler.restore_scheduled_job("one")
    assert scheduler.scheduled_jobs == ["one"]
    admit = Mock()
    monkeypatch.setattr(scheduler, "_try_job", admit)
    assert scheduler._do_schedule_job([Mock()], context, [], []) == (None, None)
    admit.assert_not_called()
    scheduler.remove_scheduled_job("one")
    scheduler.remove_scheduled_job("one")
    assert scheduler.scheduled_jobs == []


@pytest.mark.parametrize("failure", ["admission", "store"])
def test_schedule_errors_are_contained_and_cannot_publish_ready_job(monkeypatch, failure):
    scheduler, context, manager = subject.DefaultJobScheduler(), FLContext(), Mock()

    def schedule(candidates, context, failed, blocked):
        if failure == "admission":
            raise ValueError("broken admission")
        blocked.append(SimpleNamespace(job_id="job"))
        return None, None

    monkeypatch.setattr(scheduler, "_do_schedule_job", schedule)
    manager.refresh_meta.side_effect = OSError("store unavailable")
    error = Mock()
    monkeypatch.setattr(scheduler, "log_exception", error)
    assert scheduler.schedule_job(manager, [], context) == (None, None)
    assert error.call_count == 1
    manager.set_status.assert_not_called()


def test_retry_backoff_does_not_admit_job_before_interval(monkeypatch):
    scheduler, context = subject.DefaultJobScheduler(min_schedule_interval=100), FLContext()
    context.set_prop(ReservedKey.ENGINE, Mock())
    job = SimpleNamespace(
        job_id="job", meta={JobMetaKey.SCHEDULE_COUNT.value: 1, JobMetaKey.LAST_SCHEDULE_TIME.value: 100}
    )
    monkeypatch.setattr(subject, "time", SimpleNamespace(time=lambda: 150))
    admit = Mock()
    monkeypatch.setattr(scheduler, "_try_job", admit)
    failed, blocked = [], []
    assert scheduler._do_schedule_job([job], context, failed, blocked) == (None, None)
    admit.assert_not_called()
    assert failed == blocked == []


def test_cleanup_failure_after_admission_error_does_not_prevent_next_candidate(monkeypatch):
    scheduler, context = subject.DefaultJobScheduler(min_schedule_interval=0), FLContext()
    engine = Mock()
    engine.new_context.side_effect = lambda: nullcontext(context)
    context.set_prop(ReservedKey.ENGINE, engine)
    first = SimpleNamespace(job_id="first", meta={})
    second = SimpleNamespace(job_id="second", meta={})
    dispatch = {"site": object()}
    monkeypatch.setattr(scheduler, "_try_job", Mock(return_value=(subject.SCHEDULE_RESULT_OK, dispatch, "ready")))
    calls = []

    def update(job, reason, ctx):
        calls.append(job.job_id)
        if job is first:
            raise ValueError("history unavailable")
        job.meta[JobMetaKey.SCHEDULE_COUNT.value] = 1

    monkeypatch.setattr(scheduler, "_update_schedule_history", update)
    cleanup = Mock(side_effect=OSError("cancel failed"))
    monkeypatch.setattr(scheduler, "_cancel_dispatch_resources", cleanup)
    failed = []
    ready, info = scheduler._do_schedule_job([first, second], context, failed, [])
    assert ready is second and info is dispatch
    assert failed == [first]
    cleanup.assert_called_once_with(dispatch, context)
    assert first.meta[JobMetaKey.SCHEDULE_COUNT.value] == 1
    assert calls == ["first", "first", "second"]
