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

import nvflare.private.fed.server.job_completion_driver as driver_module
from nvflare.apis.job_def import RunStatus
from nvflare.private.fed.server.job_completion import Action, ClientOutcome, Event, State, transition
from nvflare.private.fed.server.job_completion_driver import JobCompletionDriver


class _Effects:
    def __init__(self, archive_failures=0, cleanup_failures=0, publish_failures=0):
        self.now = 0.0
        self.archive_failures = archive_failures
        self.cleanup_failures = cleanup_failures
        self.publish_failures = publish_failures
        self.archive_calls = 0
        self.cleanup_calls = 0
        self.publish_calls = 0
        self.published = []
        self.finalized = []
        self.on_now = None
        self.on_status = None

    def completion_now(self):
        if self.on_now:
            callback, self.on_now = self.on_now, None
            callback()
        return self.now

    def get_completion_status(self, engine, job, fl_ctx):
        if self.on_status:
            callback, self.on_status = self.on_status, None
            callback()
        return RunStatus.FINISHED_COMPLETED

    def archive_completion_workspace(self, fl_ctx):
        self.archive_calls += 1
        if self.archive_calls <= self.archive_failures:
            raise RuntimeError("archive failure")
        return ("run",)

    def cleanup_completion_workspace(self, fl_ctx, paths):
        self.cleanup_calls += 1
        if self.cleanup_calls <= self.cleanup_failures:
            raise RuntimeError("cleanup failure")

    def publish_completion_status(self, job_manager, job_id, status, fl_ctx):
        self.publish_calls += 1
        if self.publish_calls <= self.publish_failures:
            raise RuntimeError("publish failure")
        self.published.append(status)

    def finalize_completion(self, engine, job_id, status, fl_ctx):
        self.finalized.append(status)

    def log_warning(self, fl_ctx, msg):
        pass

    def log_exception(self, fl_ctx, msg):
        pass

    def log_error(self, fl_ctx, msg):
        pass


def _driver(monkeypatch, effects, retry_grace_time=10.0):
    actions = []
    real_transition = driver_module.transition

    def recording_transition(state, event):
        actions.append(event.action)
        return real_transition(state, event)

    monkeypatch.setattr(driver_module, "transition", recording_transition)
    driver = JobCompletionDriver(effects, client_wait_timeout=10.0, retry_grace_time=retry_grace_time)
    driver.start("job")
    return driver, actions


def _job():
    return SimpleNamespace(job_id="job", run_aborted=False)


def test_successful_effects_follow_the_model_action_order(monkeypatch):
    effects = _Effects()
    driver, actions = _driver(monkeypatch, effects)

    driver.advance(None, None, _job(), None)

    assert actions == [
        Action.SERVER_EXITED,
        Action.ARCHIVE_COMMITTED,
        Action.CLEANUP_SETTLED,
        Action.STATUS_PUBLISHED,
    ]
    assert effects.published == [RunStatus.FINISHED_COMPLETED]
    assert effects.finalized == [RunStatus.FINISHED_COMPLETED]
    assert driver.job_ids() == set()


def test_participant_selection_uses_checked_machine_action(monkeypatch):
    effects = _Effects()
    driver, actions = _driver(monkeypatch, effects)
    driver.start("job", ("site-1", "site-2"))

    driver.select_participants("job", ("site-1",))

    assert actions == [Action.PARTICIPANTS_SELECTED]
    assert driver.job_ids("site-1") == {"job"}
    assert driver.job_ids("site-2") == set()


def test_terminal_status_precedence_is_independent_of_event_order():
    statuses = (RunStatus.FINISHED_ABORTED, RunStatus.FINISHED_EXECUTION_EXCEPTION, RunStatus.FINISHED_ABNORMAL)

    forward = State.initial(("site-1",))
    reverse = State.initial(("site-1",))
    for status in statuses:
        forward = transition(forward, Event(Action.TERMINAL_OVERRIDE, status=status))
    for status in reversed(statuses):
        reverse = transition(reverse, Event(Action.TERMINAL_OVERRIDE, status=status))

    assert forward.status == reverse.status == RunStatus.FINISHED_ABNORMAL


def test_archive_retry_is_bounded_and_downgrades_success(monkeypatch):
    effects = _Effects(archive_failures=2)
    driver, actions = _driver(monkeypatch, effects)
    job = _job()

    driver.advance(None, None, job, None)
    assert actions == [Action.SERVER_EXITED]
    assert not effects.published

    effects.now = 10.0
    driver.advance(None, None, job, None)

    assert actions == [Action.SERVER_EXITED, Action.ARCHIVE_ABANDONED, Action.STATUS_PUBLISHED]
    assert effects.archive_calls == 2
    assert effects.cleanup_calls == 0
    assert effects.published == [RunStatus.FINISHED_EXECUTION_EXCEPTION]


def test_cleanup_retry_never_rearchives_committed_data(monkeypatch):
    effects = _Effects(cleanup_failures=2)
    driver, actions = _driver(monkeypatch, effects)
    job = _job()

    driver.advance(None, None, job, None)
    effects.now = 10.0
    driver.advance(None, None, job, None)

    assert actions == [
        Action.SERVER_EXITED,
        Action.ARCHIVE_COMMITTED,
        Action.CLEANUP_SETTLED,
        Action.STATUS_PUBLISHED,
    ]
    assert effects.archive_calls == 1
    assert effects.cleanup_calls == 2
    assert effects.published == [RunStatus.FINISHED_COMPLETED]


def test_publication_retry_does_not_repeat_archive_or_cleanup(monkeypatch):
    effects = _Effects(publish_failures=1)
    driver, actions = _driver(monkeypatch, effects)
    job = _job()

    driver.advance(None, None, job, None)
    driver.advance(None, None, job, None)

    assert actions == [
        Action.SERVER_EXITED,
        Action.ARCHIVE_COMMITTED,
        Action.CLEANUP_SETTLED,
        Action.STATUS_PUBLISHED,
    ]
    assert effects.archive_calls == effects.cleanup_calls == 1
    assert effects.publish_calls == 2


def test_override_interleaving_during_status_effect_wins(monkeypatch):
    effects = _Effects()
    driver, _ = _driver(monkeypatch, effects)
    effects.on_status = lambda: driver.terminal_override("job", RunStatus.FINISHED_ABORTED)

    driver.advance(None, None, _job(), None)

    assert effects.published == [RunStatus.FINISHED_ABORTED]


def test_last_client_resolution_interleaving_during_clock_effect_is_rechecked(monkeypatch):
    effects = _Effects()
    driver, _ = _driver(monkeypatch, effects)
    driver.start("job", ("site-1",))
    effects.on_now = lambda: driver.record_client_outcome("job", "site-1", ClientOutcome.NO_OVERRIDE)

    driver.advance(None, None, _job(), None)

    assert effects.published == [RunStatus.FINISHED_COMPLETED]
