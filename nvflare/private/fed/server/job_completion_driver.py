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

"""Side-effect driver for the pure job-completion state machine."""

import threading
from dataclasses import dataclass
from typing import Protocol

from nvflare.apis.job_def import RunStatus
from nvflare.private.fed.server.job_completion import Action, ClientOutcome, Event, Phase, State, transition
from nvflare.security.logging import secure_format_exception


class CompletionEffects(Protocol):
    """FLARE operations required by the lifecycle driver."""

    def completion_now(self) -> float:
        raise NotImplementedError

    def get_completion_status(self, engine, job, fl_ctx) -> RunStatus:
        raise NotImplementedError

    def archive_completion_workspace(self, fl_ctx) -> tuple[str, ...]:
        raise NotImplementedError

    def cleanup_completion_workspace(self, fl_ctx, paths: tuple[str, ...]) -> None:
        raise NotImplementedError

    def publish_completion_status(self, job_manager, job_id: str, status: RunStatus, fl_ctx) -> None:
        raise NotImplementedError

    def finalize_completion(self, engine, job_id: str, status: RunStatus, fl_ctx) -> None:
        raise NotImplementedError

    def log_warning(self, fl_ctx, msg: str):
        raise NotImplementedError

    def log_exception(self, fl_ctx, msg: str):
        raise NotImplementedError

    def log_error(self, fl_ctx, msg: str):
        raise NotImplementedError


@dataclass
class _CompletionRuntime:
    state: State
    client_deadline: float | None = None
    operation_started_at: float | None = None
    archive_sources: tuple[str, ...] = ()

    def advance(self, action: Action, **event_data):
        self.state = transition(self.state, Event(action, **event_data))


class JobCompletionDriver:
    """Own completion runtime and translate effect results into machine events."""

    def __init__(
        self,
        effects: CompletionEffects,
        client_wait_timeout: float,
        retry_grace_time: float,
    ):
        self.effects = effects
        self.client_wait_timeout = client_wait_timeout
        self.retry_grace_time = retry_grace_time
        self._jobs: dict[str, _CompletionRuntime] = {}
        self._lock = threading.Lock()

    def start(self, job_id: str, clients=()) -> None:
        with self._lock:
            self._jobs[job_id] = _CompletionRuntime(State.initial(clients))

    def discard(self, job_id: str) -> None:
        with self._lock:
            self._jobs.pop(job_id, None)

    def select_participants(self, job_id: str, clients) -> None:
        with self._lock:
            runtime = self._jobs[job_id]
            runtime.advance(Action.PARTICIPANTS_SELECTED, clients=frozenset(clients))

    def job_ids(self, client_name: str | None = None) -> set[str]:
        with self._lock:
            if client_name is None:
                return set(self._jobs)
            return {job_id for job_id, runtime in self._jobs.items() if client_name in runtime.state.pending_clients}

    def record_client_outcome(self, job_id: str, client_name: str, outcome: ClientOutcome) -> bool:
        with self._lock:
            runtime = self._jobs.get(job_id)
            if runtime and client_name in runtime.state.pending_clients:
                runtime.advance(Action.RECORD_CLIENT_OUTCOME, client=client_name, outcome=outcome)
                return True
            return False

    def terminal_override(self, job_id: str, status: RunStatus) -> None:
        with self._lock:
            runtime = self._jobs.get(job_id)
            if runtime:
                self._terminal_override(runtime, status)

    @staticmethod
    def _terminal_override(runtime: _CompletionRuntime, status: RunStatus) -> None:
        if runtime.state.phase in (Phase.WAITING_FOR_SERVER, Phase.WAITING_FOR_CLIENTS):
            runtime.advance(Action.TERMINAL_OVERRIDE, status=status)
        runtime.client_deadline = None

    def advance(self, engine, job_manager, job, fl_ctx) -> None:
        """Run machine-requested effects until completion or an external retry is needed."""
        job_id = job.job_id
        with self._lock:
            runtime = self._jobs.setdefault(job_id, _CompletionRuntime(State.initial()))
            if job.run_aborted:
                self._terminal_override(runtime, RunStatus.FINISHED_ABORTED)

        while True:
            state = runtime.state
            if state.phase == Phase.WAITING_FOR_SERVER:
                status = self.effects.get_completion_status(engine, job, fl_ctx)
                with self._lock:
                    runtime.advance(Action.SERVER_EXITED, status=status)
                continue

            if state.phase == Phase.WAITING_FOR_CLIENTS:
                now = self.effects.completion_now()
                with self._lock:
                    state = runtime.state
                    if state.phase != Phase.WAITING_FOR_CLIENTS:
                        continue
                    if runtime.client_deadline is None:
                        runtime.client_deadline = now + self.client_wait_timeout
                    if now < runtime.client_deadline:
                        return
                    unresolved = sorted(state.pending_clients)
                    runtime.advance(Action.CLIENT_WAIT_EXPIRED)
                    runtime.client_deadline = None
                self.effects.log_warning(
                    fl_ctx,
                    f"Timed out after {self.client_wait_timeout} seconds waiting for client outcomes "
                    f"for job ({job_id}): {unresolved}. Candidate success will be published as failed.",
                )
                continue

            if state.phase == Phase.ARCHIVING:
                proceed, sources = self._attempt_effect(
                    runtime,
                    job_id,
                    fl_ctx,
                    lambda: self.effects.archive_completion_workspace(fl_ctx),
                    Action.ARCHIVE_COMMITTED,
                    Action.ARCHIVE_ABANDONED,
                )
                if not proceed:
                    return
                if sources:
                    runtime.archive_sources = sources
                continue

            if state.phase == Phase.CLEANING:
                proceed, _ = self._attempt_effect(
                    runtime,
                    job_id,
                    fl_ctx,
                    lambda: self.effects.cleanup_completion_workspace(fl_ctx, runtime.archive_sources),
                    Action.CLEANUP_SETTLED,
                )
                if not proceed:
                    return
                continue

            if state.phase == Phase.PUBLISHING:
                assert state.status is not None
                try:
                    self.effects.publish_completion_status(job_manager, job_id, state.status, fl_ctx)
                except Exception as e:
                    self.effects.log_exception(
                        fl_ctx,
                        f"Failed to publish finished status for job ({job_id}): {secure_format_exception(e)}",
                    )
                    return
                runtime.advance(Action.STATUS_PUBLISHED)
                self.discard(job_id)
                self.effects.finalize_completion(engine, job_id, state.status, fl_ctx)
                return

            return

    def _attempt_effect(self, runtime, job_id, fl_ctx, effect, succeeded, abandoned=None):
        phase = runtime.state.phase
        try:
            result = effect()
        except Exception as e:
            now = self.effects.completion_now()
            if runtime.operation_started_at is None:
                runtime.operation_started_at = now
            if now - runtime.operation_started_at < self.retry_grace_time:
                self.effects.log_exception(
                    fl_ctx,
                    f"Completion effect {phase.value} failed for job ({job_id}); will retry: "
                    f"{secure_format_exception(e)}",
                )
                return False, None
            runtime.advance(abandoned or succeeded)
            self.effects.log_error(
                fl_ctx,
                f"Completion effect {phase.value} kept failing for {self.retry_grace_time} seconds for "
                f"job ({job_id}); continuing with {runtime.state.status.value}: {secure_format_exception(e)}",
            )
            result = None
        else:
            runtime.advance(succeeded)
        runtime.operation_started_at = None
        return True, result
