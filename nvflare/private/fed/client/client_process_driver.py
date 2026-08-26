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

"""Thread-safe owner of client job-process lifecycle state."""

import threading
from dataclasses import dataclass

from nvflare.apis.job_launcher_spec import JobHandleSpec

from .client_process import Action, Event, Phase, State, StopIntent, transition
from .client_status import ClientStatus


@dataclass(frozen=True)
class StopRequest:
    phase: Phase
    handle: JobHandleSpec | None
    intent: StopIntent


@dataclass
class _Runtime:
    state: State
    handle: JobHandleSpec | None = None


class ClientProcessDriver:
    """Serialize lifecycle transitions and the handle they govern."""

    def __init__(self):
        self._lock = threading.Lock()
        self._jobs: dict[str, _Runtime] = {}

    def register(self, job_id: str) -> None:
        with self._lock:
            if job_id in self._jobs:
                raise RuntimeError(f"client app for job '{job_id}' is still registered")
            self._jobs[job_id] = _Runtime(State())

    def attach_handle(self, job_id: str, handle: JobHandleSpec) -> StopIntent:
        with self._lock:
            runtime = self._require(job_id)
            runtime.state = transition(runtime.state, Event(Action.ATTACH_HANDLE))
            runtime.handle = handle
            return runtime.state.stop_intent

    def launch_failed(self, job_id: str) -> None:
        with self._lock:
            runtime = self._require(job_id)
            runtime.state = transition(runtime.state, Event(Action.LAUNCH_FAILED))
            self._jobs.pop(job_id)

    def record_worker_status(self, job_id: str, status: int) -> None:
        action = {ClientStatus.STARTED: Action.WORKER_STARTED, ClientStatus.STOPPED: Action.WORKER_STOPPED}.get(status)
        if action is None:
            raise ValueError(f"unsupported client process status: {status}")
        with self._lock:
            runtime = self._jobs.get(job_id)
            if not runtime:
                return
            current = self._status(runtime.state)
            if status == current or current == ClientStatus.STOPPED:
                return
            runtime.state = transition(runtime.state, Event(action))

    def request_stop(self, job_id: str, intent: StopIntent) -> StopRequest | None:
        with self._lock:
            runtime = self._jobs.get(job_id)
            if not runtime or not runtime.state.registered:
                return None
            if runtime.state.phase in {Phase.LAUNCHING, Phase.RUNNING, Phase.RUNNER_STOPPED}:
                runtime.state = transition(runtime.state, Event(Action.REQUEST_STOP, stop_intent=intent))
            return StopRequest(runtime.state.phase, runtime.handle, runtime.state.stop_intent)

    def status(self, job_id: str, default: int) -> int:
        with self._lock:
            runtime = self._jobs.get(job_id)
            return self._status(runtime.state) if runtime else default

    def handle(self, job_id: str) -> JobHandleSpec:
        with self._lock:
            handle = self._require(job_id).handle
            if handle is None:
                raise RuntimeError(f"job '{job_id}' has no attached process handle")
            return handle

    def process_exited(self, job_id: str) -> Phase:
        with self._lock:
            runtime = self._require(job_id)
            previous = runtime.state.phase
            runtime.state = transition(runtime.state, Event(Action.PROCESS_EXITED))
            return previous

    def outcome_settled(self, job_id: str) -> None:
        self._advance(job_id, Action.OUTCOME_SETTLED)

    def resources_released(self, job_id: str) -> None:
        self._advance(job_id, Action.RESOURCES_RELEASED)

    def unregister(self, job_id: str) -> None:
        self._advance(job_id, Action.UNREGISTERED)

    def completion_published(self, job_id: str) -> None:
        with self._lock:
            runtime = self._require(job_id)
            runtime.state = transition(runtime.state, Event(Action.COMPLETION_PUBLISHED))
            self._jobs.pop(job_id)

    def stop_intent(self, job_id: str) -> StopIntent:
        with self._lock:
            runtime = self._jobs.get(job_id)
            return runtime.state.stop_intent if runtime else StopIntent.NONE

    def process_may_be_running(self, job_id: str) -> bool:
        with self._lock:
            runtime = self._jobs.get(job_id)
            return bool(
                runtime
                and runtime.handle
                and runtime.state.phase in {Phase.LAUNCHING, Phase.RUNNING, Phase.RUNNER_STOPPED}
            )

    def registered_job_ids(self) -> list[str]:
        with self._lock:
            return [job_id for job_id, runtime in self._jobs.items() if runtime.state.registered]

    def state(self, job_id: str) -> State | None:
        with self._lock:
            runtime = self._jobs.get(job_id)
            return runtime.state if runtime else None

    def _advance(self, job_id: str, action: Action) -> None:
        with self._lock:
            runtime = self._require(job_id)
            runtime.state = transition(runtime.state, Event(action))

    def _require(self, job_id: str) -> _Runtime:
        runtime = self._jobs.get(job_id)
        if not runtime:
            raise RuntimeError(f"client app for job '{job_id}' is not registered")
        return runtime

    @staticmethod
    def _status(state: State) -> int:
        if state.phase == Phase.LAUNCHING:
            return ClientStatus.STARTING
        if state.phase == Phase.RUNNING:
            return ClientStatus.STARTED
        return ClientStatus.STOPPED
