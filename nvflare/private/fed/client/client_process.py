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

"""Pure lifecycle for a client-owned job process."""

from dataclasses import dataclass, replace
from enum import Enum


class Phase(str, Enum):
    LAUNCHING = "launching"
    RUNNING = "running"
    RUNNER_STOPPED = "runner_stopped"
    EXITED = "exited"
    OUTCOME_SETTLED = "outcome_settled"
    RESOURCES_RELEASED = "resources_released"
    UNREGISTERED = "unregistered"
    DONE = "done"
    LAUNCH_FAILED = "launch_failed"


class StopIntent(str, Enum):
    NONE = "none"
    HEARTBEAT_CLEANUP = "heartbeat_cleanup"
    USER_ABORT = "user_abort"


class Action(str, Enum):
    ATTACH_HANDLE = "AttachHandle"
    WORKER_STARTED = "WorkerStarted"
    WORKER_STOPPED = "WorkerStopped"
    REQUEST_STOP = "RequestStop"
    PROCESS_EXITED = "ProcessExited"
    OUTCOME_SETTLED = "OutcomeSettled"
    RESOURCES_RELEASED = "ResourcesReleased"
    UNREGISTERED = "Unregistered"
    COMPLETION_PUBLISHED = "CompletionPublished"
    LAUNCH_FAILED = "LaunchFailed"


@dataclass(frozen=True)
class Event:
    action: Action
    stop_intent: StopIntent | None = None


@dataclass(frozen=True)
class State:
    phase: Phase = Phase.LAUNCHING
    handle_attached: bool = False
    stop_intent: StopIntent = StopIntent.NONE

    @property
    def registered(self) -> bool:
        return self.phase not in {Phase.UNREGISTERED, Phase.DONE, Phase.LAUNCH_FAILED}


class InvalidTransition(ValueError):
    pass


_OWNED = {Phase.LAUNCHING, Phase.RUNNING, Phase.RUNNER_STOPPED}
_STOP_RANK = {
    StopIntent.NONE: 0,
    StopIntent.HEARTBEAT_CLEANUP: 1,
    StopIntent.USER_ABORT: 2,
}


def transition(state: State, event: Event) -> State:
    """Apply one executable client-process lifecycle rule."""
    phase = state.phase
    match event.action:
        case Action.ATTACH_HANDLE if phase in _OWNED and not state.handle_attached:
            return replace(state, handle_attached=True)
        case Action.WORKER_STARTED if phase == Phase.LAUNCHING:
            return replace(state, phase=Phase.RUNNING)
        case Action.WORKER_STOPPED if phase in {Phase.LAUNCHING, Phase.RUNNING}:
            return replace(state, phase=Phase.RUNNER_STOPPED)
        case Action.REQUEST_STOP if phase in _OWNED and event.stop_intent in {
            StopIntent.HEARTBEAT_CLEANUP,
            StopIntent.USER_ABORT,
        }:
            return replace(state, stop_intent=_stronger_stop(state.stop_intent, event.stop_intent))
        case Action.PROCESS_EXITED if phase in _OWNED and state.handle_attached:
            return replace(state, phase=Phase.EXITED, stop_intent=StopIntent.NONE)
        case Action.OUTCOME_SETTLED if phase == Phase.EXITED:
            return replace(state, phase=Phase.OUTCOME_SETTLED)
        case Action.RESOURCES_RELEASED if phase == Phase.OUTCOME_SETTLED:
            return replace(state, phase=Phase.RESOURCES_RELEASED)
        case Action.UNREGISTERED if phase == Phase.RESOURCES_RELEASED:
            return replace(state, phase=Phase.UNREGISTERED)
        case Action.COMPLETION_PUBLISHED if phase == Phase.UNREGISTERED:
            return replace(state, phase=Phase.DONE)
        case Action.LAUNCH_FAILED if phase in _OWNED and not state.handle_attached:
            return replace(state, phase=Phase.LAUNCH_FAILED, stop_intent=StopIntent.NONE)
    raise InvalidTransition(f"cannot {event.action.value} while {phase.value}")


def _stronger_stop(current: StopIntent, candidate: StopIntent) -> StopIntent:
    return current if _STOP_RANK[current] >= _STOP_RANK[candidate] else candidate
