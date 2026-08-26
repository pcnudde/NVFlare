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

import threading
from unittest.mock import MagicMock

import nvflare.private.fed.client.client_process_driver as driver_module
from nvflare.private.fed.client.client_process import Action, Phase, StopIntent
from nvflare.private.fed.client.client_process_driver import ClientProcessDriver
from nvflare.private.fed.client.client_status import ClientStatus


def test_stronger_pre_attach_stop_is_returned_when_handle_attaches():
    driver = ClientProcessDriver()
    driver.register("job")

    driver.request_stop("job", StopIntent.HEARTBEAT_CLEANUP)
    driver.request_stop("job", StopIntent.USER_ABORT)

    assert driver.attach_handle("job", MagicMock()) == StopIntent.USER_ABORT


def test_concurrent_pre_attach_stops_cannot_weaken_user_abort():
    driver = ClientProcessDriver()
    driver.register("job")
    barrier = threading.Barrier(3)

    def request(intent):
        barrier.wait()
        driver.request_stop("job", intent)

    threads = [
        threading.Thread(target=request, args=(StopIntent.HEARTBEAT_CLEANUP,)),
        threading.Thread(target=request, args=(StopIntent.USER_ABORT,)),
    ]
    for thread in threads:
        thread.start()
    barrier.wait()
    for thread in threads:
        thread.join()

    assert driver.attach_handle("job", MagicMock()) == StopIntent.USER_ABORT


def test_runner_stopped_remains_owned_until_process_exit():
    driver = ClientProcessDriver()
    driver.register("job")
    driver.attach_handle("job", MagicMock())
    driver.record_worker_status("job", ClientStatus.STOPPED)

    assert driver.state("job").phase == Phase.RUNNER_STOPPED
    assert driver.process_may_be_running("job")
    assert driver.registered_job_ids() == ["job"]

    driver.process_exited("job")

    assert not driver.process_may_be_running("job")
    assert driver.registered_job_ids() == ["job"]


def test_duplicate_and_late_worker_notifications_do_not_regress_status():
    driver = ClientProcessDriver()
    driver.register("job")
    driver.record_worker_status("job", ClientStatus.STARTED)
    driver.record_worker_status("job", ClientStatus.STARTED)
    driver.record_worker_status("job", ClientStatus.STOPPED)
    driver.record_worker_status("job", ClientStatus.STARTED)

    assert driver.status("job", ClientStatus.NOT_STARTED) == ClientStatus.STOPPED


def test_cleanup_calls_map_to_the_checked_action_order(monkeypatch):
    actions = []
    real_transition = driver_module.transition

    def recording_transition(state, event):
        actions.append(event.action)
        return real_transition(state, event)

    monkeypatch.setattr(driver_module, "transition", recording_transition)
    driver = ClientProcessDriver()
    driver.register("job")
    driver.attach_handle("job", MagicMock())
    driver.process_exited("job")
    driver.outcome_settled("job")
    driver.resources_released("job")
    driver.unregister("job")

    assert driver.registered_job_ids() == []
    assert driver.state("job").phase == Phase.UNREGISTERED

    driver.completion_published("job")

    assert actions == [
        Action.ATTACH_HANDLE,
        Action.PROCESS_EXITED,
        Action.OUTCOME_SETTLED,
        Action.RESOURCES_RELEASED,
        Action.UNREGISTERED,
        Action.COMPLETION_PUBLISHED,
    ]
    assert driver.state("job") is None
