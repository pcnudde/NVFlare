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

from unittest.mock import Mock

from nvflare.apis.event_type import EventType
from nvflare.apis.fl_context import FLContext
from nvflare.app_common.resource_managers import auto_clean_resource_manager as subject
from nvflare.app_common.resource_managers.list_resource_manager import ListResourceManager
from tests.timing_utils import isolate_time


def test_expiration_returns_only_expired_reservation_with_bounded_cleanup_iterations(monkeypatch):
    manager = ListResourceManager({"gpu": [0, 1]}, expiration_period=2)
    available, token = manager.check_resources({"gpu": 1}, FLContext())
    assert available
    assert list(manager.resources["gpu"]) == [1]
    manager._stop_event = Mock()
    manager._stop_event.is_set.side_effect = [False, False, True]
    clock = isolate_time(monkeypatch, subject)
    clock.sleep = Mock()
    manager._check_expired()
    assert manager.reserved_resources == {}
    assert sorted(manager.resources["gpu"]) == [0, 1]
    assert clock.sleep.call_count == 2
    assert token not in manager.reserved_resources


def test_system_events_start_and_join_owned_cleanup_thread_once(monkeypatch):
    thread = Mock()
    monkeypatch.setattr(subject, "Thread", Mock(return_value=thread))
    manager = ListResourceManager({"gpu": []})
    manager.handle_event(EventType.SYSTEM_START, FLContext())
    thread.start.assert_called_once()
    manager.handle_event(EventType.SYSTEM_END, FLContext())
    manager.handle_event(EventType.SYSTEM_END, FLContext())
    thread.join.assert_called_once()
    assert manager._stop_event.is_set()
    assert manager._cleanup_thread is None
