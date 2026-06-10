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

import gc
import threading

from nvflare.fuel.utils.keyed_lock import KeyedLockRegistry


class TestKeyedLockRegistry:
    def test_same_key_returns_same_lock_while_referenced(self):
        registry = KeyedLockRegistry()
        lock = registry.get("key-1")
        assert registry.get("key-1") is lock
        assert "key-1" in registry

    def test_different_keys_get_different_locks(self):
        registry = KeyedLockRegistry()
        lock_a = registry.get("key-a")
        lock_b = registry.get("key-b")
        assert lock_a is not lock_b

    def test_tuple_keys_are_supported(self):
        registry = KeyedLockRegistry()
        key = ("study", "user", "token")
        lock = registry.get(key)
        assert registry.get(("study", "user", "token")) is lock

    def test_lock_is_collected_after_release(self):
        registry = KeyedLockRegistry()
        lock = registry.get("key-1")
        del lock
        gc.collect()
        assert "key-1" not in registry

    def test_returned_lock_is_usable(self):
        registry = KeyedLockRegistry()
        with registry.get("key-1"):
            # while held, another get must hand back the very same (now locked) lock
            other = registry.get("key-1")
            assert isinstance(other, type(threading.Lock()))
            assert other.locked()
