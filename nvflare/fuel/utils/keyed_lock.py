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
import weakref


class KeyedLockRegistry:
    """Registry of in-process per-key locks that are garbage-collected when unused.

    Locks are stored in a WeakValueDictionary, so an entry is pruned automatically
    once no caller holds a strong reference to its lock. The reference returned by
    get() is what keeps the lock alive: callers must hold it for the whole time the
    lock is in use (e.g. ``with registry.get(key): ...``), otherwise two concurrent
    callers could end up with different lock objects for the same key.
    """

    def __init__(self):
        self._locks = weakref.WeakValueDictionary()
        self._guard = threading.Lock()

    def get(self, key) -> threading.Lock:
        """Returns the lock for the key, creating it if needed.

        The returned strong reference keeps the lock registered; hold it across use.
        """
        with self._guard:
            lock = self._locks.get(key)
            if lock is None:
                lock = threading.Lock()
                self._locks[key] = lock
            return lock

    def __contains__(self, key) -> bool:
        with self._guard:
            return key in self._locks
