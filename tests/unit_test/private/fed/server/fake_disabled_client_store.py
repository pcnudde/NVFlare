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

"""Shared in-memory fake of the StateStore disabled-client methods for server unit tests."""


class FakeDisabledClientStore:
    def __init__(self):
        self.disabled = {}
        self.disable_error = None
        self.enable_error = None
        self.get_error = None
        self.get_calls = []  # client_name per get_disabled_client call, for spying on cache behavior

    def get_disabled_client(self, client_name):
        self.get_calls.append(client_name)
        if self.get_error:
            raise self.get_error
        return self.disabled.get(client_name)

    def disable_client(self, client_name, disabled_by=None, reason=None):
        if self.disable_error:
            raise self.disable_error
        row = {"client_name": client_name, "disabled_by": disabled_by, "reason": reason}
        self.disabled[client_name] = row
        return row

    def enable_client(self, client_name):
        if self.enable_error:
            raise self.enable_error
        return self.disabled.pop(client_name, None) is not None
