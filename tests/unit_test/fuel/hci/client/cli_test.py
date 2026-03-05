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

from unittest.mock import Mock

from nvflare.fuel.hci.client.api_status import APIStatus
from nvflare.fuel.hci.client.cli import AdminClient


def test_run_does_not_enter_prompt_when_login_fails(capsys):
    client = AdminClient.__new__(AdminClient)
    client.api = Mock()
    client.api.login.return_value = {"status": APIStatus.ERROR_AUTHENTICATION, "details": "bad token"}
    client.login_timeout = 10.0
    client.stopped = False
    client.debug = False
    client.cmdloop = Mock()

    client.run()

    client.api.connect.assert_called_once_with(10.0)
    client.cmdloop.assert_not_called()
    assert "Login failed: bad token" in capsys.readouterr().out
