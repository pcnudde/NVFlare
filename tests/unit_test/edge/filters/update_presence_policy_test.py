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

import pytest

from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import ReturnCode, make_reply
from nvflare.edge.constants import EdgeTaskHeaderKey
from nvflare.edge.filters.add_update_header import AddUpdateHeader


@pytest.mark.parametrize("existing", [None, False, True])
def test_update_presence_default_never_overwrites_explicit_false(monkeypatch, existing):
    filter, payload = AddUpdateHeader(), make_reply(ReturnCode.OK)
    monkeypatch.setattr(filter, "log_info", Mock())
    if existing is not None:
        payload.set_header(EdgeTaskHeaderKey.HAS_UPDATE_DATA, existing)
    assert filter.process(payload, FLContext()) is payload
    assert payload.get_header(EdgeTaskHeaderKey.HAS_UPDATE_DATA) is (True if existing is None else existing)
    assert filter.log_info.call_count == int(existing is None)
    failed = make_reply(ReturnCode.EXECUTION_EXCEPTION)
    assert filter.process(failed, FLContext()) is failed
    assert failed.get_header(EdgeTaskHeaderKey.HAS_UPDATE_DATA) is None
