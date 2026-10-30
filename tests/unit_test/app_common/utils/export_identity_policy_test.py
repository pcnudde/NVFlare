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

import pytest

from nvflare.apis.fl_constant import ConnPropKey, FLMetaKey, ReservedKey
from nvflare.apis.fl_context import FLContext
from nvflare.app_common.utils import export_utils as subject


@pytest.mark.parametrize("connections", [False, True])
def test_export_identity_overwrites_stale_auth_and_adds_only_available_connection_properties(monkeypatch, connections):
    ctx = FLContext()
    ctx.set_prop(ReservedKey.IDENTITY_NAME, "site")
    ctx.set_prop(ReservedKey.RUN_NUM, "job")
    values = {FLMetaKey.AUTH_TOKEN: "token", FLMetaKey.AUTH_TOKEN_SIGNATURE: "signature"}
    if connections:
        values.update(
            {
                key: {"url": key}
                for key in (ConnPropKey.ROOT_CONN_PROPS, ConnPropKey.CP_CONN_PROPS, ConnPropKey.RELAY_CONN_PROPS)
            }
        )
    monkeypatch.setattr(subject, "get_scope_property", lambda scope_name, key, default=None: values.get(key, default))
    exported = {"existing": True, FLMetaKey.AUTH_TOKEN: "stale"}
    subject.update_export_props(exported, ctx)
    assert exported[FLMetaKey.SITE_NAME] == "site" and exported[FLMetaKey.JOB_ID] == "job"
    assert exported[FLMetaKey.AUTH_TOKEN] == "token" and exported[FLMetaKey.AUTH_TOKEN_SIGNATURE] == "signature"
    assert exported["existing"] is True
    for key in (ConnPropKey.ROOT_CONN_PROPS, ConnPropKey.CP_CONN_PROPS, ConnPropKey.RELAY_CONN_PROPS):
        assert (key in exported) is connections
        if connections:
            assert exported[key] is values[key]
