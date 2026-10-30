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

from nvflare.apis import dxo_filter as subject
from nvflare.apis.dxo import DXO, DataKind, from_shareable
from nvflare.apis.fl_constant import ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable, make_reply


class IncrementFilter(subject.DXOFilter):
    def __init__(self, mode="replace", supported=None, selected=None):
        super().__init__([DataKind.WEIGHTS] if supported is None else supported, selected)
        self.mode = mode

    def process_dxo(self, dxo, shareable, fl_ctx):
        if self.mode == "skip":
            return None
        if self.mode == "invalid":
            return "not a DXO"
        return DXO(dxo.data_kind, {"value": dxo.data["value"] + 1})


def test_nested_list_and_mapping_filter_preserves_unselected_data_and_copies_filter_history(monkeypatch):
    audit = Mock()
    monkeypatch.setattr(subject, "add_job_audit_event", audit)
    weights = DXO(DataKind.WEIGHTS, {"value": 2})
    weights.add_filter_history("previous")
    metrics = DXO(DataKind.METRICS, {"value": 4})
    nested = DXO(DataKind.COLLECTION, {"weights": weights, "metrics": metrics})
    data = ["metadata", nested]
    filter = IncrementFilter()
    filter._filter_dxos(data, Shareable(), FLContext())
    assert data[0] == "metadata"
    assert nested.data["weights"].data == {"value": 3}
    assert nested.data["weights"].get_filter_history() == ["previous", "IncrementFilter"]
    assert nested.data["metrics"] is metrics
    audit.assert_called_once()
    assert "IncrementFilter" in audit.call_args.kwargs["msg"]


@pytest.mark.parametrize("mode", ["skip", "invalid"])
def test_filter_return_contract_rejects_non_dxo_and_keeps_unchanged_data_on_skip(monkeypatch, mode):
    audit = Mock()
    monkeypatch.setattr(subject, "add_job_audit_event", audit)
    value = DXO(DataKind.WEIGHTS, {"value": 2})
    payload = value.to_shareable()
    if mode == "invalid":
        with pytest.raises(RuntimeError, match="must be DXO"):
            IncrementFilter(mode).process(payload, FLContext())
    else:
        assert from_shareable(IncrementFilter(mode).process(payload, FLContext())).data == {"value": 2}
    audit.assert_not_called()


def test_non_dxo_failed_task_and_empty_payload_are_unchanged():
    filter, ctx = IncrementFilter(), FLContext()
    for payload in [
        Shareable({"plain": True}),
        make_reply(ReturnCode.EXECUTION_EXCEPTION),
        DXO(DataKind.WEIGHTS, {}).to_shareable(),
    ]:
        assert filter.process(payload, ctx) is payload
    with pytest.raises(ValueError, match="dict or list"):
        filter._filter_dxos((1, 2), Shareable(), ctx)
    with pytest.raises(AssertionError):
        filter._filter_dxos({"invalid": 1}, Shareable(), ctx)


def test_filter_configuration_rejects_unsupported_kinds_and_non_list_selection():
    with pytest.raises(ValueError, match="supported_data_kinds"):
        IncrementFilter(supported="weights")
    with pytest.raises(ValueError, match="data_kinds_to_filter"):
        IncrementFilter(selected="weights")
    with pytest.raises(ValueError, match="invalid data kinds"):
        IncrementFilter(selected=[DataKind.METRICS])
