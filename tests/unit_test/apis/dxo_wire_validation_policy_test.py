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

import pytest

from nvflare.apis import dxo as subject
from nvflare.apis.dxo import DXO, DataKind
from nvflare.apis.shareable import ReservedHeaderKey, Shareable
from nvflare.fuel.utils import fobs


@pytest.mark.parametrize("encoded", [[], "invalid", 1])
def test_wire_dictionary_rejects_non_mapping_payload(encoded):
    with pytest.raises(ValueError, match="encoded value must be dict"):
        subject.from_dict(encoded)


@pytest.mark.parametrize("payload", [None, "not a dictionary"])
def test_shareable_rejects_missing_or_mistyped_dxo_content(payload):
    envelope = Shareable()
    envelope.set_header(ReservedHeaderKey.CONTENT_TYPE, "DXO")
    if payload is not None:
        envelope["DXO"] = payload
    with pytest.raises(ValueError, match="missing content|should be encoded as dict"):
        subject.from_shareable(envelope)


def test_wire_roundtrip_and_non_dxo_serialization_are_distinguished(tmp_path):
    value = DXO(DataKind.WEIGHTS, None)
    value.set_meta_prop("round", 2)
    data = value.to_bytes()
    restored = subject.from_bytes(data)
    assert (restored.data_kind, restored.data, restored.meta) == (DataKind.WEIGHTS, {}, {"round": 2})
    path = str(tmp_path / "model.fobs")
    value.to_file(path)
    assert subject.from_file(path).to_dict() == value.to_dict()
    with pytest.raises(ValueError, match="do not represent a valid DXO"):
        subject.from_bytes(fobs.dumps({"plain": True}))


def test_mutated_dxo_validation_and_meta_reinitialization():
    value = DXO(DataKind.WEIGHTS, {})
    value.meta = None
    assert value.get_meta_prop("absent", 5) == 5
    value.set_meta_prop("updated", 1)
    assert value.get_meta_props() == {"updated": 1}
    value.data = None
    assert value.validate() == "missing data"
    value.data = {}
    value.meta = []
    assert "invalid props" in value.validate()
