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

import sys
from types import SimpleNamespace

from nvflare.fuel.utils.obj_utils import get_size


def test_repeated_reference_is_counted_once_and_cycles_are_bounded():
    payload = bytearray(b"payload")
    values = [payload, payload]
    assert get_size(values) == sys.getsizeof(values) + sys.getsizeof(payload)
    values.append(values)
    assert get_size(values) == sys.getsizeof(values) + sys.getsizeof(payload)
    assert get_size(values, seen={id(values)}) == 0


def test_mapping_keys_instance_attributes_and_iterables_contribute_to_size():
    payload = b"example payload"
    key = "key"
    value = {key: payload}
    assert get_size(value) == sys.getsizeof(value) + sys.getsizeof(key) + sys.getsizeof(payload)
    box = SimpleNamespace(value=payload)
    assert get_size(box) == sum(sys.getsizeof(x) for x in [box, box.__dict__, "value", payload])
    pair = (payload,)
    assert get_size(pair) == sys.getsizeof(pair) + sys.getsizeof(payload)
    assert get_size(payload) == sys.getsizeof(payload)
