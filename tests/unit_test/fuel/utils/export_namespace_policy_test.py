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

from nvflare.fuel.utils.attributes_exportable import AttributesExportable, ExportMode, export_components


class ExportedComponent(AttributesExportable):
    def __init__(self, section):
        self.section = section
        self.modes = []

    def export(self, mode):
        self.modes.append(mode)
        return self.section, {"public": True}


def test_component_exports_keep_section_names_and_requested_mode():
    one, two = ExportedComponent("first"), ExportedComponent("second")
    assert export_components({"id1": one, "id2": two}, ["system"], ExportMode.PEER) == {
        "first": {"public": True},
        "second": {"public": True},
    }
    assert one.modes == two.modes == [ExportMode.PEER]


@pytest.mark.parametrize(
    "components,reserved,error,message",
    [
        ({"one": ExportedComponent("section"), "two": ExportedComponent("section")}, [], RuntimeError, "duplicated"),
        ({"one": ExportedComponent("system")}, ["system"], RuntimeError, "reserved key"),
        ({"one": object()}, [], TypeError, "one"),
    ],
)
def test_duplicate_reserved_or_nonexportable_components_are_rejected(components, reserved, error, message):
    with pytest.raises(error, match=message):
        export_components(components, reserved, ExportMode.SELF)
