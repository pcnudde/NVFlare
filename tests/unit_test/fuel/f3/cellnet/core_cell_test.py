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

import unittest
from unittest.mock import patch

from nvflare.fuel.f3.cellnet.core_cell import CoreCell


class TestCoreCellServerChildListener(unittest.TestCase):
    def setUp(self):
        self._saved_all_cells = dict(CoreCell.ALL_CELLS)
        CoreCell.ALL_CELLS.clear()

    def tearDown(self):
        CoreCell.ALL_CELLS.clear()
        CoreCell.ALL_CELLS.update(self._saved_all_cells)

    def test_server_child_with_root_url_creates_external_listener(self):
        CoreCell(fqcn="server", root_url="http://0:50001", secure=False, credentials={})
        child = CoreCell(fqcn="server.admin", root_url="http://0:50002", secure=False, credentials={})

        with (
            patch.object(child, "_create_external_listener") as create_external_listener,
            patch.object(child.communicator, "start") as communicator_start,
        ):
            child.start()

        create_external_listener.assert_called_once_with("http://0:50002")
        communicator_start.assert_called_once()

    def test_server_child_with_external_listener_is_backbone_ready(self):
        child = CoreCell(fqcn="server.admin", root_url="http://0:50003", secure=False, credentials={})
        child.ext_listeners["http://0:50003"] = object()
        child.running = True

        assert child.is_backbone_ready() is True
