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
