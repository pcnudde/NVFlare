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

from collections import namedtuple
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from nvflare.fuel.utils import fobs
from nvflare.fuel.utils.fobs.datum import DatumManager
from nvflare.fuel.utils.fobs.decomposers import via_downloader as subject


class SlottedValue:
    __slots__ = ("__hidden", "unset")

    def __init__(self, value):
        self.__hidden = value

    @property
    def value(self):
        return self.__hidden


class SingleSlot:
    __slots__ = "value"


class NativeTextDownloader(subject.ViaDownloaderDecomposer):
    def __init__(self):
        super().__init__(10, "TEST")

    def supported_type(self):
        return str

    def to_downloadable(self, items, max_chunk_size, fobs_ctx):
        raise AssertionError("no download transaction should be created by these cases")

    def download(self, **kwargs):
        raise AssertionError("no transport should be started by these cases")

    def get_download_dot(self):
        return 7654

    def native_decompose(self, target, manager=None):
        return target.encode()

    def native_recompose(self, data, manager=None):
        return data.decode()


def reference():
    return subject.LazyDownloadRef("server", "batch", "T0", 7654)


def test_graph_replacement_preserves_cycles_aliases_slots_and_tuple_types():
    ref = reference()
    Pair = namedtuple("Pair", "first second")
    slotted, single = SlottedValue(ref), SingleSlot()
    single.value = ref
    plain = SimpleNamespace(value=ref)
    data = {
        "tuple": Pair(ref, "kept"),
        "set": {ref},
        "frozen": frozenset({ref}),
        "objects": [slotted, single, plain],
        "excluded": ref,
    }
    data["cycle"] = data
    assert subject.contains_lazy_download_ref(data)
    result = subject._replace_lazy_download_refs(data, {id(ref): "materialized"}, {}, {"excluded"})
    assert result is data and data["cycle"] is data
    assert data["tuple"] == Pair("materialized", "kept") and isinstance(data["tuple"], Pair)
    assert data["set"] == {"materialized"} and data["frozen"] == frozenset({"materialized"})
    assert slotted.value == single.value == plain.value == "materialized"
    assert data["excluded"] is ref
    assert not subject.contains_lazy_download_ref(data, excluded_dict_keys={"excluded"})


def test_invalid_replacement_key_does_not_partially_mutate_mapping():
    ref = reference()
    data = {ref: "original", "other": 3}
    with pytest.raises(TypeError, match="unhashable"):
        subject._replace_lazy_download_refs(data, {id(ref): []}, {})
    assert data == {ref: "original", "other": 3}


def test_materialization_downloads_each_batch_once_and_returns_cleanup_ownership(monkeypatch):
    first, second = reference(), reference()
    second.item_id = "T1"

    class Items(dict):
        def cleanup(self):
            raise AssertionError("caller owns cleanup")

    items = Items(T0="one", T1="two")
    handler = Mock(spec=subject.ViaDownloaderDecomposer)
    handler._download_from_remote_cell.return_value = items
    resolve = Mock(return_value=handler)
    monkeypatch.setattr(subject.fobs, "get_dot_handler", resolve)
    cell = Mock()
    cell.get_fobs_context.return_value = {"download": "context"}
    owners = []
    abort = object()
    graph = [first, {"second": second, "same": first}]
    assert subject.materialize_lazy_download_refs(graph, cell, owners=owners, abort_signal=abort) == [
        "one",
        {"second": "two", "same": "one"},
    ]
    cell.get_fobs_context.assert_called_once_with(
        props={
            fobs.FOBSContextKey.PASS_THROUGH: False,
            fobs.FOBSContextKey.TENSOR_DISK_OFFLOAD: False,
            fobs.FOBSContextKey.ABORT_SIGNAL: abort,
        }
    )
    handler._download_from_remote_cell.assert_called_once_with(
        {"download": "context"}, {"fqcn": "server", "ref_id": "batch"}
    )
    resolve.assert_called_once_with(7654)
    assert owners == [items]


@pytest.mark.parametrize(
    "items, message", [(object(), "invalid type"), ({}, "missing item"), ({"T0": None}, "no value")]
)
def test_invalid_downloaded_batch_is_rejected_before_graph_mutation(monkeypatch, items, message):
    ref = reference()
    handler = Mock(spec=subject.ViaDownloaderDecomposer)
    handler._download_from_remote_cell.return_value = items
    monkeypatch.setattr(subject.fobs, "get_dot_handler", Mock(return_value=handler))
    graph = [ref]
    with pytest.raises(RuntimeError, match=message):
        subject.materialize_lazy_download_refs(graph, Mock())
    assert graph == [ref]


def test_materialization_rejects_missing_cell_or_handler_without_rewriting_reference(monkeypatch):
    ref = reference()
    with pytest.raises(RuntimeError, match="Cell is unavailable"):
        subject.materialize_lazy_download_refs([ref], None)
    monkeypatch.setattr(subject.fobs, "get_dot_handler", Mock(return_value=None))
    with pytest.raises(RuntimeError, match="no download handler"):
        subject.materialize_lazy_download_refs([ref], Mock())
    concrete = {"kept": [1, 2]}
    assert subject.materialize_lazy_download_refs(concrete, None) is concrete


@pytest.mark.parametrize(
    "data, message",
    [
        (None, "should be dict"),
        ({}, "missing 'data'"),
        ({"type": "unknown", "data": "x"}, "invalid enc_type"),
        ({"type": "ref", "data": 1}, "ref data must be str"),
    ],
)
def test_recompose_validates_wire_shape_before_downloading(data, message, caplog):
    decoder = NativeTextDownloader()
    with pytest.raises(RuntimeError):
        decoder.recompose(data, DatumManager(fobs_ctx={}))
    assert message in caplog.text


@pytest.mark.parametrize(
    "receivers,count,expected",
    [
        (1, 1, None),
        ([], 1, None),
        (["a", "b"], 1, None),
        (["a", "a"], 2, ("a",)),
        ([1, None], 2, ("1", None)),
        ("one", 1, ("one",)),
    ],
)
def test_upload_receiver_identity_normalization_rejects_inconsistent_budgets(receivers, count, expected):
    decoder = NativeTextDownloader()
    assert (
        decoder._get_result_upload_receiver_ids({subject.RESULT_UPLOAD_RECEIVER_IDS_CTX_KEY: receivers}, count)
        == expected
    )


def test_native_recomposition_and_reference_transaction_callback_errors_are_isolated(caplog):
    decoder = NativeTextDownloader()
    assert decoder.recompose({"type": "native", "data": b"hello"}, DatumManager(fobs_ctx={})) == "hello"
    callback = Mock(side_effect=RuntimeError("callback failed"))
    info = subject.DownloadTransactionInfo("transaction", (("site", "identity"),), 1.0)
    subject._notify_download_transaction_created(
        {subject.RESULT_UPLOAD_TX_CREATED_CB_CTX_KEY: callback}, info, decoder.logger
    )
    callback.assert_called_once_with(info)
    assert "callback failed" in caplog.text
