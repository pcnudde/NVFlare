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

from unittest.mock import MagicMock

import pytest
import tenseal as ts

from nvflare.apis.event_type import EventType
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.app_opt.xgboost.histogram_based_v2.defs import Constant
from nvflare.app_opt.xgboost.histogram_based_v2.sec import client_handler as client_module
from nvflare.app_opt.xgboost.histogram_based_v2.sec.client_handler import ClientSecurityHandler
from nvflare.app_opt.xgboost.histogram_based_v2.sec.data_converter import AggregationContext, FeatureContext
from nvflare.app_opt.xgboost.histogram_based_v2.sec.partial_he import util
from nvflare.app_opt.xgboost.histogram_based_v2.sec.processor_data_converter import ProcessorDataConverter
from nvflare.app_opt.xgboost.histogram_based_v2.sec.server_handler import ServerSecurityHandler


def _context(rank=0, seq=3, request=None, **props):
    ctx = FLContext()
    for key, value in {
        Constant.PARAM_KEY_RANK: rank,
        Constant.PARAM_KEY_SEQ: seq,
        Constant.PARAM_KEY_REQUEST: request or Shareable(),
        **props,
    }.items():
        ctx.set_prop(key, value, sticky=False)
    return ctx


def test_secure_server_broadcast_substitutes_dummy_and_returns_ciphertext_only_to_non_root():
    server = ServerSecurityHandler()
    request = Shareable()
    request.set_header(Constant.HEADER_KEY_ENCRYPTED_DATA, True)
    request.set_header(Constant.HEADER_KEY_ORIGINAL_BUF_SIZE, 20)
    ctx = _context(rank=0, request=request, **{Constant.PARAM_KEY_SEND_BUF: "ciphertext"})
    server._process_before_broadcast(ctx)
    assert len(ctx.get_prop(Constant.PARAM_KEY_SEND_BUF)) == Constant.DUMMY_BUFFER_SIZE
    assert server.encrypted_gh == "ciphertext"
    reply = Shareable()
    ctx.set_prop(Constant.PARAM_KEY_REPLY, reply, sticky=False)
    server._process_after_broadcast(ctx)
    assert ctx.get_prop(Constant.PARAM_KEY_RCV_BUF) is None
    assert reply.get_header(Constant.HEADER_KEY_ORIGINAL_BUF_SIZE) == 20
    ctx.set_prop(Constant.PARAM_KEY_RANK, 1, sticky=False)
    server._process_after_broadcast(ctx)
    assert ctx.get_prop(Constant.PARAM_KEY_RCV_BUF) == "ciphertext"
    ctx.set_prop(Constant.PARAM_KEY_SEQ, 4, sticky=False)
    ctx.set_prop(Constant.PARAM_KEY_RCV_BUF, b"untouched", sticky=False)
    server._process_after_broadcast(ctx)
    assert ctx.get_prop(Constant.PARAM_KEY_RCV_BUF) == b"untouched"


@pytest.mark.parametrize("horizontal", [False, True])
def test_server_gather_combines_encrypted_vectors_or_preserves_rank_mapping(horizontal):
    server = ServerSecurityHandler()
    contexts = []
    values = []
    if horizontal:
        crypto = ts.context(ts.SCHEME_TYPE.CKKS, 4096, coeff_mod_bit_sizes=[30, 20, 30], n_threads=1)
        crypto.global_scale = 2**20
        values = [ts.ckks_vector(crypto, [1, 2]), ts.ckks_vector(crypto, [3, 4])]
    else:
        values = ["rank-zero", "rank-one"]
    for rank, value in enumerate(values):
        request = Shareable()
        request.set_header(Constant.HEADER_KEY_ENCRYPTED_DATA, True)
        request.set_header(Constant.HEADER_KEY_HORIZONTAL, horizontal)
        request.set_header(Constant.HEADER_KEY_ORIGINAL_BUF_SIZE, 30)
        ctx = _context(rank=rank, request=request, **{Constant.PARAM_KEY_SEND_BUF: value})
        server._process_before_all_gather_v(ctx)
        assert len(ctx.get_prop(Constant.PARAM_KEY_SEND_BUF)) == Constant.DUMMY_BUFFER_SIZE
        ctx.set_prop(Constant.PARAM_KEY_REPLY, Shareable(), sticky=False)
        contexts.append(ctx)
    for ctx in contexts:
        server._process_after_all_gather_v(ctx)
        output = ctx.get_prop(Constant.PARAM_KEY_RCV_BUF)
        if horizontal:
            assert output.decrypt() == pytest.approx([4, 6], abs=0.02)
        else:
            assert output == {0: "rank-zero", 1: "rank-one"}
        reply = ctx.get_prop(Constant.PARAM_KEY_REPLY)
        assert reply.get_header(Constant.HEADER_KEY_WORLD_SIZE) == 2
        assert reply.get_header(Constant.HEADER_KEY_SIZE_DICT) == {0: 30, 1: 30}
    assert server.aggr_result_dict is None


def test_server_ignores_unsecured_collectives_and_aborts_when_secure_gather_has_no_data():
    server = ServerSecurityHandler()
    ctx = _context(**{Constant.PARAM_KEY_SEND_BUF: b"plain"})
    server._process_before_broadcast(ctx)
    server._process_before_all_gather_v(ctx)
    server._process_after_all_gather_v(ctx)
    assert ctx.get_prop(Constant.PARAM_KEY_SEND_BUF) == b"plain"
    server._abort = MagicMock()
    ctx.set_prop(Constant.HEADER_KEY_IN_AGGR, True, sticky=False)
    ctx.set_prop(Constant.PARAM_KEY_REPLY, Shareable(), sticky=False)
    server._process_after_all_gather_v(ctx)
    assert "no aggr result" in server._abort.call_args.args[0]


@pytest.fixture
def client():
    client = ClientSecurityHandler()
    client.data_converter = MagicMock(spec=ProcessorDataConverter)
    client._abort = MagicMock()
    return client


def test_client_encrypts_only_root_gradient_broadcast_and_restores_root_buffer(client, monkeypatch):
    ctx = _context(rank=0, **{Constant.PARAM_KEY_ROOT: 0, Constant.PARAM_KEY_SEND_BUF: b"gradients"})
    client.data_converter.decode_gh_pairs.return_value = [(2, 3), (4, 5)]
    client._process_before_broadcast(ctx)
    client._abort.assert_called_once()
    client.encryptor = MagicMock()
    client.encryptor.encrypt.return_value = ["enc-a", "enc-b"]
    encoder = MagicMock(return_value="encoded-gradients")
    monkeypatch.setattr(client_module, "encode_encrypted_data", encoder)
    client._process_before_broadcast(ctx)
    assert client.clear_ghs == [util.combine(2, 3), util.combine(4, 5)]
    client.encryptor.encrypt.assert_called_once_with(client.clear_ghs)
    assert ctx.get_prop(Constant.PARAM_KEY_SEND_BUF) == "encoded-gradients"
    reply = Shareable()
    reply.set_header(Constant.HEADER_KEY_ENCRYPTED_DATA, True)
    ctx.set_prop(Constant.PARAM_KEY_REPLY, reply, sticky=False)
    client._process_after_broadcast(ctx)
    assert ctx.get_prop(Constant.PARAM_KEY_RCV_BUF) == b"gradients"
    ctx.set_prop(Constant.PARAM_KEY_RANK, 1, sticky=False)
    client._process_before_broadcast(ctx)
    assert client.encryptor.encrypt.call_count == 1


def test_non_label_broadcast_receives_ciphertext_and_sends_size_preserving_dummy(client, monkeypatch):
    ctx = _context(rank=1, **{Constant.PARAM_KEY_RCV_BUF: "wire"})
    reply = Shareable()
    reply.set_header(Constant.HEADER_KEY_ENCRYPTED_DATA, True)
    reply.set_header(Constant.HEADER_KEY_ORIGINAL_BUF_SIZE, 15)
    ctx.set_prop(Constant.PARAM_KEY_REPLY, reply, sticky=False)
    decoder = MagicMock(return_value=("key", ["encrypted-gradient"]))
    monkeypatch.setattr(client_module, "decode_encrypted_data", decoder)
    client._process_after_broadcast(ctx)
    decoder.assert_called_once_with("wire")
    assert client.public_key == "key" and client.encrypted_ghs == ["encrypted-gradient"]
    assert len(ctx.get_prop(Constant.PARAM_KEY_RCV_BUF)) == 15


@pytest.mark.parametrize("groups", [None, {7: [0]}])
def test_label_client_aggregates_clear_gradients_by_feature_bins_and_groups(client, groups):
    ctx = _context(**{Constant.PARAM_KEY_SEND_BUF: b"aggregation"})
    client.clear_ghs = [util.combine(2, 3), util.combine(4, 5)]
    client.data_converter.decode_aggregation_context.return_value = AggregationContext(
        [FeatureContext(2, [0, 1], 2)], groups
    )
    client._process_before_all_gather_v_vertical(ctx)
    expected = [util.combine(2, 3), 0 if groups else util.combine(4, 5)]
    assert client.aggr_result == [(2, 7 if groups else 0, expected)]
    assert ctx.get_prop(Constant.PARAM_KEY_SEND_BUF) is None
    assert ctx.get_prop(Constant.PARAM_KEY_HEADERS)[Constant.HEADER_KEY_ENCRYPTED_DATA]


def test_non_label_gather_encrypts_feature_aggregation_and_rejects_missing_features(client):
    ctx = _context(rank=1, **{Constant.PARAM_KEY_SEND_BUF: b"aggregate"})
    client.data_converter.decode_aggregation_context.return_value = AggregationContext([], {})
    client._process_before_all_gather_v_vertical(ctx)
    assert "missing features" in client._abort.call_args.args[0]
    client.data_converter.decode_aggregation_context.return_value = AggregationContext([FeatureContext(1, [0], 1)], {})
    client._process_before_all_gather_v_vertical(ctx)
    assert "no encrypted" in client._abort.call_args.args[0]
    client.encrypted_ghs = ["encrypted"]
    client.adder = MagicMock()
    client.adder.add.return_value = [(1, 0, ["encrypted-sum"])]
    client._process_before_all_gather_v_vertical(ctx)
    assert ctx.get_prop(Constant.PARAM_KEY_SEND_BUF) == '[[1, 0, ["encrypted-sum"]]]'
    client.adder.add.assert_called_once_with(["encrypted"], [(1, [0], 1)], [], encode_sum=True)


def test_vertical_gather_reconstructs_sorted_histograms_pads_and_hides_non_label_results(client):
    ctx = _context(rank=0)
    reply = Shareable()
    reply.set_header(Constant.HEADER_KEY_ENCRYPTED_DATA, True)
    reply.set_header(Constant.HEADER_KEY_SIZE_DICT, {0: 10, 1: 10})
    ctx.set_prop(Constant.PARAM_KEY_REPLY, reply, sticky=False)
    ctx.set_prop(Constant.PARAM_KEY_RCV_BUF, {0: None, 1: [(2, 0, [util.combine(4, 5)])]}, sticky=False)
    client.clear_ghs = [1]
    client.aggr_result = [(1, 0, [util.combine(2, 3)])]
    encoded = []

    def encode(results, context):
        encoded.append(
            {gid: [(item.feature_id, list(item.aggregated_hist)) for item in items] for gid, items in results.items()}
        )
        return b"hist"

    client.data_converter.encode_aggregation_result.side_effect = encode
    client._process_after_all_gather_v_vertical(ctx)
    assert ctx.get_prop(Constant.PARAM_KEY_RCV_BUF) == b"histhist" + b"\x00" * 12
    assert encoded[0] == {0: [(1, [(2, 3)]), (2, [(4, 5)])]}
    assert encoded[1] == {0: [(1, [(0, 0)]), (2, [(0, 0)])]}
    client.clear_ghs = None
    ctx.set_prop(Constant.PARAM_KEY_RCV_BUF, {0: None, 1: "hidden"}, sticky=False)
    client._process_after_all_gather_v_vertical(ctx)
    assert len(ctx.get_prop(Constant.PARAM_KEY_RCV_BUF)) == 20
    ctx.set_prop(Constant.PARAM_KEY_RCV_BUF, b"invalid", sticky=False)
    client._process_after_all_gather_v_vertical(ctx)
    assert "expect a dict" in client._abort.call_args.args[0]


def test_horizontal_gather_encrypts_decrypts_and_appends_zero_worker_payloads(client):
    client.tenseal_context = ts.context(ts.SCHEME_TYPE.CKKS, 4096, coeff_mod_bit_sizes=[30, 20, 30], n_threads=1)
    client.tenseal_context.global_scale = 2**20
    client.data_converter.decode_histograms.return_value = [1.5, 2.5]
    ctx = _context(**{Constant.PARAM_KEY_SEND_BUF: b"histograms"})
    client._process_before_all_gather_v_horizontal(ctx)
    vector = ctx.get_prop(Constant.PARAM_KEY_SEND_BUF)
    assert vector.decrypt() == pytest.approx([1.5, 2.5], abs=0.02)
    ctx.set_prop(Constant.PARAM_KEY_RCV_BUF, vector, sticky=False)
    reply = Shareable()
    reply.set_header(Constant.HEADER_KEY_WORLD_SIZE, 2)
    ctx.set_prop(Constant.PARAM_KEY_REPLY, reply, sticky=False)
    client.data_converter.encode_histograms_result.side_effect = [b"hist", b"zero"]
    client._process_after_all_gather_v_horizontal(ctx)
    assert ctx.get_prop(Constant.PARAM_KEY_RCV_BUF) == b"histzero"
    hist = client.data_converter.encode_histograms_result.call_args_list[0].args[0]
    assert hist == pytest.approx([1.5, 2.5], abs=0.02)
    client.tenseal_context = None
    client._process_before_all_gather_v_horizontal(ctx)
    assert "not supported" in client._abort.call_args.args[0]


@pytest.mark.parametrize("version,expected", [("2.1.0", False), ("3.3.0", True), ("invalid", False)])
def test_client_secure_version_gate_and_context_cleanup(client, monkeypatch, version, expected):
    monkeypatch.setattr(client_module.xgboost, "__version__", version)
    assert client._check_xgboost_version(False) is expected
    assert client._check_xgboost_version(True)
    client.tenseal_context = object()
    client.handle_event(EventType.END_RUN, FLContext())
    assert client.tenseal_context is None
