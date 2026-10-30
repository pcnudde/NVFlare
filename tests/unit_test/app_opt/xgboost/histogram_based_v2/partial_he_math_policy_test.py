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

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from nvflare.app_opt.xgboost.histogram_based_v2.sec.partial_he import adder, decrypter, encryptor, util


def _inline_pool(monkeypatch, module):
    calls = []

    def run(function, items, chunksize):
        items = list(items)
        calls.append((function, items, chunksize))
        return map(function, items)

    pool = SimpleNamespace(map=run)
    factory = MagicMock(return_value=pool)
    monkeypatch.setattr(module, "concurrent", SimpleNamespace(futures=SimpleNamespace(ProcessPoolExecutor=factory)))
    return factory, calls


@pytest.mark.parametrize("workers,expected_chunksize", [(2, 2), (10, 1)])
def test_encryptor_partitions_and_forwards_each_value_to_public_key(monkeypatch, workers, expected_chunksize):
    factory, calls = _inline_pool(monkeypatch, encryptor)
    public = SimpleNamespace(encrypt=MagicMock(side_effect=lambda value: f"cipher-{value}"))
    subject = encryptor.Encryptor(public, max_workers=workers)
    assert subject.encrypt([2, 3, 4, 5]) == ["cipher-2", "cipher-3", "cipher-4", "cipher-5"]
    assert [call.args for call in public.encrypt.call_args_list] == [(2,), (3,), (4,), (5,)]
    factory.assert_called_once_with(max_workers=workers)
    assert calls[0][0] is encryptor._do_enc
    assert calls[0][2] == expected_chunksize


def test_decrypter_preserves_plain_integers_and_delegates_only_ciphertext_groups(monkeypatch):
    factory, calls = _inline_pool(monkeypatch, decrypter)
    ciphertext1, ciphertext2 = object(), object()
    private = SimpleNamespace(decrypt=MagicMock(side_effect=[4, -3]))
    subject = decrypter.Decrypter(private, max_workers=2)
    result = subject.decrypt([[0, ciphertext1], [ciphertext2, 2], []])
    assert result == [[0, 4], [-3, 2], []]
    assert [call.args for call in private.decrypt.call_args_list] == [(ciphertext1,), (ciphertext2,)]
    factory.assert_called_once_with(max_workers=2)
    assert calls[0][0] is decrypter._do_decrypt
    assert calls[0][2] == 2


@pytest.mark.parametrize("groups", [None, [(7, [0, 3]), (9, [1, 2])]])
def test_partial_he_aggregator_adds_combined_gradient_hessian_bins_and_groups(monkeypatch, groups):
    factory, calls = _inline_pool(monkeypatch, adder)
    values = [util.combine(1, 3), util.combine(2, 5), util.combine(4, 7), util.combine(-1, -2)]
    subject = adder.Adder(max_workers=2)
    result = subject.add(values, [(5, [0, 1, 0, 1], 2)], sample_groups=groups, encode_sum=False)
    unpacked = [(fid, gid, [util.split(value) for value in histogram]) for fid, gid, histogram in result]
    assert unpacked == (
        [(5, 0, [(5, 10), (1, 3)])] if groups is None else [(5, 7, [(1, 3), (-1, -2)]), (5, 9, [(4, 7), (2, 5)])]
    )
    factory.assert_called_once_with(max_workers=2)
    assert calls[0][0] is adder._do_add
    assert calls[0][2] == 1


def test_aggregation_encoding_boundary_receives_computed_bins(monkeypatch):
    encode = MagicMock(return_value="encoded-histogram")
    monkeypatch.setattr(adder, "encode_encrypted_numbers_to_str", encode)
    result = adder._do_add((True, 4, [2, 3, 5], [0, 1, 0], 2, 7, None))
    encode.assert_called_once_with([7, 3])
    assert result[:2] == (4, 7)
    assert result[2] == "encoded-histogram"


@pytest.mark.parametrize("number", [1, 255, 256, 2**257 + 3])
def test_public_key_integer_base64_roundtrip_handles_padding_and_large_values(number):
    encoded = util.int_to_base64(number)
    assert "=" not in encoded
    assert util.base64_to_int(encoded) == number


@pytest.mark.parametrize("payload", ["a", "ab", "abc", "abcd", b"binary\x00bytes"])
def test_base64url_roundtrip_accepts_text_and_bytes(payload):
    encoded = util.base64url_encode(payload)
    assert util.base64url_decode(encoded) == (payload.encode() if isinstance(payload, str) else payload)
    with pytest.raises(ValueError, match="Invalid base64"):
        util.base64url_decode("x")


def test_decoder_restores_public_key_header_and_plain_integer_bins_without_crypto_runtime(monkeypatch):
    public = object()
    create = MagicMock(return_value=public)
    monkeypatch.setattr(util, "create_pub_key", create)
    encoded = json.dumps({"key": {"n": util.int_to_base64(12345)}, "nums": [0, 2, -3]})
    key, values = util.decode_encrypted_data(encoded, n_length=2048)
    assert key is public and values == [0, 2, -3]
    create.assert_called_once_with(key=12345, n_length=2048)
    aggregation = [(5, 7, json.dumps([0, 2, -3]))]
    encoded = util.encode_feature_aggregations(aggregation)
    assert util.decode_feature_aggregations(public, encoded) == [(5, 7, [0, 2, -3])]
