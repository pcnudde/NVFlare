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

import numpy as np
import pytest
import tenseal as ts

from nvflare.apis.dxo import DXO, DataKind, MetaKey, from_shareable
from nvflare.apis.event_type import EventType
from nvflare.apis.fl_constant import ReservedKey
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.app_common.abstract.model import ModelLearnableKey, make_model_learnable
from nvflare.app_common.app_constant import AppConstants
from nvflare.app_opt.he import model_decryptor as dec
from nvflare.app_opt.he import model_encryptor as enc
from nvflare.app_opt.he import model_shareable_generator as gen
from nvflare.app_opt.he.constant import HE_ALGORITHM_CKKS
from nvflare.app_opt.he.cross_site_model_eval import HECrossSiteModelEval


@pytest.fixture
def context():
    ctx = FLContext()
    ctx.set_prop(ReservedKey.ENGINE, MagicMock())
    peer = FLContext()
    peer.set_prop(ReservedKey.IDENTITY_NAME, "site")
    ctx.set_peer_context(peer)
    return ctx


@pytest.fixture
def crypto():
    context = ts.context(ts.SCHEME_TYPE.CKKS, 4096, coeff_mod_bit_sizes=[30, 20, 30], n_threads=1)
    context.global_scale = 2**20
    return context


def test_partial_encryption_weights_by_client_and_local_iterations_then_decrypts_metadata(monkeypatch, context, crypto):
    encryptor = enc.HEModelEncryptor(encrypt_layers="weight", aggregation_weights={"site": 1.5})
    decryptor = dec.HEModelDecryptor()
    monkeypatch.setattr(enc, "load_tenseal_context_from_workspace", lambda *a: crypto)
    monkeypatch.setattr(dec, "load_tenseal_context_from_workspace", lambda *a: crypto)
    encryptor.handle_event(EventType.START_RUN, context)
    weights = {"weight": np.array([[1.0, 2.0], [3.0, 4.0]]), "bias": np.array([5.0, 6.0])}
    dxo = DXO(DataKind.WEIGHT_DIFF, weights, {MetaKey.NUM_STEPS_CURRENT_ROUND: 2})
    encrypted = encryptor.process_dxo(dxo, dxo.to_shareable(), context)
    assert encrypted.get_meta_prop(MetaKey.PROCESSED_KEYS) == {"weight": True, "bias": False}
    assert encrypted.get_meta_prop(MetaKey.PROCESSED_ALGORITHM) == HE_ALGORITHM_CKKS
    assert encrypted.data["weight"].decrypt() == pytest.approx([3, 6, 9, 12], abs=0.03)
    np.testing.assert_array_equal(encrypted.data["bias"], [15, 18])
    shareable = encrypted.to_shareable()
    decrypted = decryptor.process_dxo(encrypted, shareable, context)
    assert decrypted.data["weight"] == pytest.approx([3, 6, 9, 12], abs=0.03)
    assert decrypted.get_meta_prop(MetaKey.PROCESSED_KEYS) is None
    assert decrypted.get_meta_prop(MetaKey.PROCESSED_ALGORITHM) is None
    assert from_shareable(shareable).data["weight"] == pytest.approx([3, 6, 9, 12], abs=0.03)
    encryptor.handle_event(EventType.END_RUN, context)
    decryptor.handle_event(EventType.END_RUN, context)
    assert encryptor.tenseal_context is decryptor.tenseal_context is None


@pytest.mark.parametrize("layers", [3, ["weight", 3]])
def test_encryption_rejects_invalid_layer_selection_types(layers):
    with pytest.raises(ValueError):
        enc.HEModelEncryptor(encrypt_layers=layers)


def test_encryption_requires_matching_layers_and_iteration_metadata(context, crypto, monkeypatch):
    encryptor = enc.HEModelEncryptor(encrypt_layers="missing")
    encryptor.tenseal_context = crypto
    encryptor.n_iter = 1
    with pytest.raises(ValueError, match="No matching layers"):
        encryptor.encryption({"weight": np.array([1.0])}, context)
    encryptor = enc.HEModelEncryptor(encrypt_layers=["missing"], weigh_by_local_iter=False)
    encryptor.tenseal_context = crypto
    with pytest.raises(ValueError, match="Nothing has been encrypted"):
        encryptor.encryption({"weight": np.array([1.0])}, context)
    encryptor = enc.HEModelEncryptor()
    monkeypatch.setattr(enc, "load_tenseal_context_from_workspace", lambda *a: crypto)
    encryptor.handle_event(EventType.START_RUN, context)
    with pytest.raises(ValueError, match="local iterations"):
        encryptor.process_dxo(DXO(DataKind.WEIGHTS, {"weight": np.array([1.0])}), Shareable(), context)
    decryptor = dec.HEModelDecryptor()
    decryptor.tenseal_context = crypto
    assert decryptor.process_dxo(DXO(DataKind.WEIGHTS, {}), Shareable(), context) is None
    wrong = DXO(DataKind.WEIGHTS, {}, {MetaKey.PROCESSED_KEYS: {"w": True}, MetaKey.PROCESSED_ALGORITHM: "other"})
    assert decryptor.process_dxo(wrong, Shareable(), context) is None
    with pytest.raises(ValueError, match="encrypted_layers"):
        decryptor.decryption({}, None, context)


def test_encrypted_model_diff_addition_serialization_and_full_weight_replacement(context, crypto):
    generator = gen.HEModelShareableGenerator()
    generator.tenseal_context = crypto
    base = make_model_learnable({"weight": np.array([1.0, 2.0])}, {})
    context.set_prop(AppConstants.GLOBAL_MODEL, base)
    diff = DXO(
        DataKind.WEIGHT_DIFF,
        {"weight": ts.ckks_vector(crypto, [0.5, 1.5])},
        {MetaKey.PROCESSED_ALGORITHM: HE_ALGORITHM_CKKS},
    )
    result = generator.shareable_to_learnable(diff.to_shareable(), context)
    assert result[ModelLearnableKey.WEIGHTS]["weight"].decrypt() == pytest.approx([1.5, 3.5], abs=0.02)
    shareable = generator.learnable_to_shareable(result, context)
    assert isinstance(from_shareable(shareable).data["weight"], bytes)
    replacement = DXO(DataKind.WEIGHTS, {"weight": np.array([8.0])}, {MetaKey.PROCESSED_ALGORITHM: HE_ALGORITHM_CKKS})
    result = generator.shareable_to_learnable(replacement.to_shareable(), context)
    np.testing.assert_array_equal(result[ModelLearnableKey.WEIGHTS]["weight"], [8])
    with pytest.raises(ValueError, match="not supported"):
        gen.add_to_global_weights(np.array([1]), {"weight": "bad"}, "weight")


def test_generator_rejects_wrong_algorithm_empty_diffs_and_missing_global_model(context, crypto, monkeypatch):
    generator = gen.HEModelShareableGenerator()
    monkeypatch.setattr(gen, "load_tenseal_context_from_workspace", lambda *a: crypto)
    generator.handle_event(EventType.START_RUN, context)
    with pytest.raises(ValueError, match="expected encryption algorithm"):
        generator.shareable_to_learnable(DXO(DataKind.WEIGHTS, {"w": 1}).to_shareable(), context)
    context.set_prop(AppConstants.GLOBAL_MODEL, make_model_learnable({"w": np.array([1])}, {}))
    with pytest.raises(ValueError, match="data is empty"):
        generator.shareable_to_learnable(
            DXO(DataKind.WEIGHT_DIFF, {}, {MetaKey.PROCESSED_ALGORITHM: HE_ALGORITHM_CKKS}).to_shareable(), context
        )
    with pytest.raises(ValueError, match="not supported"):
        generator.shareable_to_learnable(
            DXO(DataKind.METRICS, {"a": 1}, {MetaKey.PROCESSED_ALGORITHM: HE_ALGORITHM_CKKS}).to_shareable(), context
        )
    context.set_prop(AppConstants.GLOBAL_MODEL, None)
    generator.system_panic = MagicMock()
    assert (
        generator.shareable_to_learnable(
            DXO(DataKind.WEIGHTS, {"w": 1}, {MetaKey.PROCESSED_ALGORITHM: HE_ALGORITHM_CKKS}).to_shareable(), context
        )
        is None
    )
    generator.system_panic.assert_called_once()
    generator.handle_event(EventType.END_RUN, context)
    assert generator.tenseal_context is None


def test_he_cross_validation_content_round_trip_and_filesystem_errors(context, tmp_path, isolated_fobs):
    controller = HECrossSiteModelEval()
    data = DXO(DataKind.WEIGHTS, {"w": [1.0, 2.0]})
    filename = controller._save_validation_content("model", str(tmp_path), data, context)
    assert filename == str(tmp_path / "model")
    restored = controller._load_validation_content("model", str(tmp_path), context)
    np.testing.assert_array_equal(restored.data["w"], [1, 2])
    with pytest.raises(ValueError, match="Unable to save"):
        controller._save_validation_content("model", str(tmp_path / "missing"), data, context)
    with pytest.raises(ValueError, match="loading shareable"):
        controller._load_validation_content("missing", str(tmp_path), context)
