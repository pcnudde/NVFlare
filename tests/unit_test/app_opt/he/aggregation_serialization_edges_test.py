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

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

ts = pytest.importorskip("tenseal")

from nvflare.apis.dxo import DXO, DataKind, MetaKey, from_shareable  # noqa: E402
from nvflare.apis.event_type import EventType  # noqa: E402
from nvflare.apis.fl_constant import FLContextKey, ReservedKey  # noqa: E402
from nvflare.apis.fl_context import FLContext  # noqa: E402
from nvflare.app_common.app_constant import AppConstants  # noqa: E402
from nvflare.app_opt.he import homomorphic_encrypt as crypto  # noqa: E402
from nvflare.app_opt.he import intime_accumulate_model_aggregator as aggregation  # noqa: E402
from nvflare.app_opt.he import model_serialize_filter as serialization  # noqa: E402
from nvflare.fuel.sec.security_content_service import LoadResult  # noqa: E402


@pytest.fixture
def context():
    result = ts.context(ts.SCHEME_TYPE.CKKS, 8192, coeff_mod_bit_sizes=[60, 40, 40, 60], n_threads=1)
    result.global_scale = 2**40
    return result


def test_he_weighted_aggregation_preserves_ciphertexts_and_marks_clear_layers(monkeypatch, context):
    monkeypatch.setattr(aggregation, "decomposers", SimpleNamespace(register=Mock()))
    aggregator = aggregation.HEInTimeAccumulateWeightedAggregator(
        expected_data_kind=DataKind.WEIGHTS, aggregation_weights={"a": 1, "b": 1}
    )
    ctx = FLContext()
    ctx.set_prop(ReservedKey.ENGINE, Mock())
    ctx.set_prop(AppConstants.CURRENT_ROUND, 0)
    aggregator.handle_event(EventType.START_RUN, ctx)
    for site, values, bias in [("a", [2.0, 4.0], 1.0), ("b", [4.0, 8.0], 3.0)]:
        shareable = DXO(
            DataKind.WEIGHTS,
            {"w": ts.ckks_vector(context, values), "bias": np.array([bias])},
            {MetaKey.NUM_STEPS_CURRENT_ROUND: 1},
        ).to_shareable()
        shareable.set_peer_props({ReservedKey.IDENTITY_NAME: site})
        shareable.add_cookie(AppConstants.CONTRIBUTION_ROUND, 0)
        assert aggregator.accept(shareable, ctx)
    result = from_shareable(aggregator.aggregate(ctx))
    assert result.data["w"].decrypt() == pytest.approx([3.0, 6.0], abs=0.04)
    np.testing.assert_array_equal(result.data["bias"], [2.0])
    assert result.get_meta_prop(MetaKey.PROCESSED_KEYS) == {"w": True, "bias": False}


def test_model_serialization_filter_roundtrips_nested_real_ciphertext_and_releases_context(monkeypatch, context):
    monkeypatch.setattr(serialization, "load_tenseal_context_from_workspace", lambda *args: context)
    item = serialization.HEModelSerializeFilter()
    ctx = FLContext()
    item.handle_event(EventType.START_RUN, ctx)
    learnable = {"weights": {"w": ts.ckks_vector(context, [2.0, 5.0]), "bias": np.array([3.0])}, "meta": {"round": 2}}
    saved = item.process_pre_save(learnable, ctx)
    assert isinstance(saved["weights"]["w"], bytes)
    assert saved["meta"]["round"] == 2
    loaded = item.process_post_load(saved, ctx)
    assert loaded["weights"]["w"].decrypt() == pytest.approx([2.0, 5.0], abs=0.03)
    saved_again = item.process_pre_save(loaded, ctx)
    restored = item.process_post_save(saved_again, ctx)
    assert item.process_post_get(restored, ctx)["weights"]["w"].decrypt() == pytest.approx([2.0, 5.0], abs=0.03)
    item.handle_event(EventType.END_RUN, ctx)
    assert item.tenseal_context is None


@pytest.mark.parametrize(
    "result",
    [LoadResult.INVALID_CONTENT, LoadResult.NO_SUCH_CONTENT, LoadResult.INVALID_SIGNATURE, LoadResult.NOT_SIGNED],
)
def test_secure_he_context_rejects_untrusted_content_before_native_deserialization(monkeypatch, result):
    service = SimpleNamespace(load_content=Mock(return_value=(b"not-a-context", result)))
    monkeypatch.setattr(crypto, "SecurityContentService", service)
    ctx = FLContext()
    ctx.set_prop(FLContextKey.SECURE_MODE, True)
    with pytest.raises(ValueError, match="Cannot load tenseal_context"):
        crypto.load_tenseal_context_from_workspace("keys.tenseal", ctx)
    service.load_content.assert_called_once_with("keys.tenseal")
    assert crypto.count_encrypted_layers({"w": True, "bias": False}) == (1, 2)
