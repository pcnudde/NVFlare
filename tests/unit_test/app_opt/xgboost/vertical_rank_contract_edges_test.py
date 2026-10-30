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

pytest.importorskip("xgboost")

from nvflare.app_opt.xgboost.recipes.vertical import XGBVerticalRecipe  # noqa: E402


def recipe(**kwargs):
    return XGBVerticalRecipe(name="vertical-contract", min_clients=2, num_rounds=1, label_owner="site-1", **kwargs)


@pytest.mark.parametrize(
    "kwargs,reason", [({"label_owner": "client"}, "format 'site-X'"), ({"num_rounds": 0}, "at least 1")]
)
def test_vertical_parameters_reject_invalid_owner_and_round_budget(kwargs, reason):
    values = dict(name="invalid", min_clients=2, num_rounds=1, label_owner="site-1")
    values.update(kwargs)
    with pytest.raises(ValueError, match=reason):
        XGBVerticalRecipe(**values)


@pytest.mark.parametrize(
    "ranks,reason",
    [
        ({"site-1": 0}, "exactly"),
        ({"site-1": 0, "site-2": True}, "integers"),
        ({"site-1": 0, "site-2": 0}, "unique and consecutive"),
        ({"site-1": 1, "site-2": 0}, "label_owner must be assigned rank 0"),
    ],
)
def test_vertical_client_rank_contract_checks_identity_type_uniqueness_and_label_owner(ranks, reason):
    item = recipe(client_ranks=ranks)
    with pytest.raises(ValueError, match=reason):
        item._resolve_client_ranks({"site-1": {}, "site-2": {}})
    assert not item._job.clients and item.per_site_config is None


def test_missing_label_owner_rejected_and_secure_controller_forces_in_process_training():
    item = recipe(secure=True, in_process=False)
    with pytest.raises(ValueError, match="must be included"):
        item._resolve_client_ranks({"site-2": {}, "site-3": {}})
    controller = item._create_controller({"site-1": 0, "site-2": 1})
    assert controller.in_process and controller.secure_training
    assert controller.data_split_mode == 1
