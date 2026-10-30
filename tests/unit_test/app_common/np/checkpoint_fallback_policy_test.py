# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

from unittest.mock import Mock

import numpy as np
import pytest

from nvflare.apis.fl_constant import FLContextKey
from nvflare.apis.fl_context import FLContext
from nvflare.app_common.np.utils import load_numpy_model


@pytest.mark.parametrize("source", ["relative", "absolute", "missing", "corrupt", "saved", "fallback"])
def test_checkpoint_priority_resolves_custom_path_and_never_silently_ignores_explicit_source(tmp_path, source):
    custom = tmp_path / "custom"
    custom.mkdir()
    checkpoint, saved = custom / "checkpoint.npy", tmp_path / "saved.npy"
    np.save(checkpoint, np.array([1, 2]))
    np.save(saved, np.array([3, 4]))
    ctx = FLContext()
    ctx.set_prop(FLContextKey.APP_ROOT, str(tmp_path))
    fallback = Mock(return_value=np.array([5, 6]))
    logger = Mock()
    explicit = (
        str(checkpoint)
        if source in ("absolute", "corrupt")
        else checkpoint.name if source in ("relative", "missing") else None
    )
    if source == "missing":
        checkpoint.unlink()
    if source == "corrupt":
        checkpoint.write_bytes(b"invalid checkpoint")
    if source == "fallback":
        saved.write_bytes(b"invalid saved model")
    if source in ("missing", "corrupt"):
        with pytest.raises(ValueError, match="not found|Failed to load"):
            load_numpy_model(ctx, logger, explicit, str(saved), fallback)
        fallback.assert_not_called()
    else:
        actual = load_numpy_model(ctx, logger, explicit, str(saved), fallback)
        np.testing.assert_array_equal(
            actual, [1, 2] if source in ("absolute", "relative") else [3, 4] if source == "saved" else [5, 6]
        )
        assert fallback.call_count == int(source == "fallback")
