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

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from nvflare.app_opt.hf import api as module  # noqa: E402


def test_checkpoint_resume_allowlist_contains_numpy_reconstruction_and_dtypes_without_duplicates(monkeypatch):
    register = Mock()
    monkeypatch.setattr(torch.serialization, "add_safe_globals", register)
    module._allow_torch_checkpoint_resume_globals()
    register.assert_called_once()
    allowed = register.call_args.args[0]
    assert np.ndarray in allowed and np.dtype in allowed
    for name in ("uint32", "int64", "float32", "float64"):
        assert type(np.dtype(name)) in allowed
    assert any(getattr(value, "__name__", None) == "_reconstruct" for value in allowed)
    assert len({id(value) for value in allowed}) == len(allowed)


def test_checkpoint_allowlisting_supports_torch_without_safe_globals_api(monkeypatch):
    monkeypatch.setattr(torch.serialization, "add_safe_globals", None)
    assert module._allow_torch_checkpoint_resume_globals() is None


def test_single_process_marker_cannot_override_initialized_nonzero_distributed_rank(monkeypatch):
    monkeypatch.setattr(module, "environment_declares_single_client_api_process", lambda: True)
    monkeypatch.setattr(module, "_torch_dist", lambda: SimpleNamespace(get_rank=lambda: 2))
    with pytest.raises(RuntimeError, match="Torch rank 2"):
        module._resolve_rank(SimpleNamespace())


@pytest.mark.parametrize("version", ["", "unparseable-version"])
def test_unknown_transformers_version_uses_conservative_checkpoint_restore_strategy(monkeypatch, version):
    monkeypatch.setattr(module, "_transformers_version", lambda: version)
    assert not module._transformers_version_is_verified()
