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

import pytest

from nvflare.fuel.utils import memory_utils as subject


@pytest.mark.parametrize(
    "library, expected",
    [
        (SimpleNamespace(), "unknown"),
        (SimpleNamespace(malloc_trim=Mock()), "glibc"),
        (SimpleNamespace(mallctl=Mock(), malloc_trim=Mock()), "jemalloc"),
    ],
)
def test_allocator_detection_prioritizes_jemalloc_without_loading_native_library(monkeypatch, library, expected):
    load = Mock(return_value=library)
    monkeypatch.setattr(subject, "ctypes", SimpleNamespace(CDLL=load))
    assert subject.get_allocator_type.__wrapped__() == expected
    load.assert_called_once_with(None)
    load.side_effect = OSError("not available")
    assert subject.get_allocator_type.__wrapped__() == "unknown"


def test_glibc_probe_configures_trim_signature_and_tolerates_missing_symbol(monkeypatch):
    trim = Mock()
    library = SimpleNamespace(malloc_trim=trim)
    load = Mock(return_value=library)
    monkeypatch.setattr(subject, "CDLL", load)
    assert subject._get_glibc.__wrapped__() is library
    assert trim.argtypes == [subject.c_size_t] and trim.restype is int
    load.return_value = SimpleNamespace()
    assert subject._get_glibc.__wrapped__() is None
    load.side_effect = OSError("missing libc")
    assert subject._get_glibc.__wrapped__() is None


def test_trim_failure_is_nonfatal_and_allocator_cleanup_never_forces_jemalloc(monkeypatch):
    library = SimpleNamespace(malloc_trim=Mock(side_effect=OSError("allocator rejected")))
    monkeypatch.setattr(subject, "_get_glibc", Mock(return_value=library))
    assert subject.try_malloc_trim() is None
    library.malloc_trim.assert_called_once_with(0)
    collect, trim = Mock(return_value=2), Mock()
    monkeypatch.setattr(subject, "gc", SimpleNamespace(collect=collect))
    monkeypatch.setattr(subject, "get_allocator_type", Mock(return_value="jemalloc"))
    monkeypatch.setattr(subject, "try_malloc_trim", trim)
    subject.cleanup_memory()
    collect.assert_called_once_with()
    trim.assert_not_called()
