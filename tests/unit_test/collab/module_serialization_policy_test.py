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

import copy
import importlib
import pickle
import sys
from types import ModuleType

import pytest

from nvflare.collab.api import module_wrapper as subject
from nvflare.collab.api.decorators import get_param_names, supports_context


@pytest.fixture
def decorated_module(tmp_path, monkeypatch):
    package = tmp_path / "collab_policy_fixture"
    package.mkdir()
    (package / "__init__.py").write_text("")
    path = package / "program.py"
    path.write_text(
        "from nvflare.collab import collab\n"
        "events = []\n"
        "@collab.publish\ndef exchange(value, context=None):\n    return value + 2\n"
        "@collab.init\ndef initialize(context):\n    events.append(('init', context))\n"
        "@collab.final\ndef finalize(context):\n    events.append(('final', context))\n"
        "@collab.main\ndef main():\n    return 9\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    try:
        yield importlib.import_module("collab_policy_fixture.program"), path
    finally:
        for name in list(sys.modules):
            if name == "collab_policy_fixture" or name.startswith("collab_policy_fixture."):
                sys.modules.pop(name, None)


def test_wrapper_reconstructs_decorated_module_without_serializing_runtime_state(decorated_module):
    module, _ = decorated_module
    wrapper = subject.ModuleWrapper(module.__name__)
    assert wrapper.exchange(5) == 7
    assert wrapper.main() == 9
    assert get_param_names(wrapper.exchange) == ["value", "context"]
    assert supports_context(wrapper.exchange)
    assert supports_context(wrapper.initialize)
    marker = object()
    wrapper.initialize(marker)
    wrapper.finalize(marker)
    assert module.events == [("init", marker), ("final", marker)]
    wrapper.temporary = "must not transfer"
    for restored in (copy.deepcopy(wrapper), pickle.loads(pickle.dumps(wrapper))):
        assert restored.module_name == module.__name__
        assert restored.exchange(8) == 10
        assert not hasattr(restored, "temporary")
    blank = subject.ModuleWrapper()
    assert blank.module_name is None
    blank.__setstate__(wrapper.__getstate__())
    assert blank.main() == 9
    with pytest.raises(TypeError, match="ModuleType or str"):
        subject.ModuleWrapper(4)


def test_main_module_resolution_uses_real_importable_qualified_package(decorated_module, monkeypatch):
    module, path = decorated_module
    main = ModuleType("__main__")
    main.__file__ = str(path)
    # Both short and qualified candidates import; prefer the qualified package.
    monkeypatch.syspath_prepend(str(path.parent))
    try:
        assert subject.get_importable_module_name(main) == module.__name__
    finally:
        sys.modules.pop("program", None)
    main.__file__ = subject.__file__
    assert subject.get_importable_module_name(main) == "nvflare.collab.api.module_wrapper"


def test_main_module_resolution_reports_missing_file_or_unimportable_location(tmp_path):
    main = ModuleType("__main__")
    with pytest.raises(ValueError, match="Cannot determine importable module name for __main__"):
        subject.get_importable_module_name(main)
    main.__file__ = str(tmp_path / "not_importable.py")
    with pytest.raises(ValueError, match="Cannot determine importable module name from file"):
        subject.get_importable_module_name(main)
    assert subject.get_caller_module(stack_level=10000) is None
