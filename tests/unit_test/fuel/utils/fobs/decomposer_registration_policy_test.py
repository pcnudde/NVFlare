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

from nvflare.fuel.utils.class_loader import get_class_name
from nvflare.fuel.utils.fobs import fobs as subject
from nvflare.fuel.utils.fobs.datum import DatumManager
from nvflare.fuel.utils.fobs.decomposer import Decomposer


class WireValue:
    pass


class AlternateValue:
    pass


class SharedAlias:
    pass


class ValueCodec(Decomposer):
    def __init__(self, kind=WireValue, aliases=(), dots=()):
        self.kind, self.aliases, self.dots = kind, aliases, dots

    def supported_type(self):
        return self.kind

    def supported_aliases(self):
        return self.aliases

    def supported_dots(self):
        return self.dots

    def decompose(self, target, manager=None):
        return vars(target).copy()

    def recompose(self, data, manager=None):
        result = self.kind()
        vars(result).update(data)
        return result


@pytest.fixture(autouse=True)
def registry(monkeypatch):
    monkeypatch.setattr(subject, "_decomposers", {})
    monkeypatch.setattr(subject, "_dot_handlers", {})
    monkeypatch.setattr(subject, "_type_name_whitelist", set())
    monkeypatch.setattr(subject, "_decomposers_registered", True)


def test_conflicting_aliases_and_dots_preserve_first_trusted_registration(caplog):
    first = ValueCodec(aliases=[SharedAlias], dots=[10])
    second = ValueCodec(AlternateValue, [SharedAlias], ["11", 0, -1, 10, 12])
    subject.register(first)
    subject.register(second)
    subject.register(ValueCodec(dots=[99]))
    assert subject._decomposers[get_class_name(SharedAlias)] is first
    assert subject._decomposers[get_class_name(AlternateValue)] is second
    assert subject.get_dot_handler(10) is first
    assert subject.get_dot_handler(12) is second
    assert subject.get_dot_handler(99) is None
    assert set(subject._dot_handlers) == {10, 12}
    assert "Duplicate registration for type" in caplog.text
    assert "Duplicate registration for DOT 10" in caplog.text
    assert caplog.text.count("Bad DOT") == 3


def test_non_decomposer_cannot_gain_registration_by_reporting_a_type(caplog):
    subject.register(SimpleNamespace(supported_type=lambda: WireValue))
    assert subject._decomposers == {}
    assert "is not a decomposer" in caplog.text


@pytest.mark.parametrize("decomposer, message", [("untrusted.Codec", "must be registered"), (None, "is not allowed")])
def test_untrusted_envelope_is_rejected_before_class_loader(monkeypatch, decomposer, message):
    loader = Mock()
    monkeypatch.setattr(subject, "load_class", loader)
    envelope = {subject.FOBS_TYPE: "untrusted.Value", subject.FOBS_DATA: {}}
    if decomposer:
        envelope[subject.FOBS_DECOMPOSER] = decomposer
    with pytest.raises(ValueError, match=message):
        subject.Packer(None).unpack(envelope)
    loader.assert_not_called()


@pytest.mark.parametrize("failure", [ValueError("payload too large"), TypeError("unsupported value")])
def test_serialization_failure_records_error_and_always_finalizes_manager(monkeypatch, failure):
    pack = Mock(side_effect=failure)
    monkeypatch.setattr(subject, "msgpack", SimpleNamespace(packb=pack))
    manager = DatumManager(fobs_ctx={})
    finalize = Mock(wraps=manager.post_process)
    monkeypatch.setattr(manager, "post_process", finalize)
    payload = "x" * (subject.MAX_CONTENT_LEN + 30)
    with pytest.raises(RuntimeError) as caught:
        subject.serialize(payload, manager=manager, use_bin_type=True)
    finalize.assert_called_once_with()
    assert manager.get_error() == str(caught.value)
    assert pack.call_args.args == (payload,)
    assert pack.call_args.kwargs["strict_types"] is True
    assert pack.call_args.kwargs["use_bin_type"] is True
    assert callable(pack.call_args.kwargs["default"])
    if isinstance(failure, ValueError):
        assert "not serializable" in str(caught.value)
        assert str(caught.value).endswith("x" * subject.MAX_CONTENT_LEN + " ...")
    else:
        assert "Exception serializing" in str(caught.value)
