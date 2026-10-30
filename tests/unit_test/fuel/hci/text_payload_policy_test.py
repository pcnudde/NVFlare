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

from unittest.mock import Mock

import pytest

from nvflare.fuel.hci import base64_utils as codec
from nvflare.fuel.hci import cmd_arg_utils as args
from nvflare.fuel.hci.client.event import EventContext, EventPropKey


def test_binary_and_utf8_files_roundtrip_and_count_encoded_bytes(tmp_path):
    source, decoded = tmp_path / "source", tmp_path / "decoded"
    source.write_bytes(b"\x00\xff\x80hello\n")
    encoded = codec.binary_file_to_b64str(source)
    assert encoded == "AP+AaGVsbG8K"
    assert codec.b64str_to_binary_file(encoded, decoded) == 9
    assert decoded.read_bytes() == source.read_bytes()
    source.write_text("hello \u03bb", encoding="utf-8")
    text = codec.text_file_to_b64str(source)
    assert codec.b64str_to_str(text) == "hello \u03bb"
    assert codec.b64str_to_text_file(text, decoded) == len("hello \u03bb".encode())
    assert decoded.read_text(encoding="utf-8") == "hello \u03bb"


@pytest.mark.parametrize(
    "validate, value, message",
    [
        (args.process_targets_into_str, "site", "not a list"),
        (args.process_targets_into_str, [1], "must be strings"),
        (args.process_targets_into_str, ["bad site"], "invalid target"),
        (args.validate_required_target_string, "", "not specified"),
        (args.validate_required_target_string, 1, "not str"),
        (args.validate_options_string, 1, "not str"),
        (args.validate_options_string, ";shell", "valid characters"),
        (args.validate_path_string, 1, "not str"),
        (args.validate_path_string, "unsafe?", "unsupported characters"),
        (args.validate_path_string, "/absolute", "absolute path"),
        (args.validate_path_string, "nested/../secret", "not allowed"),
        (args.validate_file_string, "archive.zip", "permitted"),
    ],
)
def test_shell_argument_validators_reject_unsafe_or_wrong_type_values(validate, value, message):
    with pytest.raises(SyntaxError, match=message):
        validate(value)


def test_argument_parser_reports_failures_without_exiting_and_usage_remains_available(monkeypatch):
    parser = args.ArgValidator("inspect")
    parser.add_argument("--count", type=int)
    error, result = parser.validate(["--unknown"])
    assert "unrecognized arguments" in error and result.count is None
    assert "--count COUNT" in parser.get_usage()
    monkeypatch.setattr(parser, "parse_args", Mock(side_effect=RuntimeError("unexpected")))
    error, result = parser.validate([])
    assert "argument error" in error and result is None


def test_event_custom_properties_preserve_existing_values_and_honor_missing_defaults():
    context = EventContext()
    assert context.get_custom_prop("missing", "fallback") == "fallback"
    context.set_custom_prop("first", 1)
    context.set_custom_prop("second", 2)
    assert context.get_prop(EventPropKey.CUSTOM_PROPS) == {"first": 1, "second": 2}
    assert context.get_custom_prop("first", None) == 1
    assert context.get_custom_prop("missing", 3) == 3
