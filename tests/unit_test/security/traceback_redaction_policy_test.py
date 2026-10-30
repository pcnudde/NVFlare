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

from nvflare.security import logging as subject


def recursive_error(depth):
    if depth:
        return recursive_error(depth - 1)
    raise ValueError("private credential text")


@pytest.mark.parametrize("secure", [True, False])
def test_traceback_redacts_exception_text_but_keeps_type_and_recursion_structure(monkeypatch, secure):
    monkeypatch.setenv(subject.SECURE_LOGGING_VAR_NAME, "true" if secure else "false")
    logger = Mock()
    try:
        recursive_error(4)
    except ValueError as error:
        text = subject.secure_format_traceback()
        formatted_error = subject.secure_format_exception(error)
        subject.secure_log_traceback(logger)
    assert "ValueError" in text and "recursive_error" in text
    assert ("private credential text" in text) is not secure
    assert ("private credential text" in formatted_error) is not secure
    assert "Previous line repeated" in text
    logger.error.assert_called_once_with(text)


def test_default_traceback_logger_uses_root_without_changing_handlers(monkeypatch):
    monkeypatch.setenv(subject.SECURE_LOGGING_VAR_NAME, "1")
    logger = Mock()
    monkeypatch.setattr(subject, "logging", SimpleNamespace(getLogger=Mock(return_value=logger)))
    try:
        raise RuntimeError("secret")
    except RuntimeError:
        subject.secure_log_traceback()
    logger.error.assert_called_once()
    assert "secret" not in logger.error.call_args.args[0]
