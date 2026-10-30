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

import logging
from types import SimpleNamespace

from nvflare.fuel.utils import log_utils as subject


def test_hierarchy_inherits_closest_parent_and_marks_explicit_levels(monkeypatch, capsys):
    package = logging.Logger("sample", logging.WARNING)
    child = logging.Logger("sample.child", logging.NOTSET)
    explicit = logging.Logger("sample.critical", logging.ERROR)
    orphan = logging.PlaceHolder(logging.Logger("sample.placeholder.leaf"))
    root = logging.Logger("root", logging.INFO)
    root.manager = SimpleNamespace(
        loggerDict={
            "sample": package,
            "sample.child": child,
            "sample.critical": explicit,
            "sample.placeholder": orphan,
            "sample.placeholder.leaf": logging.Logger("sample.placeholder.leaf"),
            "outside": logging.Logger("outside"),
        }
    )
    monkeypatch.setattr(
        subject, "logging", SimpleNamespace(root=root, NOTSET=logging.NOTSET, getLevelName=logging.getLevelName)
    )
    subject.print_logger_hierarchy("sample")
    out = capsys.readouterr().out
    assert "hierarchical loggers (5)" in out
    assert "sample [WARNING (SET)]" in out
    assert "sample.child [WARNING]" in out
    assert "sample.critical [ERROR (SET)]" in out
    assert "sample.placeholder [WARNING]" in out
    assert "sample.placeholder.leaf [WARNING]" in out
    assert "outside" not in out
    assert out.index("sample.child") < out.index("sample.critical")
    package.setLevel(logging.NOTSET)
    subject.print_logger_hierarchy("sample")
    assert "sample.child [INFO]" in capsys.readouterr().out


def test_added_rotating_handler_inherits_primary_format_and_level(monkeypatch, tmp_path):
    logger = logging.Logger("isolated")
    main = logging.StreamHandler()
    formatter = logging.Formatter("%(levelname)s:%(message)s")
    main.setFormatter(formatter)
    main.setLevel(logging.ERROR)
    logger.addHandler(main)
    monkeypatch.setattr(subject, "logging", SimpleNamespace(getLogger=lambda: logger))
    try:
        subject.add_log_file_handler(str(tmp_path / "run.log"))
        handler = logger.handlers[1]
        assert handler.level == logging.ERROR
        assert handler.formatter is formatter
        assert handler.maxBytes == 20 * 1024 * 1024
        assert handler.backupCount == 10
        logger.error("retained")
        handler.flush()
        assert (tmp_path / "run.log").read_text().strip() == "ERROR:retained"
    finally:
        for handler in list(logger.handlers):
            logger.removeHandler(handler)
            handler.close()
