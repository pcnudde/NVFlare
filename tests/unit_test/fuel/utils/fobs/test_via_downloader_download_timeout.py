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

"""Unit tests for ViaDownloaderDecomposer streamed download timeouts.

DownloadService transaction timeout is an inactivity timeout. It must honor the
generic streaming_idle_timeout default, per-decomposer min_download_timeout
overrides, and a floor at least as large as streaming_per_request_timeout.

CONTRACT:
- When no job config value is set, use the default effective idle timeout
- When job config sets streaming_idle_timeout=500, min_timeout=500
- When job config sets np_min_download_timeout=600, min_timeout=600
- msg_root_ttl > min_timeout means timeout = msg_root_ttl
- msg_root_ttl < min_timeout means timeout is floored to min_timeout
- msg_root_ttl absent means timeout = min_timeout
- ObjectDownloader receives the computed timeout value
"""

from unittest.mock import MagicMock, patch

from nvflare.apis.fl_constant import ConfigVarName
from nvflare.fuel.utils import fobs
from nvflare.fuel.utils.fobs.decomposers.via_downloader import ViaDownloaderDecomposer, _CtxKey

# ---------------------------------------------------------------------------
# Minimal concrete subclass; only _create_downloader is under test
# ---------------------------------------------------------------------------


class _FakeDecomposer(ViaDownloaderDecomposer):
    """Concrete stub of ViaDownloaderDecomposer for testing _create_downloader."""

    def __init__(self, config_var_prefix="np_"):
        super().__init__(max_chunk_size=1024 * 1024, config_var_prefix=config_var_prefix)

    def to_downloadable(self, items, max_chunk_size, fobs_ctx):
        return MagicMock()

    def download(self, from_fqcn, ref_id, per_request_timeout, cell, secure=False, optional=False, abort_signal=None):
        return None, {}

    def get_download_dot(self):
        return 99

    def native_decompose(self, target, manager=None):
        return b""

    def native_recompose(self, data, manager=None):
        return data

    def supported_type(self):
        return object


def _make_fobs_ctx(msg_root_id=None, msg_root_ttl=None, cell=None):
    ctx = {}
    if msg_root_id is not None:
        ctx[_CtxKey.MSG_ROOT_ID] = msg_root_id
    if msg_root_ttl is not None:
        ctx[_CtxKey.MSG_ROOT_TTL] = msg_root_ttl
    ctx[fobs.FOBSContextKey.CELL] = cell or MagicMock()
    return ctx


# ---------------------------------------------------------------------------
# min_timeout comes from job config via acu.get_positive_float_var
# ---------------------------------------------------------------------------


class TestCreateDownloaderTimeouts:

    def test_default_min_timeout_honors_per_request_floor(self, monkeypatch):
        """When no timeout config is set, the idle timeout is at least the per-request timeout."""
        decomposer = _FakeDecomposer()
        captured = []

        def fake_object_downloader(**kwargs):
            captured.append(kwargs["timeout"])
            od = MagicMock()
            od.add_object = MagicMock()
            return od

        # acu returns default (no job config value set)
        monkeypatch.setattr(
            "nvflare.fuel.utils.fobs.decomposers.via_downloader.acu.get_positive_float_var",
            lambda name, default: default,
        )
        with patch(
            "nvflare.fuel.utils.fobs.decomposers.via_downloader.ObjectDownloader",
            side_effect=fake_object_downloader,
        ):
            ctx = _make_fobs_ctx()
            decomposer._create_downloader(ctx)

        assert len(captured) == 1
        assert captured[0] == 600.0

    def test_job_config_min_timeout_overrides_default(self, monkeypatch):
        """When job config sets np_min_download_timeout above per-request timeout, ObjectDownloader uses it."""
        decomposer = _FakeDecomposer()
        captured = []

        def fake_object_downloader(**kwargs):
            captured.append(kwargs["timeout"])
            od = MagicMock()
            od.add_object = MagicMock()
            return od

        # Simulate job config returning 600 for the np_min_download_timeout key
        def fake_get_positive_float_var(name, default):
            if name == "np_min_download_timeout":
                return 700.0
            return default

        monkeypatch.setattr(
            "nvflare.fuel.utils.fobs.decomposers.via_downloader.acu.get_positive_float_var",
            fake_get_positive_float_var,
        )
        with patch(
            "nvflare.fuel.utils.fobs.decomposers.via_downloader.ObjectDownloader",
            side_effect=fake_object_downloader,
        ):
            ctx = _make_fobs_ctx()
            decomposer._create_downloader(ctx)

        assert len(captured) == 1
        assert captured[0] == 700.0

    def test_streaming_idle_timeout_is_generic_default(self, monkeypatch):
        """When only streaming_idle_timeout is set, ObjectDownloader uses it as the idle timeout."""
        decomposer = _FakeDecomposer()
        captured = []

        def fake_object_downloader(**kwargs):
            captured.append(kwargs["timeout"])
            od = MagicMock()
            od.add_object = MagicMock()
            return od

        def fake_get_positive_float_var(name, default):
            if name == ConfigVarName.STREAMING_IDLE_TIMEOUT:
                return 700.0
            return default

        monkeypatch.setattr(
            "nvflare.fuel.utils.fobs.decomposers.via_downloader.acu.get_positive_float_var",
            fake_get_positive_float_var,
        )
        with patch(
            "nvflare.fuel.utils.fobs.decomposers.via_downloader.ObjectDownloader",
            side_effect=fake_object_downloader,
        ):
            ctx = _make_fobs_ctx()
            decomposer._create_downloader(ctx)

        assert captured[0] == 700.0

    def test_explicit_min_timeout_below_per_request_is_floored(self, monkeypatch):
        """A too-low min_download_timeout is floored to the per-request timeout."""
        decomposer = _FakeDecomposer()
        captured = []

        def fake_object_downloader(**kwargs):
            captured.append(kwargs["timeout"])
            od = MagicMock()
            od.add_object = MagicMock()
            return od

        def fake_get_positive_float_var(name, default):
            if name == "np_min_download_timeout":
                return 120.0
            if name == "np_streaming_per_request_timeout":
                return 600.0
            return default

        monkeypatch.setattr(
            "nvflare.fuel.utils.fobs.decomposers.via_downloader.acu.get_positive_float_var",
            fake_get_positive_float_var,
        )
        with patch(
            "nvflare.fuel.utils.fobs.decomposers.via_downloader.ObjectDownloader",
            side_effect=fake_object_downloader,
        ):
            ctx = _make_fobs_ctx()
            decomposer._create_downloader(ctx)

        assert captured[0] == 600.0

    def test_msg_root_ttl_above_min_not_clamped(self, monkeypatch):
        """msg_root_ttl larger than min_timeout is passed through as-is."""
        decomposer = _FakeDecomposer()
        captured = []

        def fake_object_downloader(**kwargs):
            captured.append(kwargs["timeout"])
            od = MagicMock()
            od.add_object = MagicMock()
            return od

        monkeypatch.setattr(
            "nvflare.fuel.utils.fobs.decomposers.via_downloader.acu.get_positive_float_var",
            lambda name, default: default,
        )
        with patch(
            "nvflare.fuel.utils.fobs.decomposers.via_downloader.ObjectDownloader",
            side_effect=fake_object_downloader,
        ):
            ctx = _make_fobs_ctx(msg_root_ttl=700.0)
            decomposer._create_downloader(ctx)

        assert captured[0] == 700.0  # not raised above msg_root_ttl

    def test_msg_root_ttl_below_min_is_floored(self, monkeypatch):
        """msg_root_ttl smaller than min_timeout is floored to min_timeout."""
        decomposer = _FakeDecomposer()
        captured = []

        def fake_object_downloader(**kwargs):
            captured.append(kwargs["timeout"])
            od = MagicMock()
            od.add_object = MagicMock()
            return od

        # Simulate job config setting min_timeout to 600
        monkeypatch.setattr(
            "nvflare.fuel.utils.fobs.decomposers.via_downloader.acu.get_positive_float_var",
            lambda name, default: 600.0 if "min_download_timeout" in name else default,
        )
        with patch(
            "nvflare.fuel.utils.fobs.decomposers.via_downloader.ObjectDownloader",
            side_effect=fake_object_downloader,
        ):
            ctx = _make_fobs_ctx(msg_root_ttl=30.0)  # much lower than min
            decomposer._create_downloader(ctx)

        assert captured[0] == 600.0  # floored to min_timeout

    def test_no_msg_root_ttl_uses_min_timeout(self, monkeypatch):
        """When msg_root_ttl is absent, ObjectDownloader gets min_timeout."""
        decomposer = _FakeDecomposer()
        captured = []

        def fake_object_downloader(**kwargs):
            captured.append(kwargs["timeout"])
            od = MagicMock()
            od.add_object = MagicMock()
            return od

        monkeypatch.setattr(
            "nvflare.fuel.utils.fobs.decomposers.via_downloader.acu.get_positive_float_var",
            lambda name, default: (
                120.0
                if "min_download_timeout" in name
                else 100.0 if "streaming_per_request_timeout" in name else default
            ),
        )
        with patch(
            "nvflare.fuel.utils.fobs.decomposers.via_downloader.ObjectDownloader",
            side_effect=fake_object_downloader,
        ):
            ctx = _make_fobs_ctx()  # no msg_root_ttl
            decomposer._create_downloader(ctx)

        assert captured[0] == 120.0

    def test_config_var_name_uses_prefix(self, monkeypatch):
        """acu.get_positive_float_var must be called with the prefixed key."""
        decomposer = _FakeDecomposer(config_var_prefix="np_")
        names_queried = []

        def tracking_get(name, default):
            names_queried.append(name)
            return default

        monkeypatch.setattr(
            "nvflare.fuel.utils.fobs.decomposers.via_downloader.acu.get_positive_float_var",
            tracking_get,
        )
        with patch("nvflare.fuel.utils.fobs.decomposers.via_downloader.ObjectDownloader", return_value=MagicMock()):
            ctx = _make_fobs_ctx()
            decomposer._create_downloader(ctx)

        assert f"np_{ConfigVarName.MIN_DOWNLOAD_TIMEOUT}" in names_queried
