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

"""Tests for PTClientAPILauncherExecutor timeout/resend parameters."""

import inspect


class TestPTClientAPILauncherExecutorParams:
    """PTClientAPILauncherExecutor must expose and forward timeout/resend params."""

    def test_submit_result_timeout_param_exists(self):
        """submit_result_timeout must be in PTClientAPILauncherExecutor signature."""
        from nvflare.app_opt.pt.client_api_launcher_executor import PTClientAPILauncherExecutor

        sig = inspect.signature(PTClientAPILauncherExecutor.__init__)
        assert "submit_result_timeout" in sig.parameters

    def test_max_resends_param_exists(self):
        """max_resends must be in PTClientAPILauncherExecutor signature."""
        from nvflare.app_opt.pt.client_api_launcher_executor import PTClientAPILauncherExecutor

        sig = inspect.signature(PTClientAPILauncherExecutor.__init__)
        assert "max_resends" in sig.parameters

    def test_download_complete_timeout_param_exists(self):
        """download_complete_timeout must be in PTClientAPILauncherExecutor signature."""
        from nvflare.app_opt.pt.client_api_launcher_executor import PTClientAPILauncherExecutor

        sig = inspect.signature(PTClientAPILauncherExecutor.__init__)
        assert "download_complete_timeout" in sig.parameters

    def test_streaming_idle_timeout_param_exists(self):
        """streaming_idle_timeout must be in PTClientAPILauncherExecutor signature."""
        from nvflare.app_opt.pt.client_api_launcher_executor import PTClientAPILauncherExecutor

        sig = inspect.signature(PTClientAPILauncherExecutor.__init__)
        assert "streaming_idle_timeout" in sig.parameters

    def test_defaults_match_base_class(self):
        """Default values for new params must match ClientAPILauncherExecutor defaults."""
        from nvflare.app_common.executors.client_api_launcher_executor import ClientAPILauncherExecutor
        from nvflare.app_opt.pt.client_api_launcher_executor import PTClientAPILauncherExecutor

        base_sig = inspect.signature(ClientAPILauncherExecutor.__init__)
        pt_sig = inspect.signature(PTClientAPILauncherExecutor.__init__)

        for param_name in (
            "submit_result_timeout",
            "max_resends",
            "download_complete_timeout",
            "streaming_idle_timeout",
        ):
            base_default = base_sig.parameters[param_name].default
            pt_default = pt_sig.parameters[param_name].default
            assert (
                pt_default == base_default
            ), f"{param_name}: PT default {pt_default!r} must match base default {base_default!r}"

    def test_new_params_forwarded_to_base(self):
        """Timeout/resend params are forwarded to base __init__.

        Verifies by patching ClientAPILauncherExecutor.__init__ and checking
        the kwargs it receives.
        """
        from unittest.mock import patch

        from nvflare.app_opt.pt.client_api_launcher_executor import PTClientAPILauncherExecutor

        received_kwargs = {}

        def mock_base_init(self_inner, **kwargs):
            received_kwargs.update(kwargs)

        with patch(
            "nvflare.app_opt.pt.client_api_launcher_executor.ClientAPILauncherExecutor.__init__",
            side_effect=mock_base_init,
        ):
            obj = PTClientAPILauncherExecutor.__new__(PTClientAPILauncherExecutor)
            PTClientAPILauncherExecutor.__init__(
                obj,
                pipe_id="test_pipe",
                submit_result_timeout=999.0,
                max_resends=7,
                download_complete_timeout=3600.0,
                streaming_idle_timeout=777.0,
            )

        assert received_kwargs.get("submit_result_timeout") == 999.0
        assert received_kwargs.get("max_resends") == 7
        assert received_kwargs.get("download_complete_timeout") == 3600.0
        assert received_kwargs.get("streaming_idle_timeout") == 777.0
