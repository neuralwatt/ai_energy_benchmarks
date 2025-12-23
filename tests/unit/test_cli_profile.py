"""Unit tests for CLI profile module."""

import sys
from unittest.mock import MagicMock, patch

import pytest


class TestCLIArgumentParsing:
    """Tests for CLI argument parsing."""

    def test_default_arguments(self):
        """Test CLI with default arguments."""
        with patch.object(sys, "argv", ["ai-energy-profile", "--model", "test-model"]):
            from ai_energy_benchmarks.cli.profile import main

            with patch("ai_energy_benchmarks.cli.profile.GenAIPerfExecutor") as mock_executor:
                mock_instance = MagicMock()
                mock_instance.run.return_value = True
                mock_executor.return_value = mock_instance

                with patch("ai_energy_benchmarks.cli.profile.get_profile") as mock_get:
                    from ai_energy_benchmarks.profiles import LoadProfileConfig

                    mock_get.return_value = LoadProfileConfig(name="moderate", request_count=40)

                    with patch("ai_energy_benchmarks.cli.profile.is_multi_phase") as mock_is_multi:
                        mock_is_multi.return_value = False

                        main()

                # Default profile should be moderate
                mock_get.assert_called_with("moderate")

    def test_profile_argument(self):
        """Test CLI with --profile argument."""
        with patch.object(
            sys, "argv", ["ai-energy-profile", "--profile", "stress", "--model", "test-model"]
        ):
            from ai_energy_benchmarks.cli.profile import main

            with patch("ai_energy_benchmarks.cli.profile.GenAIPerfExecutor") as mock_executor:
                mock_instance = MagicMock()
                mock_instance.run.return_value = True
                mock_executor.return_value = mock_instance

                with patch("ai_energy_benchmarks.cli.profile.get_profile") as mock_get:
                    from ai_energy_benchmarks.profiles import LoadProfileConfig

                    mock_get.return_value = LoadProfileConfig(name="stress", request_count=240)

                    with patch("ai_energy_benchmarks.cli.profile.is_multi_phase") as mock_is_multi:
                        mock_is_multi.return_value = False

                        main()

                mock_get.assert_called_with("stress")

    def test_endpoint_argument(self):
        """Test CLI with --endpoint argument."""
        with patch.object(
            sys,
            "argv",
            ["ai-energy-profile", "--endpoint", "http://custom:9000/v1", "--model", "test-model"],
        ):
            from ai_energy_benchmarks.cli.profile import main

            with patch("ai_energy_benchmarks.cli.profile.GenAIPerfExecutor") as mock_executor:
                mock_instance = MagicMock()
                mock_instance.run.return_value = True
                mock_executor.return_value = mock_instance

                with patch("ai_energy_benchmarks.cli.profile.get_profile") as mock_get:
                    from ai_energy_benchmarks.profiles import LoadProfileConfig

                    mock_get.return_value = LoadProfileConfig(name="moderate", request_count=40)

                    with patch("ai_energy_benchmarks.cli.profile.is_multi_phase") as mock_is_multi:
                        mock_is_multi.return_value = False

                        main()

                # Check that run was called with custom endpoint (positional arg)
                call_args = mock_instance.run.call_args
                # endpoint is the second positional arg (index 1)
                assert call_args[0][1] == "http://custom:9000/v1"

    def test_model_argument(self):
        """Test CLI with --model argument."""
        with patch.object(sys, "argv", ["ai-energy-profile", "--model", "custom-model"]):
            from ai_energy_benchmarks.cli.profile import main

            with patch("ai_energy_benchmarks.cli.profile.GenAIPerfExecutor") as mock_executor:
                mock_instance = MagicMock()
                mock_instance.run.return_value = True
                mock_executor.return_value = mock_instance

                with patch("ai_energy_benchmarks.cli.profile.get_profile") as mock_get:
                    from ai_energy_benchmarks.profiles import LoadProfileConfig

                    mock_get.return_value = LoadProfileConfig(name="moderate", request_count=40)

                    with patch("ai_energy_benchmarks.cli.profile.is_multi_phase") as mock_is_multi:
                        mock_is_multi.return_value = False

                        main()

                # model is the third positional arg (index 2)
                call_args = mock_instance.run.call_args
                assert call_args[0][2] == "custom-model"

    def test_seed_argument(self):
        """Test CLI with --seed argument."""
        with patch.object(
            sys, "argv", ["ai-energy-profile", "--seed", "42", "--model", "test-model"]
        ):
            from ai_energy_benchmarks.cli.profile import main

            with patch("ai_energy_benchmarks.cli.profile.GenAIPerfExecutor") as mock_executor:
                mock_instance = MagicMock()
                mock_instance.run.return_value = True
                mock_executor.return_value = mock_instance

                with patch("ai_energy_benchmarks.cli.profile.get_profile") as mock_get:
                    from ai_energy_benchmarks.profiles import LoadProfileConfig

                    mock_get.return_value = LoadProfileConfig(name="moderate", request_count=40)

                    with patch("ai_energy_benchmarks.cli.profile.is_multi_phase") as mock_is_multi:
                        mock_is_multi.return_value = False

                        main()

                # Executor should be created with seed
                mock_executor.assert_called_with(seed=42)


class TestCLIExecution:
    """Tests for CLI execution paths."""

    def test_single_phase_profile_execution(self):
        """Test CLI executes single-phase profile correctly."""
        with patch.object(
            sys, "argv", ["ai-energy-profile", "--profile", "light", "--model", "test-model"]
        ):
            from ai_energy_benchmarks.cli.profile import main

            with patch("ai_energy_benchmarks.cli.profile.GenAIPerfExecutor") as mock_executor:
                mock_instance = MagicMock()
                mock_instance.run.return_value = True
                mock_executor.return_value = mock_instance

                with patch("ai_energy_benchmarks.cli.profile.get_profile") as mock_get:
                    from ai_energy_benchmarks.profiles import LoadProfileConfig

                    mock_get.return_value = LoadProfileConfig(name="light", request_count=20)

                    with patch("ai_energy_benchmarks.cli.profile.is_multi_phase") as mock_is_multi:
                        mock_is_multi.return_value = False

                        main()

                # Should call run, not run_multi_phase
                mock_instance.run.assert_called_once()
                mock_instance.run_multi_phase.assert_not_called()

    def test_multi_phase_profile_execution(self):
        """Test CLI executes multi-phase profile correctly."""
        with patch.object(
            sys, "argv", ["ai-energy-profile", "--profile", "multiphase", "--model", "test-model"]
        ):
            from ai_energy_benchmarks.cli.profile import main

            with patch("ai_energy_benchmarks.cli.profile.GenAIPerfExecutor") as mock_executor:
                mock_instance = MagicMock()
                mock_instance.run_multi_phase.return_value = True
                mock_executor.return_value = mock_instance

                with patch("ai_energy_benchmarks.cli.profile.get_profile") as mock_get:
                    from ai_energy_benchmarks.profiles import (
                        LoadProfileConfig,
                        MultiPhaseProfile,
                    )

                    mock_get.return_value = MultiPhaseProfile(
                        name="multiphase",
                        phases=[LoadProfileConfig(name="test", request_count=10)],
                    )

                    with patch("ai_energy_benchmarks.cli.profile.is_multi_phase") as mock_is_multi:
                        mock_is_multi.return_value = True

                        main()

                # Should call run_multi_phase, not run
                mock_instance.run_multi_phase.assert_called_once()
                mock_instance.run.assert_not_called()

    def test_exit_on_failure(self):
        """Test CLI exits with code 1 on failure."""
        with patch.object(sys, "argv", ["ai-energy-profile", "--model", "test-model"]):
            from ai_energy_benchmarks.cli.profile import main

            with patch("ai_energy_benchmarks.cli.profile.GenAIPerfExecutor") as mock_executor:
                mock_instance = MagicMock()
                mock_instance.run.return_value = False  # Simulate failure
                mock_executor.return_value = mock_instance

                with patch("ai_energy_benchmarks.cli.profile.get_profile") as mock_get:
                    from ai_energy_benchmarks.profiles import LoadProfileConfig

                    mock_get.return_value = LoadProfileConfig(name="moderate", request_count=40)

                    with patch("ai_energy_benchmarks.cli.profile.is_multi_phase") as mock_is_multi:
                        mock_is_multi.return_value = False

                        with pytest.raises(SystemExit) as excinfo:
                            main()

                        assert excinfo.value.code == 1


class TestCLIHelp:
    """Tests for CLI help output."""

    def test_help_shows_all_profiles(self):
        """Test that --help shows all available profiles."""
        import subprocess
        import sys
        from pathlib import Path

        # Get the project root dynamically (works in any environment)
        project_root = Path(__file__).parent.parent.parent

        result = subprocess.run(
            [sys.executable, "-m", "ai_energy_benchmarks.cli.profile", "--help"],
            capture_output=True,
            text=True,
            cwd=project_root,
        )

        # Check all profiles are mentioned
        assert "light" in result.stdout
        assert "moderate" in result.stdout
        assert "heavy" in result.stdout
        assert "stress" in result.stdout
        assert "multiphase" in result.stdout
        assert "pattern" in result.stdout
        assert "power_test" in result.stdout

    def test_help_shows_examples(self):
        """Test that --help shows usage examples."""
        import subprocess
        import sys
        from pathlib import Path

        project_root = Path(__file__).parent.parent.parent

        result = subprocess.run(
            [sys.executable, "-m", "ai_energy_benchmarks.cli.profile", "--help"],
            capture_output=True,
            text=True,
            cwd=project_root,
        )

        assert "Examples:" in result.stdout
        assert "ai-energy-profile" in result.stdout


class TestCLIRunIdSuffix:
    """Tests for --run-id-suffix argument."""

    def test_run_id_suffix_passed_to_executor(self):
        """Test that run-id-suffix is passed to executor."""
        with patch.object(
            sys,
            "argv",
            ["ai-energy-profile", "--run-id-suffix", "test_suffix", "--model", "test-model"],
        ):
            from ai_energy_benchmarks.cli.profile import main

            with patch("ai_energy_benchmarks.cli.profile.GenAIPerfExecutor") as mock_executor:
                mock_instance = MagicMock()
                mock_instance.run.return_value = True
                mock_executor.return_value = mock_instance

                with patch("ai_energy_benchmarks.cli.profile.get_profile") as mock_get:
                    from ai_energy_benchmarks.profiles import LoadProfileConfig

                    mock_get.return_value = LoadProfileConfig(name="moderate", request_count=40)

                    with patch("ai_energy_benchmarks.cli.profile.is_multi_phase") as mock_is_multi:
                        mock_is_multi.return_value = False

                        main()

                call_args = mock_instance.run.call_args
                assert call_args[1]["run_id_suffix"] == "test_suffix"


class TestCLIApiKey:
    """Tests for --api-key argument."""

    def test_api_key_passed_to_executor(self):
        """Test that api-key is passed to executor run()."""
        with patch.object(
            sys,
            "argv",
            ["ai-energy-profile", "--api-key", "sk-test-key-123", "--model", "test-model"],
        ):
            from ai_energy_benchmarks.cli.profile import main

            with patch("ai_energy_benchmarks.cli.profile.GenAIPerfExecutor") as mock_executor:
                mock_instance = MagicMock()
                mock_instance.run.return_value = True
                mock_executor.return_value = mock_instance

                with patch("ai_energy_benchmarks.cli.profile.get_profile") as mock_get:
                    from ai_energy_benchmarks.profiles import LoadProfileConfig

                    mock_get.return_value = LoadProfileConfig(name="moderate", request_count=40)

                    with patch("ai_energy_benchmarks.cli.profile.is_multi_phase") as mock_is_multi:
                        mock_is_multi.return_value = False

                        main()

                call_args = mock_instance.run.call_args
                assert call_args[1]["api_key"] == "sk-test-key-123"

    def test_api_key_passed_to_multi_phase_executor(self):
        """Test that api-key is passed to run_multi_phase()."""
        with patch.object(
            sys,
            "argv",
            [
                "ai-energy-profile",
                "--profile",
                "multiphase",
                "--api-key",
                "sk-multi-key",
                "--model",
                "test-model",
            ],
        ):
            from ai_energy_benchmarks.cli.profile import main

            with patch("ai_energy_benchmarks.cli.profile.GenAIPerfExecutor") as mock_executor:
                mock_instance = MagicMock()
                mock_instance.run_multi_phase.return_value = True
                mock_executor.return_value = mock_instance

                with patch("ai_energy_benchmarks.cli.profile.get_profile") as mock_get:
                    from ai_energy_benchmarks.profiles import (
                        LoadProfileConfig,
                        MultiPhaseProfile,
                    )

                    mock_get.return_value = MultiPhaseProfile(
                        name="multiphase",
                        phases=[LoadProfileConfig(name="test", request_count=10)],
                    )

                    with patch("ai_energy_benchmarks.cli.profile.is_multi_phase") as mock_is_multi:
                        mock_is_multi.return_value = True

                        main()

                call_args = mock_instance.run_multi_phase.call_args
                assert call_args[1]["api_key"] == "sk-multi-key"

    def test_no_api_key_by_default(self):
        """Test that api_key is None by default."""
        with patch.object(sys, "argv", ["ai-energy-profile", "--model", "test-model"]):
            from ai_energy_benchmarks.cli.profile import main

            with patch("ai_energy_benchmarks.cli.profile.GenAIPerfExecutor") as mock_executor:
                mock_instance = MagicMock()
                mock_instance.run.return_value = True
                mock_executor.return_value = mock_instance

                with patch("ai_energy_benchmarks.cli.profile.get_profile") as mock_get:
                    from ai_energy_benchmarks.profiles import LoadProfileConfig

                    mock_get.return_value = LoadProfileConfig(name="moderate", request_count=40)

                    with patch("ai_energy_benchmarks.cli.profile.is_multi_phase") as mock_is_multi:
                        mock_is_multi.return_value = False

                        main()

                call_args = mock_instance.run.call_args
                assert call_args[1]["api_key"] is None


class TestCLIEndpointType:
    """Tests for --endpoint-type argument."""

    def test_endpoint_type_chat_passed_to_executor(self):
        """Test that endpoint-type 'chat' is passed to executor."""
        with patch.object(
            sys, "argv", ["ai-energy-profile", "--endpoint-type", "chat", "--model", "test-model"]
        ):
            from ai_energy_benchmarks.cli.profile import main

            with patch("ai_energy_benchmarks.cli.profile.GenAIPerfExecutor") as mock_executor:
                mock_instance = MagicMock()
                mock_instance.run.return_value = True
                mock_executor.return_value = mock_instance

                with patch("ai_energy_benchmarks.cli.profile.get_profile") as mock_get:
                    from ai_energy_benchmarks.profiles import LoadProfileConfig

                    mock_get.return_value = LoadProfileConfig(name="moderate", request_count=40)

                    with patch("ai_energy_benchmarks.cli.profile.is_multi_phase") as mock_is_multi:
                        mock_is_multi.return_value = False

                        main()

                call_args = mock_instance.run.call_args
                assert call_args[1]["endpoint_type"] == "chat"

    def test_endpoint_type_completions_passed_to_executor(self):
        """Test that endpoint-type 'completions' is passed to executor."""
        with patch.object(
            sys,
            "argv",
            ["ai-energy-profile", "--endpoint-type", "completions", "--model", "test-model"],
        ):
            from ai_energy_benchmarks.cli.profile import main

            with patch("ai_energy_benchmarks.cli.profile.GenAIPerfExecutor") as mock_executor:
                mock_instance = MagicMock()
                mock_instance.run.return_value = True
                mock_executor.return_value = mock_instance

                with patch("ai_energy_benchmarks.cli.profile.get_profile") as mock_get:
                    from ai_energy_benchmarks.profiles import LoadProfileConfig

                    mock_get.return_value = LoadProfileConfig(name="moderate", request_count=40)

                    with patch("ai_energy_benchmarks.cli.profile.is_multi_phase") as mock_is_multi:
                        mock_is_multi.return_value = False

                        main()

                call_args = mock_instance.run.call_args
                assert call_args[1]["endpoint_type"] == "completions"

    def test_endpoint_type_default_is_chat(self):
        """Test that default endpoint-type is 'chat'."""
        with patch.object(sys, "argv", ["ai-energy-profile", "--model", "test-model"]):
            from ai_energy_benchmarks.cli.profile import main

            with patch("ai_energy_benchmarks.cli.profile.GenAIPerfExecutor") as mock_executor:
                mock_instance = MagicMock()
                mock_instance.run.return_value = True
                mock_executor.return_value = mock_instance

                with patch("ai_energy_benchmarks.cli.profile.get_profile") as mock_get:
                    from ai_energy_benchmarks.profiles import LoadProfileConfig

                    mock_get.return_value = LoadProfileConfig(name="moderate", request_count=40)

                    with patch("ai_energy_benchmarks.cli.profile.is_multi_phase") as mock_is_multi:
                        mock_is_multi.return_value = False

                        main()

                call_args = mock_instance.run.call_args
                assert call_args[1]["endpoint_type"] == "chat"

    def test_endpoint_type_passed_to_multi_phase_executor(self):
        """Test that endpoint-type is passed to run_multi_phase()."""
        with patch.object(
            sys,
            "argv",
            [
                "ai-energy-profile",
                "--profile",
                "multiphase",
                "--endpoint-type",
                "completions",
                "--model",
                "test-model",
            ],
        ):
            from ai_energy_benchmarks.cli.profile import main

            with patch("ai_energy_benchmarks.cli.profile.GenAIPerfExecutor") as mock_executor:
                mock_instance = MagicMock()
                mock_instance.run_multi_phase.return_value = True
                mock_executor.return_value = mock_instance

                with patch("ai_energy_benchmarks.cli.profile.get_profile") as mock_get:
                    from ai_energy_benchmarks.profiles import (
                        LoadProfileConfig,
                        MultiPhaseProfile,
                    )

                    mock_get.return_value = MultiPhaseProfile(
                        name="multiphase",
                        phases=[LoadProfileConfig(name="test", request_count=10)],
                    )

                    with patch("ai_energy_benchmarks.cli.profile.is_multi_phase") as mock_is_multi:
                        mock_is_multi.return_value = True

                        main()

                call_args = mock_instance.run_multi_phase.call_args
                assert call_args[1]["endpoint_type"] == "completions"

    def test_help_shows_endpoint_type_options(self):
        """Test that --help shows endpoint-type options."""
        import subprocess
        import sys
        from pathlib import Path

        project_root = Path(__file__).parent.parent.parent

        result = subprocess.run(
            [sys.executable, "-m", "ai_energy_benchmarks.cli.profile", "--help"],
            capture_output=True,
            text=True,
            cwd=project_root,
        )

        assert "--endpoint-type" in result.stdout
        assert "chat" in result.stdout
        assert "completions" in result.stdout

    def test_help_shows_api_key_option(self):
        """Test that --help shows api-key option."""
        import subprocess
        import sys
        from pathlib import Path

        project_root = Path(__file__).parent.parent.parent

        result = subprocess.run(
            [sys.executable, "-m", "ai_energy_benchmarks.cli.profile", "--help"],
            capture_output=True,
            text=True,
            cwd=project_root,
        )

        assert "--api-key" in result.stdout
        # Help text wraps, so check for parts separately
        assert "Bearer" in result.stdout
        assert "authenticated" in result.stdout
