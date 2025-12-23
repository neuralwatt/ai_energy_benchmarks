"""Unit tests for GenAIPerfExecutor."""

import json
import os
import tempfile
from unittest.mock import MagicMock, patch

from ai_energy_benchmarks.executors.genai_perf import GenAIPerfExecutor
from ai_energy_benchmarks.profiles import LoadProfileConfig, MultiPhaseProfile


class TestGenAIPerfExecutorInit:
    """Tests for GenAIPerfExecutor initialization."""

    def test_init_default(self):
        """Test executor initializes with default values."""
        executor = GenAIPerfExecutor()

        assert executor.seed is None
        assert executor.running is True
        assert executor.current_process is None

    def test_init_with_seed(self):
        """Test executor initializes with seed."""
        executor = GenAIPerfExecutor(seed=42)

        assert executor.seed == 42
        assert executor.running is True


class TestBuildCommand:
    """Tests for GenAIPerfExecutor.build_command()."""

    def test_build_command_basic(self):
        """Test basic command building."""
        executor = GenAIPerfExecutor()
        profile = LoadProfileConfig(
            name="test",
            concurrency=4,
            input_token_range=(100, 200),
            output_token_range=(200, 400),
            request_count=100,
        )

        cmd = executor.build_command(
            profile,
            endpoint="http://localhost:8000/v1",
            model="test-model",
            output_dir="/tmp/output",
        )

        assert "genai-perf" in cmd
        assert "profile" in cmd
        assert "--model" in cmd
        assert "test-model" in cmd
        assert "--endpoint-type" in cmd
        assert "chat" in cmd  # Default endpoint type is chat
        assert "--url" in cmd
        assert "http://localhost:8000" in cmd
        assert "--concurrency" in cmd
        assert "4" in cmd
        assert "--artifact-dir" in cmd
        assert "/tmp/output" in cmd

    def test_build_command_strips_v1_from_endpoint(self):
        """Test that /v1 is stripped from endpoint URL."""
        executor = GenAIPerfExecutor()
        profile = LoadProfileConfig(name="test", concurrency=1, request_count=100)

        cmd = executor.build_command(
            profile,
            endpoint="http://localhost:8000/v1",
            model="test-model",
            output_dir="/tmp/output",
        )

        # Find the URL in command
        url_idx = cmd.index("--url") + 1
        assert cmd[url_idx] == "http://localhost:8000"

    def test_build_command_with_request_count(self):
        """Test command building with request count."""
        executor = GenAIPerfExecutor()
        profile = LoadProfileConfig(name="test", concurrency=1, request_count=100)

        cmd = executor.build_command(
            profile,
            endpoint="http://localhost:8000/v1",
            model="test-model",
            output_dir="/tmp/output",
        )

        assert "--request-count" in cmd
        count_idx = cmd.index("--request-count") + 1
        assert cmd[count_idx] == "100"

    def test_build_command_request_count_at_least_concurrency(self):
        """Test that request count is at least as large as concurrency."""
        executor = GenAIPerfExecutor()
        profile = LoadProfileConfig(
            name="test",
            concurrency=50,
            request_count=10,  # Less than concurrency
        )

        cmd = executor.build_command(
            profile,
            endpoint="http://localhost:8000/v1",
            model="test-model",
            output_dir="/tmp/output",
        )

        assert "--request-count" in cmd
        count_idx = cmd.index("--request-count") + 1
        # Should be at least concurrency
        assert int(cmd[count_idx]) >= 50

    def test_build_command_with_dict_profile(self):
        """Test command building with dict profile."""
        executor = GenAIPerfExecutor()
        profile = {
            "name": "test",
            "concurrency": 8,
            "input_token_range": [150, 300],
            "output_token_range": [300, 600],
            "request_count": 50,
        }

        cmd = executor.build_command(
            profile,
            endpoint="http://localhost:8000/v1",
            model="test-model",
            output_dir="/tmp/output",
        )

        assert "--concurrency" in cmd
        conc_idx = cmd.index("--concurrency") + 1
        assert cmd[conc_idx] == "8"

    def test_build_command_calculates_token_averages(self):
        """Test that command calculates token averages correctly."""
        executor = GenAIPerfExecutor()
        profile = LoadProfileConfig(
            name="test",
            concurrency=1,
            request_count=100,
            input_token_range=(100, 300),  # avg = 200
            output_token_range=(400, 600),  # avg = 500
        )

        cmd = executor.build_command(
            profile,
            endpoint="http://localhost:8000/v1",
            model="test-model",
            output_dir="/tmp/output",
        )

        assert "--synthetic-input-tokens-mean" in cmd
        input_idx = cmd.index("--synthetic-input-tokens-mean") + 1
        assert cmd[input_idx] == "200"

        assert "--output-tokens-mean" in cmd
        output_idx = cmd.index("--output-tokens-mean") + 1
        assert cmd[output_idx] == "500"

    def test_build_command_with_seed_adds_stddev(self):
        """Test that seed causes token stddev to be added."""
        executor = GenAIPerfExecutor(seed=42)
        profile = LoadProfileConfig(
            name="test",
            concurrency=1,
            request_count=100,
            input_token_range=(100, 300),
            output_token_range=(400, 600),
        )

        cmd = executor.build_command(
            profile,
            endpoint="http://localhost:8000/v1",
            model="test-model",
            output_dir="/tmp/output",
        )

        assert "--synthetic-input-tokens-stddev" in cmd
        assert "--output-tokens-stddev" in cmd
        assert "--random-seed" in cmd

    def test_build_command_with_api_key(self):
        """Test that API key is added as Authorization header."""
        executor = GenAIPerfExecutor()
        profile = LoadProfileConfig(name="test", concurrency=1, request_count=100)

        cmd = executor.build_command(
            profile,
            endpoint="http://localhost:8000/v1",
            model="test-model",
            output_dir="/tmp/output",
            api_key="test-api-key-123",
        )

        assert "--header" in cmd
        header_idx = cmd.index("--header") + 1
        assert cmd[header_idx] == "Authorization: Bearer test-api-key-123"

    def test_build_command_without_api_key(self):
        """Test that no header is added when API key is None."""
        executor = GenAIPerfExecutor()
        profile = LoadProfileConfig(name="test", concurrency=1, request_count=100)

        cmd = executor.build_command(
            profile,
            endpoint="http://localhost:8000/v1",
            model="test-model",
            output_dir="/tmp/output",
            api_key=None,
        )

        assert "--header" not in cmd

    def test_build_command_with_endpoint_type_chat(self):
        """Test that endpoint type 'chat' is used correctly."""
        executor = GenAIPerfExecutor()
        profile = LoadProfileConfig(name="test", concurrency=1, request_count=100)

        cmd = executor.build_command(
            profile,
            endpoint="http://localhost:8000/v1",
            model="test-model",
            output_dir="/tmp/output",
            endpoint_type="chat",
        )

        assert "--endpoint-type" in cmd
        endpoint_idx = cmd.index("--endpoint-type") + 1
        assert cmd[endpoint_idx] == "chat"

    def test_build_command_with_endpoint_type_completions(self):
        """Test that endpoint type 'completions' is used correctly."""
        executor = GenAIPerfExecutor()
        profile = LoadProfileConfig(name="test", concurrency=1, request_count=100)

        cmd = executor.build_command(
            profile,
            endpoint="http://localhost:8000/v1",
            model="test-model",
            output_dir="/tmp/output",
            endpoint_type="completions",
        )

        assert "--endpoint-type" in cmd
        endpoint_idx = cmd.index("--endpoint-type") + 1
        assert cmd[endpoint_idx] == "completions"

    def test_build_command_default_endpoint_type_is_chat(self):
        """Test that default endpoint type is 'chat'."""
        executor = GenAIPerfExecutor()
        profile = LoadProfileConfig(name="test", concurrency=1, request_count=100)

        # Don't pass endpoint_type, use default
        cmd = executor.build_command(
            profile,
            endpoint="http://localhost:8000/v1",
            model="test-model",
            output_dir="/tmp/output",
        )

        assert "--endpoint-type" in cmd
        endpoint_idx = cmd.index("--endpoint-type") + 1
        assert cmd[endpoint_idx] == "chat"

    def test_build_command_with_api_key_and_endpoint_type(self):
        """Test that both API key and endpoint type work together."""
        executor = GenAIPerfExecutor()
        profile = LoadProfileConfig(name="test", concurrency=1, request_count=100)

        cmd = executor.build_command(
            profile,
            endpoint="https://api.example.com/v1",
            model="test-model",
            output_dir="/tmp/output",
            api_key="sk-secret-key",
            endpoint_type="chat",
        )

        # Check API key header
        assert "--header" in cmd
        header_idx = cmd.index("--header") + 1
        assert cmd[header_idx] == "Authorization: Bearer sk-secret-key"

        # Check endpoint type
        assert "--endpoint-type" in cmd
        endpoint_idx = cmd.index("--endpoint-type") + 1
        assert cmd[endpoint_idx] == "chat"

        # Check URL is stripped of /v1
        url_idx = cmd.index("--url") + 1
        assert cmd[url_idx] == "https://api.example.com"


class TestGenerateDeterministicPrompts:
    """Tests for GenAIPerfExecutor.generate_deterministic_prompts()."""

    def test_generates_prompts_file(self):
        """Test that prompts file is generated."""
        executor = GenAIPerfExecutor(seed=42)
        profile = LoadProfileConfig(
            name="test",
            input_token_range=(50, 100),
            request_count=5,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            result = executor.generate_deterministic_prompts(profile, tmpdir)

            assert result is not None
            assert os.path.exists(result)
            assert result.endswith(".jsonl")

    def test_prompts_file_has_correct_count(self):
        """Test that prompts file has correct number of entries."""
        executor = GenAIPerfExecutor(seed=42)
        profile = LoadProfileConfig(
            name="test",
            input_token_range=(50, 100),
            request_count=10,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            result = executor.generate_deterministic_prompts(profile, tmpdir)
            assert result is not None

            with open(result, "r") as f:
                lines = [line for line in f if line.strip()]

            assert len(lines) == 10

    def test_prompts_are_valid_json(self):
        """Test that each prompt is valid JSON."""
        executor = GenAIPerfExecutor(seed=42)
        profile = LoadProfileConfig(
            name="test",
            input_token_range=(50, 100),
            request_count=5,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            result = executor.generate_deterministic_prompts(profile, tmpdir)
            assert result is not None

            with open(result, "r") as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line)
                        assert "text_input" in data
                        assert "stream" in data

    def test_prompts_are_deterministic(self):
        """Test that same seed produces same prompts."""
        profile = LoadProfileConfig(
            name="test",
            input_token_range=(50, 100),
            request_count=5,
        )

        with tempfile.TemporaryDirectory() as tmpdir1:
            executor1 = GenAIPerfExecutor(seed=42)
            result1 = executor1.generate_deterministic_prompts(profile, tmpdir1)
            assert result1 is not None
            with open(result1, "r") as f:
                content1 = f.read()

        with tempfile.TemporaryDirectory() as tmpdir2:
            executor2 = GenAIPerfExecutor(seed=42)
            result2 = executor2.generate_deterministic_prompts(profile, tmpdir2)
            assert result2 is not None
            with open(result2, "r") as f:
                content2 = f.read()

        assert content1 == content2

    def test_different_seeds_produce_different_prompts(self):
        """Test that different seeds produce different prompts."""
        profile = LoadProfileConfig(
            name="test",
            input_token_range=(50, 100),
            request_count=5,
        )

        with tempfile.TemporaryDirectory() as tmpdir1:
            executor1 = GenAIPerfExecutor(seed=42)
            result1 = executor1.generate_deterministic_prompts(profile, tmpdir1)
            assert result1 is not None
            with open(result1, "r") as f:
                content1 = f.read()

        with tempfile.TemporaryDirectory() as tmpdir2:
            executor2 = GenAIPerfExecutor(seed=99)
            result2 = executor2.generate_deterministic_prompts(profile, tmpdir2)
            assert result2 is not None
            with open(result2, "r") as f:
                content2 = f.read()

        assert content1 != content2

    def test_caches_prompts_file(self):
        """Test that prompts file is cached on second call."""
        executor = GenAIPerfExecutor(seed=42)
        profile = LoadProfileConfig(
            name="test",
            input_token_range=(50, 100),
            request_count=5,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            result1 = executor.generate_deterministic_prompts(profile, tmpdir)
            assert result1 is not None
            mtime1 = os.path.getmtime(result1)

            # Second call should return cached file
            result2 = executor.generate_deterministic_prompts(profile, tmpdir)
            assert result2 is not None
            mtime2 = os.path.getmtime(result2)

            assert result1 == result2
            assert mtime1 == mtime2


class TestRunMethod:
    """Tests for GenAIPerfExecutor.run() method."""

    @patch("subprocess.Popen")
    def test_run_creates_output_directory(self, mock_popen):
        """Test that run creates output directory."""
        mock_process = MagicMock()
        mock_process.stdout.readline.side_effect = ["", ""]
        mock_process.poll.return_value = 0
        mock_process.returncode = 0
        mock_process.wait.return_value = None
        mock_popen.return_value = mock_process

        executor = GenAIPerfExecutor()
        profile = LoadProfileConfig(name="test", concurrency=1, request_count=100)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = os.path.join(tmpdir, "new_subdir")
            assert not os.path.exists(output_dir)

            with patch.object(executor, "build_command", return_value=["echo", "test"]):
                executor.run(
                    profile,
                    endpoint="http://localhost:8000/v1",
                    model="test-model",
                    output_dir=output_dir,
                )

            assert os.path.exists(output_dir)

    @patch("subprocess.Popen")
    def test_run_returns_true_on_success(self, mock_popen):
        """Test that run returns True on successful execution."""
        mock_process = MagicMock()
        mock_process.stdout.readline.side_effect = ["output line", ""]
        mock_process.poll.return_value = 0
        mock_process.returncode = 0
        mock_process.wait.return_value = None
        mock_popen.return_value = mock_process

        executor = GenAIPerfExecutor()
        profile = LoadProfileConfig(name="test", concurrency=1, request_count=100)

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.object(executor, "build_command", return_value=["echo", "test"]):
                with patch("ai_energy_benchmarks.formatters.episode.EpisodeFormatter"):
                    result = executor.run(
                        profile,
                        endpoint="http://localhost:8000/v1",
                        model="test-model",
                        output_dir=tmpdir,
                    )

        assert result is True

    @patch("subprocess.Popen")
    def test_run_returns_false_on_failure(self, mock_popen):
        """Test that run returns False on failed execution."""
        mock_process = MagicMock()
        mock_process.stdout.readline.side_effect = ["error", ""]
        mock_process.poll.return_value = 1
        mock_process.returncode = 1
        mock_process.wait.return_value = None
        mock_popen.return_value = mock_process

        executor = GenAIPerfExecutor()
        profile = LoadProfileConfig(name="test", concurrency=1, request_count=100)

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.object(executor, "build_command", return_value=["false"]):
                result = executor.run(
                    profile,
                    endpoint="http://localhost:8000/v1",
                    model="test-model",
                    output_dir=tmpdir,
                )

        assert result is False

    def test_run_returns_false_when_not_running(self):
        """Test that run returns False when executor is stopped."""
        executor = GenAIPerfExecutor()
        executor.running = False

        profile = LoadProfileConfig(name="test", concurrency=1, request_count=100)

        result = executor.run(
            profile,
            endpoint="http://localhost:8000/v1",
            model="test-model",
            output_dir="/tmp",
        )

        assert result is False


class TestRunMultiPhaseMethod:
    """Tests for GenAIPerfExecutor.run_multi_phase() method."""

    def test_run_multi_phase_empty_phases(self):
        """Test run_multi_phase with empty phases list."""
        executor = GenAIPerfExecutor()
        profile = MultiPhaseProfile(name="empty", phases=[])

        result = executor.run_multi_phase(
            profile,
            endpoint="http://localhost:8000/v1",
            model="test-model",
            output_dir="/tmp",
        )

        assert result is False

    @patch.object(GenAIPerfExecutor, "run")
    def test_run_multi_phase_executes_phases(self, mock_run):
        """Test run_multi_phase executes each phase."""
        mock_run.return_value = True

        executor = GenAIPerfExecutor()
        phases = [
            LoadProfileConfig(name="phase1", concurrency=2, request_count=10),
            LoadProfileConfig(name="phase2", concurrency=4, request_count=20),
        ]
        profile = MultiPhaseProfile(name="test", phases=phases)

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch(
                "ai_energy_benchmarks.formatters.episode.EpisodeFormatter"
            ) as mock_formatter:
                mock_formatter.return_value.aggregate_phases.return_value = True
                result = executor.run_multi_phase(
                    profile,
                    endpoint="http://localhost:8000/v1",
                    model="test-model",
                    output_dir=tmpdir,
                )

        assert result is True
        assert mock_run.call_count == 2

    @patch("time.sleep")
    @patch.object(GenAIPerfExecutor, "run")
    def test_run_multi_phase_skips_zero_request_phases(self, mock_run, mock_sleep):
        """Test run_multi_phase skips phases with 0 requests."""
        mock_run.return_value = True

        executor = GenAIPerfExecutor()
        phases = [
            LoadProfileConfig(name="skip", concurrency=0, request_count=0),
            LoadProfileConfig(name="active", concurrency=4, request_count=20),
        ]
        profile = MultiPhaseProfile(name="test", phases=phases)

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch(
                "ai_energy_benchmarks.formatters.episode.EpisodeFormatter"
            ) as mock_formatter:
                mock_formatter.return_value.aggregate_phases.return_value = True
                executor.run_multi_phase(
                    profile,
                    endpoint="http://localhost:8000/v1",
                    model="test-model",
                    output_dir=tmpdir,
                )

        # Should only run the active phase
        assert mock_run.call_count == 1

    def test_run_multi_phase_returns_false_when_not_running(self):
        """Test run_multi_phase returns False when executor is stopped."""
        executor = GenAIPerfExecutor()
        executor.running = False

        phases = [
            LoadProfileConfig(name="phase1", concurrency=2, request_count=10),
        ]
        profile = MultiPhaseProfile(name="test", phases=phases)

        # Should exit early without executing phases
        with patch.object(GenAIPerfExecutor, "run") as mock_run:
            executor.run_multi_phase(
                profile,
                endpoint="http://localhost:8000/v1",
                model="test-model",
                output_dir="/tmp",
            )

        assert mock_run.call_count == 0


class TestSignalHandling:
    """Tests for signal handling in GenAIPerfExecutor."""

    def test_signal_handler_sets_running_false(self):
        """Test that signal handler sets running to False."""
        executor = GenAIPerfExecutor()
        assert executor.running is True

        # Simulate signal
        executor._signal_handler(2, None)  # SIGINT

        assert executor.running is False

    @patch("subprocess.Popen")
    def test_signal_handler_terminates_process(self, mock_popen):
        """Test that signal handler terminates running process."""
        mock_process = MagicMock()
        mock_popen.return_value = mock_process

        executor = GenAIPerfExecutor()
        executor.current_process = mock_process

        executor._signal_handler(2, None)

        mock_process.terminate.assert_called_once()
