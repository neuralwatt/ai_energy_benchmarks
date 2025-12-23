"""Unit tests for EpisodeFormatter."""

import json
import os
import tempfile

from ai_energy_benchmarks.formatters.episode import EpisodeFormatter


class TestEpisodeFormatterInit:
    """Tests for EpisodeFormatter initialization."""

    def test_init(self):
        """Test formatter initializes correctly."""
        formatter = EpisodeFormatter()
        assert formatter is not None


class TestFixOutput:
    """Tests for EpisodeFormatter.fix_output() method."""

    def test_fix_output_no_files(self):
        """Test fix_output with no genai-perf files."""
        formatter = EpisodeFormatter()

        with tempfile.TemporaryDirectory() as tmpdir:
            result = formatter.fix_output(tmpdir)

        assert result is False

    def test_fix_output_extracts_episode_id_from_path(self):
        """Test fix_output extracts episode ID from path."""
        formatter = EpisodeFormatter()

        with tempfile.TemporaryDirectory() as tmpdir:
            episode_dir = os.path.join(tmpdir, "episode_123")
            os.makedirs(episode_dir)

            # Create mock genai-perf output
            json_data = {
                "request_count": {"avg": 10},
                "output_sequence_length": {"avg": 100},
                "input_sequence_length": {"avg": 50},
                "request_latency": {"avg": 1000},
            }

            json_path = os.path.join(episode_dir, "profile_export_genai_perf.json")
            with open(json_path, "w") as f:
                json.dump(json_data, f)

            result = formatter.fix_output(episode_dir)

        assert result is True


class TestFallbackRename:
    """Tests for EpisodeFormatter._fallback_rename() method."""

    def test_fallback_rename_no_csv_files(self):
        """Test fallback_rename with no CSV files."""
        formatter = EpisodeFormatter()

        with tempfile.TemporaryDirectory() as tmpdir:
            result = formatter._fallback_rename(tmpdir)

        assert result is False

    def test_fallback_rename_renames_files(self):
        """Test fallback_rename renames CSV files."""
        formatter = EpisodeFormatter()

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create mock genai-perf CSV file
            csv_path = os.path.join(tmpdir, "profile_export_genai_perf.csv")
            with open(csv_path, "w") as f:
                f.write("header1,header2\n")
                f.write("value1,value2\n")

            result = formatter._fallback_rename(tmpdir)

            # Original file should be gone
            assert not os.path.exists(csv_path)

            # New file should exist with renamed pattern
            csv_files = [f for f in os.listdir(tmpdir) if f.endswith(".csv")]
            assert len(csv_files) == 1
            assert "inference.load.genai_perf_phase_" in csv_files[0]

        assert result is True


class TestAggregatePhases:
    """Tests for EpisodeFormatter.aggregate_phases() method."""

    def test_aggregate_phases_no_json_files(self):
        """Test aggregate_phases with no JSON files."""
        formatter = EpisodeFormatter()

        with tempfile.TemporaryDirectory() as tmpdir:
            result = formatter.aggregate_phases(tmpdir, "test_episode")

        assert result is False

    def test_aggregate_phases_creates_episode_csv(self):
        """Test aggregate_phases creates episode summary CSV."""
        formatter = EpisodeFormatter()

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create mock phase output
            phase_dir = os.path.join(tmpdir, "phase_1_test")
            os.makedirs(phase_dir)

            json_data = {
                "request_count": {"avg": 10},
                "output_sequence_length": {"avg": 100},
                "input_sequence_length": {"avg": 50},
                "request_latency": {"avg": 1000},  # 1 second
            }

            json_path = os.path.join(phase_dir, "profile_export_genai_perf.json")
            with open(json_path, "w") as f:
                json.dump(json_data, f)

            result = formatter.aggregate_phases(tmpdir, "test_episode")

            # Check episode CSV was created
            csv_files = [
                f for f in os.listdir(tmpdir) if f.startswith("inference.load.genai_perf_episode_")
            ]
            assert len(csv_files) == 1

        assert result is True

    def test_aggregate_phases_multiple_phases(self):
        """Test aggregate_phases with multiple phases."""
        formatter = EpisodeFormatter()

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create mock phase outputs
            for i, (name, requests) in enumerate([("light", 10), ("moderate", 20), ("stress", 50)]):
                phase_dir = os.path.join(tmpdir, f"phase_{i + 1}_{name}")
                os.makedirs(phase_dir)

                json_data = {
                    "request_count": {"avg": requests},
                    "output_sequence_length": {"avg": 100},
                    "input_sequence_length": {"avg": 50},
                    "request_latency": {"avg": 500},
                }

                json_path = os.path.join(phase_dir, "profile_export_genai_perf.json")
                with open(json_path, "w") as f:
                    json.dump(json_data, f)

            result = formatter.aggregate_phases(tmpdir, "multi_episode")

        assert result is True

    def test_aggregate_phases_skips_empty_phases(self):
        """Test aggregate_phases skips phases with no requests."""
        formatter = EpisodeFormatter()

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create phase with no requests
            phase_dir = os.path.join(tmpdir, "phase_1_idle")
            os.makedirs(phase_dir)

            json_data = {
                "request_count": {"avg": 0},
                "output_sequence_length": {"avg": 0},
                "input_sequence_length": {"avg": 0},
                "request_latency": {"avg": 0},
            }

            json_path = os.path.join(phase_dir, "profile_export_genai_perf.json")
            with open(json_path, "w") as f:
                json.dump(json_data, f)

            result = formatter.aggregate_phases(tmpdir, "empty_episode")

        # Should return False since no valid phases
        assert result is False

    def test_aggregate_phases_with_run_id_suffix(self):
        """Test aggregate_phases includes run_id_suffix in output."""
        formatter = EpisodeFormatter()

        with tempfile.TemporaryDirectory() as tmpdir:
            phase_dir = os.path.join(tmpdir, "phase_1_test")
            os.makedirs(phase_dir)

            json_data = {
                "request_count": {"avg": 10},
                "output_sequence_length": {"avg": 100},
                "input_sequence_length": {"avg": 50},
                "request_latency": {"avg": 1000},
            }

            json_path = os.path.join(phase_dir, "profile_export_genai_perf.json")
            with open(json_path, "w") as f:
                json.dump(json_data, f)

            result = formatter.aggregate_phases(tmpdir, "test_episode", run_id_suffix="monitor")

            # Read the generated CSV and check RunId
            import pandas as pd

            csv_files = [
                f for f in os.listdir(tmpdir) if f.startswith("inference.load.genai_perf_episode_")
            ]
            df = pd.read_csv(os.path.join(tmpdir, csv_files[0]))

            assert "monitor" in df["RunId"].iloc[0]

        assert result is True


class TestExtractActualTiming:
    """Tests for EpisodeFormatter._extract_actual_timing() method."""

    def test_extract_timing_no_files(self):
        """Test extract_timing with no matching files."""
        formatter = EpisodeFormatter()

        start, end = formatter._extract_actual_timing([])

        assert start is None
        assert end is None

    def test_extract_timing_from_profile_export(self):
        """Test extract_timing extracts timestamps from profile_export.json."""
        formatter = EpisodeFormatter()

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create profile_export_genai_perf.json (dummy)
            genai_json = os.path.join(tmpdir, "profile_export_genai_perf.json")
            with open(genai_json, "w") as f:
                json.dump({}, f)

            # Create profile_export.json with timestamps
            profile_json = os.path.join(tmpdir, "profile_export.json")
            # Timestamps in nanoseconds
            ts1 = 1700000000_000_000_000  # ~2023-11-14
            ts2 = 1700000100_000_000_000  # 100 seconds later

            profile_data = {
                "experiments": [
                    {
                        "requests": [
                            {"timestamp": ts1},
                            {"timestamp": ts2},
                        ]
                    }
                ]
            }
            with open(profile_json, "w") as f:
                json.dump(profile_data, f)

            start, end = formatter._extract_actual_timing([genai_json])

            assert start is not None
            assert end is not None
            assert end > start
            # Difference should be ~100 seconds
            diff = (end - start).total_seconds()
            assert 99 < diff < 101


class TestExtractActualTokens:
    """Tests for EpisodeFormatter._extract_actual_tokens() method."""

    def test_extract_tokens_no_files(self):
        """Test extract_tokens with no matching files."""
        formatter = EpisodeFormatter()

        prompt, response = formatter._extract_actual_tokens([])

        assert prompt == 0
        assert response == 0

    def test_extract_tokens_from_usage(self):
        """Test extract_tokens extracts from usage field in responses."""
        formatter = EpisodeFormatter()

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create profile_export_genai_perf.json (dummy)
            genai_json = os.path.join(tmpdir, "profile_export_genai_perf.json")
            with open(genai_json, "w") as f:
                json.dump({}, f)

            # Create profile_export.json with usage data
            response_json = json.dumps({"usage": {"prompt_tokens": 100, "completion_tokens": 200}})

            profile_data = {
                "experiments": [
                    {
                        "requests": [
                            {"response_outputs": [{"response": response_json}]},
                            {"response_outputs": [{"response": response_json}]},
                        ]
                    }
                ]
            }

            profile_json = os.path.join(tmpdir, "profile_export.json")
            with open(profile_json, "w") as f:
                json.dump(profile_data, f)

            prompt, response = formatter._extract_actual_tokens([genai_json])

            assert prompt == 200  # 100 * 2
            assert response == 400  # 200 * 2


class TestCleanupPhaseFiles:
    """Tests for EpisodeFormatter._cleanup_phase_files() method."""

    def test_cleanup_removes_phase_csv_files(self):
        """Test cleanup removes individual phase CSV files."""
        formatter = EpisodeFormatter()

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create some phase CSV files
            phase_file1 = os.path.join(tmpdir, "genai_perf_phase_20240101_120000.csv")
            phase_file2 = os.path.join(tmpdir, "genai_perf_phase_20240101_120100.csv")

            for f in [phase_file1, phase_file2]:
                with open(f, "w") as fp:
                    fp.write("test")

            # Create episode summary (should not be deleted)
            episode_file = os.path.join(
                tmpdir, "inference.load.genai_perf_episode_20240101_120200.csv"
            )
            with open(episode_file, "w") as ep_fp:
                ep_fp.write("test")

            formatter._cleanup_phase_files(tmpdir)

            # Phase files should be removed
            assert not os.path.exists(phase_file1)
            assert not os.path.exists(phase_file2)

            # Episode file should remain
            assert os.path.exists(episode_file)

    def test_cleanup_no_phase_files(self):
        """Test cleanup with no phase files does nothing."""
        formatter = EpisodeFormatter()

        with tempfile.TemporaryDirectory() as tmpdir:
            # No phase files, just episode summary
            episode_file = os.path.join(tmpdir, "inference.load.genai_perf_episode.csv")
            with open(episode_file, "w") as f:
                f.write("test")

            formatter._cleanup_phase_files(tmpdir)

            # Episode file should still exist
            assert os.path.exists(episode_file)


class TestEpisodeSummaryFormat:
    """Tests for the format of generated episode summaries."""

    def test_episode_summary_has_required_columns(self):
        """Test that episode summary CSV has all required columns."""
        formatter = EpisodeFormatter()

        with tempfile.TemporaryDirectory() as tmpdir:
            phase_dir = os.path.join(tmpdir, "phase_1_test")
            os.makedirs(phase_dir)

            json_data = {
                "request_count": {"avg": 10},
                "output_sequence_length": {"avg": 100},
                "input_sequence_length": {"avg": 50},
                "request_latency": {"avg": 1000},
            }

            json_path = os.path.join(phase_dir, "profile_export_genai_perf.json")
            with open(json_path, "w") as f:
                json.dump(json_data, f)

            formatter.aggregate_phases(tmpdir, "test_episode")

            import pandas as pd

            csv_files = [
                f for f in os.listdir(tmpdir) if f.startswith("inference.load.genai_perf_episode_")
            ]
            df = pd.read_csv(os.path.join(tmpdir, csv_files[0]))

            required_columns = [
                "RunId",
                "StartTime",
                "EndTime",
                "TotalTokens",
                "TotalDurationSeconds",
                "TokensPerSecond",
                "PromptTokens",
                "PromptEvalDurationSeconds",
                "PromptTokensPerSecond",
                "ResponseTokens",
                "ResponseEvalDurationSeconds",
                "ResponseTokensPerSecond",
                "LongOrShortPrompt",
                "Model",
            ]

            for col in required_columns:
                assert col in df.columns, f"Missing column: {col}"
