"""Integration tests for benchmark runner."""

import os
import tempfile
from typing import Any
from unittest.mock import Mock, patch

import pytest

from ai_energy_benchmarks.config.parser import BenchmarkConfig
from ai_energy_benchmarks.runner import BenchmarkRunner


class TestBenchmarkRunnerIntegration:
    """Integration tests for benchmark runner."""

    @pytest.fixture
    def mock_config(self):
        """Create a mock configuration."""
        config = BenchmarkConfig()
        config.name = "integration_test"
        config.backend.type = "vllm"
        config.backend.model = "test-model"
        config.backend.endpoint = "http://localhost:8000/v1"
        config.scenario.dataset_name = "test-dataset"
        config.scenario.num_samples = 2
        return config

    @patch("ai_energy_benchmarks.backends.vllm.requests.get")
    @patch("ai_energy_benchmarks.backends.vllm.requests.post")
    @patch("datasets.load_dataset")
    def test_full_benchmark_run(self, mock_load_dataset, mock_post, mock_get):
        """Test full benchmark execution flow."""
        # Mock vLLM responses
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = {"data": [{"id": "test-model"}]}

        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = {
            "choices": [{"message": {"content": "response"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
        }

        # Mock dataset
        mock_dataset = Mock()
        mock_dataset.column_names = ["text"]
        mock_dataset.__getitem__ = Mock(return_value=["prompt1", "prompt2"])
        mock_load_dataset.return_value = mock_dataset

        with tempfile.TemporaryDirectory() as tmpdir:
            config = BenchmarkConfig()
            config.name = "integration_test"
            config.backend.type = "vllm"
            config.backend.model = "test-model"
            config.backend.endpoint = "http://localhost:8000/v1"
            config.scenario.dataset_name = "test-dataset"
            config.scenario.num_samples = 2
            config.reporter.output_file = os.path.join(tmpdir, "results.csv")
            config.metrics.enabled = False  # Disable for integration test

            runner = BenchmarkRunner(config)
            results = runner.run()

            # Verify results
            assert results["summary"]["total_prompts"] == 2
            assert results["summary"]["successful_prompts"] == 2
            assert os.path.exists(config.reporter.output_file)

    def test_runner_initialization(self, mock_config):
        """Test runner initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_config.reporter.output_file = os.path.join(tmpdir, "results.csv")
            mock_config.metrics.enabled = False

            runner = BenchmarkRunner(mock_config)

            assert runner.config == mock_config
            assert runner.backend is not None
            assert runner.dataset is not None
            assert runner.reporter is not None

    @pytest.mark.parametrize(
        "device_ids,expected_count",
        [
            ([0], 1),
            ([0, 1], 2),
            ([0, 1, 2, 3], 4),
        ],
    )
    def test_multi_gpu_config_validation(self, device_ids, expected_count):
        """Test multi-GPU configuration validation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = BenchmarkConfig()
            config.name = "multi_gpu_test"
            config.backend.type = "pytorch"
            config.backend.model = "gpt2"
            config.backend.device = "cuda"
            config.backend.device_ids = device_ids
            config.scenario.dataset_name = "test-dataset"
            config.scenario.num_samples = 2
            config.reporter.output_file = os.path.join(tmpdir, "results.csv")
            config.metrics.enabled = False

            runner = BenchmarkRunner(config)

            # Verify device_ids configuration
            assert runner.config.backend.device_ids == device_ids
            assert len(runner.config.backend.device_ids) == expected_count

            # Verify backend info includes device configuration
            if runner.backend is not None:
                backend_info = runner.backend.get_endpoint_info()
                assert backend_info["device_ids"] == device_ids
                assert "device_map" in backend_info

    @patch("ai_energy_benchmarks.utils.gpu.GPUMonitor.check_gpu_available")
    def test_gpu_availability_check(self, mock_check_gpu):
        """Test GPU availability checking for multi-GPU setup."""
        # Simulate GPU 0 and 1 available, GPU 2 not available
        mock_check_gpu.side_effect = lambda gpu_id: gpu_id < 2

        with tempfile.TemporaryDirectory() as tmpdir:
            config = BenchmarkConfig()
            config.name = "gpu_check_test"
            config.backend.type = "pytorch"
            config.backend.model = "gpt2"
            config.backend.device = "cuda"
            config.backend.device_ids = [0, 1, 2]
            config.scenario.dataset_name = "test-dataset"
            config.scenario.num_samples = 1
            config.reporter.output_file = os.path.join(tmpdir, "results.csv")
            config.metrics.enabled = False

            from ai_energy_benchmarks.utils import GPUMonitor

            availability = GPUMonitor.check_multi_gpu_available([0, 1, 2])

            # Verify availability results
            assert availability["requested_gpus"] == [0, 1, 2]
            assert 0 in availability["available_gpus"]
            assert 1 in availability["available_gpus"]
            assert 2 in availability["unavailable_gpus"]
            assert not availability["all_available"]

    def test_multi_gpu_aggregate_results_structure(self):
        """Test that _aggregate_results properly structures multi-GPU stats."""
        import time

        from ai_energy_benchmarks.utils.gpu import GPUStats

        with tempfile.TemporaryDirectory() as tmpdir:
            config = BenchmarkConfig()
            config.name = "multi_gpu_structure_test"
            config.backend.type = "pytorch"
            config.backend.model = "gpt2"
            config.backend.device = "cuda"
            config.backend.device_ids = [0, 1]
            config.scenario.dataset_name = "test-dataset"
            config.scenario.num_samples = 1
            config.reporter.output_file = os.path.join(tmpdir, "results.csv")
            config.metrics.enabled = False

            runner = BenchmarkRunner(config)

            # Create mock GPU stats
            gpu_stats = {
                0: GPUStats(
                    gpu_id=0,
                    utilization_percent=85.0,
                    memory_used_mb=15000.0,
                    memory_total_mb=16384.0,
                    memory_percent=91.5,
                    temperature_c=72.0,
                    power_draw_w=250.0,
                    timestamp=time.time(),
                ),
                1: GPUStats(
                    gpu_id=1,
                    utilization_percent=78.0,
                    memory_used_mb=14000.0,
                    memory_total_mb=16384.0,
                    memory_percent=85.4,
                    temperature_c=70.0,
                    power_draw_w=240.0,
                    timestamp=time.time(),
                ),
            }

            # Test aggregate results structure
            inference_results = [
                {
                    "success": True,
                    "total_tokens": 100,
                    "prompt_tokens": 50,
                    "completion_tokens": 50,
                    "latency_seconds": 1.5,
                }
            ]
            energy_metrics: dict[str, Any] = {}

            results = runner._aggregate_results(inference_results, energy_metrics, 10.0, gpu_stats)

            # Verify structure
            assert "gpu_stats" in results
            assert "gpu_0" in results["gpu_stats"]
            assert "gpu_1" in results["gpu_stats"]

            # Verify GPU 0 stats
            assert results["gpu_stats"]["gpu_0"]["utilization_percent"] == 85.0
            assert results["gpu_stats"]["gpu_0"]["memory_used_mb"] == 15000.0
            assert results["gpu_stats"]["gpu_0"]["temperature_c"] == 72.0
            assert results["gpu_stats"]["gpu_0"]["power_draw_w"] == 250.0

            # Verify GPU 1 stats
            assert results["gpu_stats"]["gpu_1"]["utilization_percent"] == 78.0
            assert results["gpu_stats"]["gpu_1"]["memory_used_mb"] == 14000.0

            # Verify device_ids in config
            assert results["config"]["device_ids"] == [0, 1]
