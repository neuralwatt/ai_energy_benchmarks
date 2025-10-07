"""Integration tests for benchmark runner."""

import pytest
import tempfile
import os
from unittest.mock import Mock, patch
from ai_energy_benchmarks.config.parser import BenchmarkConfig, BackendConfig, ScenarioConfig
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

    @patch('ai_energy_benchmarks.backends.vllm.requests.get')
    @patch('ai_energy_benchmarks.backends.vllm.requests.post')
    @patch('ai_energy_benchmarks.datasets.huggingface.load_dataset')
    def test_full_benchmark_run(self, mock_load_dataset, mock_post, mock_get):
        """Test full benchmark execution flow."""
        # Mock vLLM responses
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = {
            'data': [{'id': 'test-model'}]
        }

        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = {
            'choices': [{'message': {'content': 'response'}}],
            'usage': {'prompt_tokens': 10, 'completion_tokens': 20, 'total_tokens': 30}
        }

        # Mock dataset
        mock_dataset = Mock()
        mock_dataset.column_names = ['text']
        mock_dataset.__getitem__ = Mock(return_value=['prompt1', 'prompt2'])
        mock_load_dataset.return_value = mock_dataset

        with tempfile.TemporaryDirectory() as tmpdir:
            config = BenchmarkConfig()
            config.name = "integration_test"
            config.backend.type = "vllm"
            config.backend.model = "test-model"
            config.backend.endpoint = "http://localhost:8000/v1"
            config.scenario.dataset_name = "test-dataset"
            config.scenario.num_samples = 2
            config.reporter.output_file = os.path.join(tmpdir, 'results.csv')
            config.metrics.enabled = False  # Disable for integration test

            runner = BenchmarkRunner(config)
            results = runner.run()

            # Verify results
            assert results['summary']['total_prompts'] == 2
            assert results['summary']['successful_prompts'] == 2
            assert os.path.exists(config.reporter.output_file)

    def test_runner_initialization(self, mock_config):
        """Test runner initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_config.reporter.output_file = os.path.join(tmpdir, 'results.csv')
            mock_config.metrics.enabled = False

            runner = BenchmarkRunner(mock_config)

            assert runner.config == mock_config
            assert runner.backend is not None
            assert runner.dataset is not None
            assert runner.reporter is not None
