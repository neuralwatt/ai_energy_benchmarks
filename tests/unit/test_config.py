"""Unit tests for configuration parser."""

import os
import tempfile

import pytest

from ai_energy_benchmarks.config.parser import BenchmarkConfig, ConfigParser


class TestConfigParser:
    """Test configuration parser."""

    def test_load_config_from_file(self):
        """Test loading config from YAML file."""
        config_content = """
name: test_benchmark
backend:
  type: vllm
  model: test-model
  endpoint: http://localhost:8000/v1
scenario:
  dataset_name: test-dataset
  num_samples: 5
metrics:
  type: codecarbon
  enabled: true
reporter:
  type: csv
  output_file: ./results/test.csv
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(config_content)
            config_path = f.name

        try:
            config = ConfigParser.load_config(config_path)
            assert config.name == "test_benchmark"
            assert config.backend.type == "vllm"
            assert config.backend.model == "test-model"
            assert config.scenario.num_samples == 5
        finally:
            os.unlink(config_path)

    def test_load_config_missing_file(self):
        """Test loading non-existent config file."""
        with pytest.raises(FileNotFoundError):
            ConfigParser.load_config("nonexistent.yaml")

    def test_load_config_with_overrides(self):
        """Test loading config with overrides."""
        config_content = """
name: test_benchmark
backend:
  type: vllm
  model: test-model
scenario:
  num_samples: 5
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(config_content)
            config_path = f.name

        try:
            overrides = {"scenario": {"num_samples": 10}, "name": "overridden"}
            config = ConfigParser.load_config_with_overrides(config_path, overrides)
            assert config.name == "overridden"
            assert config.scenario.num_samples == 10
        finally:
            os.unlink(config_path)

    def test_validate_config_valid(self):
        """Test validation of valid config."""
        config = BenchmarkConfig()
        assert ConfigParser.validate_config(config) is True

    def test_validate_config_invalid_backend(self):
        """Test validation with invalid backend."""
        config = BenchmarkConfig()
        config.backend.type = "invalid"

        with pytest.raises(ValueError, match="Invalid backend type"):
            ConfigParser.validate_config(config)

    def test_validate_config_vllm_no_endpoint(self):
        """Test validation of vLLM backend without endpoint."""
        config = BenchmarkConfig()
        config.backend.type = "vllm"
        config.backend.endpoint = None

        with pytest.raises(ValueError, match="vLLM backend requires endpoint"):
            ConfigParser.validate_config(config)

    def test_validate_config_invalid_num_samples(self):
        """Test validation with invalid num_samples."""
        config = BenchmarkConfig()
        config.scenario.num_samples = 0

        with pytest.raises(ValueError, match="num_samples must be"):
            ConfigParser.validate_config(config)
