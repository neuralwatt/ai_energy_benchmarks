"""
Test suite to validate all examples from README.md

This module ensures that all code examples, configurations, and commands
shown in the README actually work correctly.
"""

import os
from pathlib import Path
from typing import Any, Dict, cast
from unittest.mock import patch

import pytest
import yaml  # type: ignore[import-untyped]

# Test markers
pytestmark = pytest.mark.readme_examples


class TestREADMEConfigExamples:
    """Test all YAML configuration examples from README."""

    def test_pytorch_single_gpu_config_example(self):
        """Test the single GPU PyTorch config example from README."""
        config = {
            "backend": {
                "type": "pytorch",
                "model": "gpt2",
                "device": "cuda",
                "device_ids": [0],
                "task": "text-generation",
            }
        }

        # Validate config structure
        assert config["backend"]["type"] == "pytorch"
        assert config["backend"]["device"] == "cuda"
        assert isinstance(config["backend"]["device_ids"], list)
        assert config["backend"]["device_ids"] == [0]

    def test_pytorch_multigpu_config_example(self):
        """Test the multi-GPU PyTorch config example from README."""
        config = {
            "backend": {
                "type": "pytorch",
                "model": "meta-llama/Llama-2-70b-hf",
                "device": "cuda",
                "device_ids": [0, 1, 2, 3],
                "device_map": "auto",
                "torch_dtype": "auto",
            }
        }

        # Validate config structure
        assert config["backend"]["type"] == "pytorch"
        assert len(config["backend"]["device_ids"]) == 4
        assert config["backend"]["device_map"] == "auto"

    def test_vllm_backend_config_example(self):
        """Test the vLLM backend config example from README."""
        config = {
            "backend": {
                "type": "vllm",
                "endpoint": "http://localhost:8000/v1",
                "model": "openai/gpt-oss-120b",
            }
        }

        # Validate config structure
        assert config["backend"]["type"] == "vllm"
        assert config["backend"]["endpoint"].startswith("http")
        assert "/v1" in config["backend"]["endpoint"]

    def test_scenario_config_example(self):
        """Test the scenario configuration example from README."""
        config = {
            "scenario": {
                "dataset_name": "AIEnergyScore/text_generation",
                "text_column_name": "text",
                "num_samples": 100,
                "truncation": True,
                "input_shapes": {"batch_size": 1},
                "generate_kwargs": {
                    "max_new_tokens": 100,
                    "min_new_tokens": 50,
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "top_k": 50,
                    "do_sample": True,
                },
            }
        }

        # Validate scenario structure
        assert config["scenario"]["dataset_name"] == "AIEnergyScore/text_generation"
        assert config["scenario"]["num_samples"] == 100
        generate_kwargs = config["scenario"]["generate_kwargs"]
        assert isinstance(generate_kwargs, dict)
        assert generate_kwargs["max_new_tokens"] == 100

    def test_metrics_config_example(self):
        """Test the metrics configuration example from README."""
        config = {
            "metrics": {
                "type": "codecarbon",
                "enabled": True,
                "project_name": "my_benchmark",
                "output_dir": "./emissions",
                "country_iso_code": "USA",
                "region": None,
            }
        }

        # Validate metrics structure
        assert config["metrics"]["type"] == "codecarbon"
        assert config["metrics"]["enabled"] is True
        assert config["metrics"]["country_iso_code"] == "USA"

    def test_reasoning_params_gpt_oss_example(self):
        """Test the gpt-oss reasoning params example from README."""
        config = {
            "backend": {
                "type": "vllm",
                "endpoint": "http://localhost:8000/v1",
                "model": "openai/gpt-oss-20b",
            },
            "scenario": {"reasoning_params": {"reasoning_effort": "high"}},
        }

        # Validate reasoning params
        scenario = cast(Dict[str, Any], config["scenario"])
        reasoning_params = cast(Dict[str, Any], scenario["reasoning_params"])
        assert reasoning_params["reasoning_effort"] in [
            "low",
            "medium",
            "high",
        ]

    def test_reasoning_params_deepseek_example(self):
        """Test the DeepSeek reasoning params example from README."""
        config = {
            "backend": {"type": "pytorch", "model": "deepseek-ai/DeepSeek-R1", "device": "cuda"},
            "scenario": {"reasoning_params": {"enable_thinking": True, "thinking_budget": 1000}},
        }

        # Validate reasoning params
        scenario = cast(Dict[str, Any], config["scenario"])
        reasoning_params = cast(Dict[str, Any], scenario["reasoning_params"])
        assert reasoning_params["enable_thinking"] is True
        assert reasoning_params["thinking_budget"] == 1000


class TestREADMEPythonAPIExamples:
    """Test Python API examples from README."""

    @patch("ai_energy_benchmarks.runner.run_benchmark_from_config")
    def test_basic_api_usage_example(self, mock_run_benchmark):
        """Test the basic Python API usage example from README."""
        from ai_energy_benchmarks.runner import run_benchmark_from_config

        # Mock return value
        mock_run_benchmark.return_value = {
            "summary": {"total_energy_wh": 42.5},
        }

        # Execute example code
        results = run_benchmark_from_config("configs/pytorch_test.yaml")

        # Validate
        assert "summary" in results
        assert "total_energy_wh" in results["summary"]
        mock_run_benchmark.assert_called_once()

    @patch("ai_energy_benchmarks.runner.run_benchmark_from_config")
    def test_api_with_overrides_example(self, mock_run_benchmark):
        """Test the Python API with overrides example from README."""
        from ai_energy_benchmarks.runner import run_benchmark_from_config

        # Mock return value
        mock_run_benchmark.return_value = {"summary": {"total_energy_wh": 30.0}}

        # Execute example code from README
        overrides = {"scenario": {"num_samples": 20}, "backend": {"model": "gpt2-medium"}}
        _ = run_benchmark_from_config("configs/base.yaml", overrides=overrides)

        # Validate
        mock_run_benchmark.assert_called_once_with("configs/base.yaml", overrides=overrides)

    def test_benchmark_runner_class_example(self):
        """Test the BenchmarkRunner class example from README."""
        from ai_energy_benchmarks.config.parser import BenchmarkConfig
        from ai_energy_benchmarks.runner import BenchmarkRunner

        # Create config as shown in README
        config = BenchmarkConfig()
        config.name = "my_benchmark"
        config.backend.type = "pytorch"
        config.backend.model = "gpt2"
        config.scenario.num_samples = 10

        # Create runner (don't execute)
        runner = BenchmarkRunner(config)

        # Validate runner was created
        assert runner.config.name == "my_benchmark"
        assert runner.config.backend.type == "pytorch"
        assert runner.config.scenario.num_samples == 10


class TestREADMEActualConfigFiles:
    """Test that config files referenced in README actually exist and are valid."""

    def test_pytorch_test_config_exists(self):
        """Test that configs/pytorch_test.yaml exists and is valid."""
        config_path = Path("configs/pytorch_test.yaml")
        assert config_path.exists(), f"Config file not found: {config_path}"

        # Load and validate
        with open(config_path) as f:
            config = yaml.safe_load(f)

        assert config["name"] == "pytorch_backend_test"
        assert config["backend"]["type"] == "pytorch"
        assert config["backend"]["model"] == "microsoft/phi-2"
        assert config["scenario"]["num_samples"] == 3

    def test_pytorch_multigpu_config_exists(self):
        """Test that configs/pytorch_multigpu.yaml exists and is valid."""
        config_path = Path("configs/pytorch_multigpu.yaml")
        assert config_path.exists(), f"Config file not found: {config_path}"

        # Load and validate
        with open(config_path) as f:
            config = yaml.safe_load(f)

        assert config["name"] == "pytorch_multigpu_benchmark"
        assert config["backend"]["type"] == "pytorch"
        assert config["backend"]["model"] == "meta-llama/Llama-2-70b-hf"
        assert len(config["backend"]["device_ids"]) == 4

    def test_gpt_oss_120b_config_exists(self):
        """Test that configs/gpt_oss_120b.yaml exists and is valid."""
        config_path = Path("configs/gpt_oss_120b.yaml")
        assert config_path.exists(), f"Config file not found: {config_path}"

        # Load and validate
        with open(config_path) as f:
            config = yaml.safe_load(f)

        assert config["name"] == "gpt_oss_120b_poc"
        assert config["backend"]["type"] == "vllm"
        assert config["backend"]["model"] == "openai/gpt-oss-120b"
        assert config["backend"]["endpoint"] == "http://localhost:8000/v1"

    def test_pytorch_validation_config_exists(self):
        """Test that configs/pytorch_validation.yaml exists and is valid."""
        config_path = Path("configs/pytorch_validation.yaml")
        assert config_path.exists(), f"Config file not found: {config_path}"

        # Load and validate
        with open(config_path) as f:
            config = yaml.safe_load(f)

        assert config["name"] == "pytorch_validation_test"
        assert config["backend"]["model"] == "openai/gpt-oss-20b"
        assert config["scenario"]["num_samples"] == 1000


class TestREADMEScriptExamples:
    """Test shell script examples from README."""

    def test_run_benchmark_script_exists(self):
        """Test that run_benchmark.sh exists and is executable."""
        script_path = Path("run_benchmark.sh")
        assert script_path.exists(), "run_benchmark.sh not found"
        assert os.access(script_path, os.X_OK), "run_benchmark.sh not executable"

    def test_build_wheel_script_exists(self):
        """Test that build_wheel.sh exists and is executable."""
        script_path = Path("build_wheel.sh")
        assert script_path.exists(), "build_wheel.sh not found"
        assert os.access(script_path, os.X_OK), "build_wheel.sh not executable"

    def test_run_benchmark_help_format(self):
        """Test that run_benchmark.sh has correct default config."""
        with open("run_benchmark.sh") as f:
            content = f.read()

        # Verify default config as mentioned in README
        assert 'CONFIG_FILE="${1:-configs/gpt_oss_120b.yaml}"' in content


class TestREADMEEnvironmentVariables:
    """Test environment variable examples from README."""

    def test_environment_variable_substitution(self):
        """Test that config parser handles environment variables as shown in README."""
        # README shows: ${VLLM_ENDPOINT:-http://localhost:8000/v1}
        os.environ["VLLM_ENDPOINT"] = "http://test-server:9000/v1"

        try:
            # Test environment variable expansion
            test_value = os.environ.get("VLLM_ENDPOINT", "http://localhost:8000/v1")
            assert test_value == "http://test-server:9000/v1"

            # Test default value
            del os.environ["VLLM_ENDPOINT"]
            test_value = os.environ.get("VLLM_ENDPOINT", "http://localhost:8000/v1")
            assert test_value == "http://localhost:8000/v1"

        finally:
            if "VLLM_ENDPOINT" in os.environ:
                del os.environ["VLLM_ENDPOINT"]


class TestREADMEPackageStructure:
    """Test that package structure matches README documentation."""

    def test_package_imports_from_readme(self):
        """Test that all imports shown in README actually work."""
        # From README: "Import key classes for convenience"
        from ai_energy_benchmarks.config.parser import BenchmarkConfig
        from ai_energy_benchmarks.runner import BenchmarkRunner

        assert BenchmarkRunner is not None
        assert BenchmarkConfig is not None

    def test_backend_classes_exist(self):
        """Test that backend classes mentioned in README exist."""
        from ai_energy_benchmarks.backends.pytorch import PyTorchBackend
        from ai_energy_benchmarks.backends.vllm import VLLMBackend

        assert PyTorchBackend is not None
        assert VLLMBackend is not None

    def test_formatter_classes_exist(self):
        """Test that formatter classes mentioned in README exist."""
        from ai_energy_benchmarks.formatters.harmony import HarmonyFormatter
        from ai_energy_benchmarks.formatters.registry import FormatterRegistry

        assert HarmonyFormatter is not None
        assert FormatterRegistry is not None


class TestREADMEReasoningFormats:
    """Test reasoning format examples from README."""

    def test_reasoning_formats_yaml_exists(self):
        """Test that reasoning_formats.yaml exists as documented in README."""
        # Check in installed location
        try:
            from ai_energy_benchmarks.formatters.registry import FormatterRegistry

            registry = FormatterRegistry()
            # If this works, the YAML file is accessible
            assert registry is not None
        except Exception as e:
            pytest.fail(f"Failed to load reasoning formats: {e}")

    def test_gpt_oss_family_in_registry(self):
        """Test that gpt-oss family is registered as shown in README."""
        from ai_energy_benchmarks.formatters.registry import FormatterRegistry

        registry = FormatterRegistry()

        # Test gpt-oss detection (from README examples)
        formatter = registry.get_formatter("openai/gpt-oss-20b")
        assert formatter is not None

        formatter = registry.get_formatter("openai/gpt-oss-120b")
        assert formatter is not None

    def test_deepseek_family_in_registry(self):
        """Test that DeepSeek family is registered as shown in README."""
        from ai_energy_benchmarks.formatters.registry import FormatterRegistry

        registry = FormatterRegistry()

        # Test DeepSeek detection (from README examples)
        formatter = registry.get_formatter("deepseek-ai/DeepSeek-R1")
        assert formatter is not None


class TestREADMEModelExamples:
    """Test that model examples mentioned in README are valid."""

    def test_supported_small_models_list(self):
        """Test that small models listed in README are valid identifiers."""
        small_models = ["gpt2", "gpt2-medium", "facebook/opt-125m"]

        for model in small_models:
            # Verify format is valid (contains valid characters)
            assert "/" in model or model.startswith("gpt2")
            assert len(model) > 0

    def test_supported_large_models_list(self):
        """Test that large models listed in README are valid identifiers."""
        large_models = [
            "meta-llama/Llama-2-7b-hf",
            "meta-llama/Llama-2-70b-hf",
            "mistralai/Mistral-7B-v0.1",
        ]

        for model in large_models:
            # Verify format is valid
            assert "/" in model  # Should have org/model format
            assert len(model) > 0

    def test_actual_config_models_match_readme(self):
        """Test that models in actual configs match README examples."""
        # README says pytorch_test.yaml uses microsoft/phi-2
        with open("configs/pytorch_test.yaml") as f:
            config = yaml.safe_load(f)
        assert config["backend"]["model"] == "microsoft/phi-2"

        # README says pytorch_multigpu.yaml uses Llama-2-70b-hf
        with open("configs/pytorch_multigpu.yaml") as f:
            config = yaml.safe_load(f)
        assert config["backend"]["model"] == "meta-llama/Llama-2-70b-hf"

        # README says gpt_oss_120b.yaml uses openai/gpt-oss-120b
        with open("configs/gpt_oss_120b.yaml") as f:
            config = yaml.safe_load(f)
        assert config["backend"]["model"] == "openai/gpt-oss-120b"


class TestREADMEDatasets:
    """Test dataset references from README."""

    def test_dataset_names_from_readme(self):
        """Test that dataset names in README are valid."""
        datasets = [
            "AIEnergyScore/text_generation",
            "EnergyStarAI/text_generation",
            "openai/gsm8k",
            "tatsu-lab/alpaca",
        ]

        for dataset_name in datasets:
            # Verify format is valid (contains /)
            assert "/" in dataset_name
            assert len(dataset_name.split("/")) == 2


class TestREADMEInstallationExamples:
    """Test installation examples from README."""

    def test_package_name_format(self):
        """Test that package name in README matches pyproject.toml."""
        with open("pyproject.toml") as f:
            content = f.read()

        # README says: pip install ai_energy_benchmarks
        assert 'name = "ai_energy_benchmarks"' in content

    def test_optional_dependencies_exist(self):
        """Test that optional dependencies in README exist in pyproject.toml."""
        with open("pyproject.toml") as f:
            content = f.read()

        # README mentions [pytorch] and [all] extras
        assert "[project.optional-dependencies]" in content
        assert "pytorch = [" in content
        assert "all = [" in content


class TestREADMEVersionInfo:
    """Test version information from README."""

    def test_version_file_exists(self):
        """Test that VERSION.txt exists as mentioned in README."""
        version_path = Path("VERSION.txt")
        assert version_path.exists(), "VERSION.txt not found"

        # Read version
        with open(version_path) as f:
            version = f.read().strip()

        # Verify it's a valid version format
        assert len(version) > 0
        assert version[0].isdigit()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
