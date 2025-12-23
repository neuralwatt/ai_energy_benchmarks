"""
Interface compatibility tests for AIEnergyScore.

This test module documents and verifies the exact interface that AIEnergyScore
depends on from ai_energy_benchmarks. Any changes that break these tests would
break AIEnergyScore.

The interface is used by:
- AIEnergyScore/model_config_parser.py
- AIEnergyScore/run_ai_energy_benchmark.py
- AIEnergyScore/batch_runner.py

Key interface points:
1. FormatterRegistry and _find_model_config
2. HarmonyFormatter
3. Config dataclasses (BenchmarkConfig, BackendConfig, ScenarioConfig, MetricsConfig, ReporterConfig)
4. BenchmarkRunner class
"""

import os
import tempfile
from typing import Any, Dict
from unittest.mock import Mock, patch

import pytest


class TestFormatterRegistryInterface:
    """Test FormatterRegistry interface as used by AIEnergyScore/model_config_parser.py."""

    def test_import_path(self):
        """Verify the expected import paths work."""
        from ai_energy_benchmarks.formatters.harmony import HarmonyFormatter
        from ai_energy_benchmarks.formatters.registry import FormatterRegistry

        assert HarmonyFormatter is not None
        assert FormatterRegistry is not None

    def test_registry_instantiation(self):
        """FormatterRegistry can be instantiated without arguments."""
        from ai_energy_benchmarks.formatters.registry import FormatterRegistry

        registry = FormatterRegistry()
        assert registry is not None

    def test_get_formatter_returns_formatter_or_none(self):
        """get_formatter returns a ReasoningFormatter or None."""
        from ai_energy_benchmarks.formatters.registry import FormatterRegistry

        registry = FormatterRegistry()

        # Known model should return a formatter
        formatter = registry.get_formatter("openai/gpt-oss-20b")
        assert formatter is not None

        # Unknown model should return None (not raise)
        formatter = registry.get_formatter("unknown/model-xyz")
        assert formatter is None

    def test_find_model_config_method_exists(self):
        """_find_model_config method exists (used by AIEnergyScore for registry check)."""
        from ai_energy_benchmarks.formatters.registry import FormatterRegistry

        registry = FormatterRegistry()

        # Method should exist and be callable
        assert hasattr(registry, "_find_model_config")
        assert callable(registry._find_model_config)

        # Should return dict for known model
        config = registry._find_model_config("openai/gpt-oss-20b")
        assert config is not None
        assert isinstance(config, dict)

        # Should return None for unknown model
        config = registry._find_model_config("unknown/model-xyz")
        assert config is None

    def test_find_model_config_for_null_formatter_models(self):
        """_find_model_config returns config for models with null formatter (e.g., Qwen).

        AIEnergyScore uses this to distinguish between:
        - Model not in registry (truly unknown)
        - Model in registry with null formatter (uses chat template)
        """
        from ai_energy_benchmarks.formatters.registry import FormatterRegistry

        registry = FormatterRegistry()

        # Qwen models have type: null in config (use chat template)
        qwen_config = registry._find_model_config("Qwen/Qwen-2.5")
        assert qwen_config is not None  # Model IS in registry
        # But get_formatter returns None (no custom formatter needed)
        assert registry.get_formatter("Qwen/Qwen-2.5") is None


class TestHarmonyFormatterInterface:
    """Test HarmonyFormatter interface as used by AIEnergyScore/model_config_parser.py."""

    def test_isinstance_check(self):
        """AIEnergyScore uses isinstance() to check formatter type."""
        from ai_energy_benchmarks.formatters.harmony import HarmonyFormatter
        from ai_energy_benchmarks.formatters.registry import FormatterRegistry

        registry = FormatterRegistry()
        formatter = registry.get_formatter("openai/gpt-oss-20b")

        # This exact check is used in model_config_parser.py
        assert isinstance(formatter, HarmonyFormatter)

    def test_format_prompt_with_reasoning_params(self):
        """format_prompt accepts reasoning_params dict."""
        from ai_energy_benchmarks.formatters.harmony import HarmonyFormatter

        formatter = HarmonyFormatter()

        # AIEnergyScore passes reasoning_params with reasoning_effort
        result = formatter.format_prompt(
            "Test prompt", reasoning_params={"reasoning_effort": "high"}
        )

        assert isinstance(result, str)
        assert "Test prompt" in result

    def test_format_prompt_with_various_effort_levels(self):
        """format_prompt handles different reasoning_effort values."""
        from ai_energy_benchmarks.formatters.harmony import HarmonyFormatter

        formatter = HarmonyFormatter()

        for effort in ["high", "medium", "low"]:
            result = formatter.format_prompt(
                "Test prompt", reasoning_params={"reasoning_effort": effort}
            )
            assert isinstance(result, str)
            assert effort in result.lower()  # Effort level should appear in output


class TestConfigDataclassesInterface:
    """Test config dataclass interface as used by AIEnergyScore.

    AIEnergyScore creates config dataclasses directly (not via ConfigParser),
    so these tests verify direct instantiation patterns work.
    """

    def test_import_paths(self):
        """Verify the expected import paths work."""
        from ai_energy_benchmarks.config.parser import (
            BackendConfig,
            BenchmarkConfig,
            MetricsConfig,
            ReporterConfig,
            ScenarioConfig,
        )

        assert BenchmarkConfig is not None
        assert BackendConfig is not None
        assert ScenarioConfig is not None
        assert MetricsConfig is not None
        assert ReporterConfig is not None

    def test_backend_config_pytorch_instantiation(self):
        """BackendConfig can be created for PyTorch backend as in run_ai_energy_benchmark.py."""
        from ai_energy_benchmarks.config.parser import BackendConfig

        # Pattern from run_ai_energy_benchmark.py run_pytorch_backend()
        backend_cfg = BackendConfig(
            type="pytorch",
            model="test-model",
            device="cuda",
            device_ids=[0],
        )

        assert backend_cfg.type == "pytorch"
        assert backend_cfg.model == "test-model"
        assert backend_cfg.device == "cuda"
        assert backend_cfg.device_ids == [0]

    def test_backend_config_vllm_instantiation(self):
        """BackendConfig can be created for vLLM backend as in run_ai_energy_benchmark.py."""
        from ai_energy_benchmarks.config.parser import BackendConfig

        # Pattern from run_ai_energy_benchmark.py run_vllm_backend()
        backend_cfg = BackendConfig(
            type="vllm",
            model="test-model",
            endpoint="http://localhost:8000/v1",
        )

        assert backend_cfg.type == "vllm"
        assert backend_cfg.model == "test-model"
        assert backend_cfg.endpoint == "http://localhost:8000/v1"

    def test_scenario_config_with_reasoning(self):
        """ScenarioConfig can be created with reasoning parameters.

        Pattern from run_ai_energy_benchmark.py and batch_runner.py.
        """
        from ai_energy_benchmarks.config.parser import ScenarioConfig

        scenario_cfg = ScenarioConfig(
            dataset_name="EnergyStarAI/text_generation",
            text_column_name="text",
            num_samples=1000,
            reasoning=True,
            reasoning_params={"reasoning_effort": "high"},
            generate_kwargs={"max_new_tokens": 100, "min_new_tokens": 50},
        )

        assert scenario_cfg.dataset_name == "EnergyStarAI/text_generation"
        assert scenario_cfg.text_column_name == "text"
        assert scenario_cfg.num_samples == 1000
        assert scenario_cfg.reasoning is True
        assert scenario_cfg.reasoning_params == {"reasoning_effort": "high"}
        assert scenario_cfg.generate_kwargs["max_new_tokens"] == 100

    def test_scenario_config_without_reasoning(self):
        """ScenarioConfig can be created without reasoning (default case)."""
        from ai_energy_benchmarks.config.parser import ScenarioConfig

        scenario_cfg = ScenarioConfig(
            dataset_name="EnergyStarAI/text_generation",
            text_column_name="text",
            num_samples=10,
            reasoning=False,
            reasoning_params=None,
            generate_kwargs={"max_new_tokens": 10, "min_new_tokens": 10},
        )

        assert scenario_cfg.reasoning is False
        assert scenario_cfg.reasoning_params is None

    def test_metrics_config_instantiation(self):
        """MetricsConfig can be created as in run_ai_energy_benchmark.py."""
        from ai_energy_benchmarks.config.parser import MetricsConfig

        metrics_cfg = MetricsConfig(
            enabled=True,
            type="codecarbon",
            project_name="ai_energy_benchmark",
            output_dir="/results/emissions",
            country_iso_code="USA",
            region="california",
        )

        assert metrics_cfg.enabled is True
        assert metrics_cfg.type == "codecarbon"
        assert metrics_cfg.project_name == "ai_energy_benchmark"
        assert metrics_cfg.output_dir == "/results/emissions"
        assert metrics_cfg.country_iso_code == "USA"
        assert metrics_cfg.region == "california"

    def test_reporter_config_instantiation(self):
        """ReporterConfig can be created as in run_ai_energy_benchmark.py."""
        from ai_energy_benchmarks.config.parser import ReporterConfig

        reporter_cfg = ReporterConfig(
            type="csv",
            output_file="/results/benchmark_results.csv",
        )

        assert reporter_cfg.type == "csv"
        assert reporter_cfg.output_file == "/results/benchmark_results.csv"

    def test_benchmark_config_full_instantiation(self):
        """BenchmarkConfig can be created with all sub-configs.

        Pattern from run_ai_energy_benchmark.py run_pytorch_backend().
        """
        from ai_energy_benchmarks.config.parser import (
            BackendConfig,
            BenchmarkConfig,
            MetricsConfig,
            ReporterConfig,
            ScenarioConfig,
        )

        backend_cfg = BackendConfig(
            type="pytorch",
            model="test-model",
            device="cuda",
            device_ids=[0],
        )

        scenario_cfg = ScenarioConfig(
            dataset_name="EnergyStarAI/text_generation",
            text_column_name="text",
            num_samples=10,
            reasoning=False,
            reasoning_params=None,
            generate_kwargs={"max_new_tokens": 10, "min_new_tokens": 10},
        )

        metrics_cfg = MetricsConfig(
            enabled=True,
            type="codecarbon",
            project_name="test_project",
            output_dir="/results/emissions",
            country_iso_code="USA",
            region="california",
        )

        reporter_cfg = ReporterConfig(
            type="csv",
            output_file="/results/benchmark_results.csv",
        )

        bench_config = BenchmarkConfig(
            name="ai_energy_benchmark",
            backend=backend_cfg,
            scenario=scenario_cfg,
            metrics=metrics_cfg,
            reporter=reporter_cfg,
            output_dir="/results",
        )

        assert bench_config.name == "ai_energy_benchmark"
        assert bench_config.backend == backend_cfg
        assert bench_config.scenario == scenario_cfg
        assert bench_config.metrics == metrics_cfg
        assert bench_config.reporter == reporter_cfg
        assert bench_config.output_dir == "/results"

    def test_backend_config_multi_gpu(self):
        """BackendConfig supports multiple device_ids for multi-GPU."""
        from ai_energy_benchmarks.config.parser import BackendConfig

        backend_cfg = BackendConfig(
            type="pytorch",
            model="large-model",
            device="cuda",
            device_ids=[0, 1, 2, 3],
        )

        assert backend_cfg.device_ids == [0, 1, 2, 3]


class TestBenchmarkRunnerInterface:
    """Test BenchmarkRunner interface as used by AIEnergyScore.

    AIEnergyScore uses:
    - BenchmarkRunner(config) constructor
    - runner.validate() method
    - runner.run() method
    - runner.backend attribute
    """

    def test_import_path(self):
        """Verify the expected import path works."""
        from ai_energy_benchmarks.runner import BenchmarkRunner

        assert BenchmarkRunner is not None

    def test_runner_accepts_benchmark_config(self):
        """BenchmarkRunner constructor accepts BenchmarkConfig."""
        from ai_energy_benchmarks.config.parser import BenchmarkConfig
        from ai_energy_benchmarks.runner import BenchmarkRunner

        with tempfile.TemporaryDirectory() as tmpdir:
            config = BenchmarkConfig()
            config.name = "test_benchmark"
            config.backend.type = "vllm"
            config.backend.model = "test-model"
            config.backend.endpoint = "http://localhost:8000/v1"
            config.scenario.dataset_name = "test-dataset"
            config.scenario.num_samples = 5
            config.reporter.output_file = os.path.join(tmpdir, "results.csv")
            config.metrics.enabled = False

            runner = BenchmarkRunner(config)

            assert runner.config == config

    def test_runner_has_validate_method(self):
        """BenchmarkRunner has validate() method returning bool."""
        from ai_energy_benchmarks.config.parser import BenchmarkConfig
        from ai_energy_benchmarks.runner import BenchmarkRunner

        with tempfile.TemporaryDirectory() as tmpdir:
            config = BenchmarkConfig()
            config.backend.type = "vllm"
            config.backend.model = "test-model"
            config.backend.endpoint = "http://localhost:8000/v1"
            config.reporter.output_file = os.path.join(tmpdir, "results.csv")
            config.metrics.enabled = False

            runner = BenchmarkRunner(config)

            # validate() should exist and return bool
            assert hasattr(runner, "validate")
            assert callable(runner.validate)
            # Note: actual validation may fail without real server, that's OK
            result = runner.validate()
            assert isinstance(result, bool)

    def test_runner_has_run_method(self):
        """BenchmarkRunner has run() method."""
        from ai_energy_benchmarks.runner import BenchmarkRunner

        assert hasattr(BenchmarkRunner, "run")

    def test_runner_has_backend_attribute(self):
        """BenchmarkRunner has backend attribute (used for cleanup in batch_runner.py)."""
        from ai_energy_benchmarks.config.parser import BenchmarkConfig
        from ai_energy_benchmarks.runner import BenchmarkRunner

        with tempfile.TemporaryDirectory() as tmpdir:
            config = BenchmarkConfig()
            config.backend.type = "vllm"
            config.backend.model = "test-model"
            config.backend.endpoint = "http://localhost:8000/v1"
            config.reporter.output_file = os.path.join(tmpdir, "results.csv")
            config.metrics.enabled = False

            runner = BenchmarkRunner(config)

            # batch_runner.py accesses runner.backend for cleanup
            assert hasattr(runner, "backend")
            assert runner.backend is not None

    @patch("ai_energy_benchmarks.backends.vllm.requests.get")
    @patch("ai_energy_benchmarks.backends.vllm.requests.post")
    @patch("datasets.load_dataset")
    def test_run_returns_expected_structure(self, mock_load_dataset, mock_post, mock_get):
        """runner.run() returns dict with expected keys for AIEnergyScore.

        AIEnergyScore expects:
        - results["summary"]["successful_prompts"]
        - results["summary"]["total_prompts"]
        - results["energy"]["gpu_energy_wh"] (optional)
        """
        from ai_energy_benchmarks.config.parser import BenchmarkConfig
        from ai_energy_benchmarks.runner import BenchmarkRunner

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
        mock_dataset.__getitem__ = Mock(return_value=["prompt1"])
        mock_load_dataset.return_value = mock_dataset

        with tempfile.TemporaryDirectory() as tmpdir:
            config = BenchmarkConfig()
            config.name = "test"
            config.backend.type = "vllm"
            config.backend.model = "test-model"
            config.backend.endpoint = "http://localhost:8000/v1"
            config.scenario.dataset_name = "test-dataset"
            config.scenario.num_samples = 1
            config.reporter.output_file = os.path.join(tmpdir, "results.csv")
            config.metrics.enabled = False

            runner = BenchmarkRunner(config)
            results = runner.run()

            # Verify structure expected by AIEnergyScore
            assert "summary" in results
            assert "successful_prompts" in results["summary"]
            assert "total_prompts" in results["summary"]
            assert isinstance(results["summary"]["successful_prompts"], int)
            assert isinstance(results["summary"]["total_prompts"], int)

            # Energy key should exist (may be empty if metrics disabled)
            assert "energy" in results


class TestResultsStructure:
    """Test that benchmark results match structure expected by AIEnergyScore.

    See AIEnergyScore/summarize_gpu_wh.py for expected format.
    """

    def test_results_format_detection(self):
        """Results should be detectable as ai_energy_benchmarks format.

        AIEnergyScore/summarize_gpu_wh.py:
        - Looks for 'energy' key at top level
        - Checks for 'gpu_energy_wh' within energy
        """
        # Simulate expected results structure
        results: Dict[str, Any] = {
            "summary": {
                "total_prompts": 10,
                "successful_prompts": 10,
            },
            "energy": {
                "gpu_energy_wh": 0.5,
                "cpu_energy_wh": 0.1,
                "energy_wh": 0.6,
            },
        }

        # This is the detection logic from summarize_gpu_wh.py
        def detect_format(rep: dict) -> str:
            if "energy" in rep and isinstance(rep.get("energy"), dict):
                if "gpu_energy_wh" in rep["energy"]:
                    return "ai_energy_benchmarks"
            return "unknown"

        assert detect_format(results) == "ai_energy_benchmarks"


class TestConfigParserInterface:
    """Test ConfigParser interface for completeness.

    While AIEnergyScore mostly creates configs directly, some paths
    use ConfigParser methods.
    """

    def test_validate_config_method(self):
        """ConfigParser.validate_config exists and accepts BenchmarkConfig."""
        from ai_energy_benchmarks.config.parser import (
            BenchmarkConfig,
            ConfigParser,
        )

        config = BenchmarkConfig()

        # Method should exist
        assert hasattr(ConfigParser, "validate_config")

        # Should return True for valid config
        result = ConfigParser.validate_config(config)
        assert result is True

    def test_validate_config_raises_on_invalid(self):
        """ConfigParser.validate_config raises ValueError for invalid config."""
        from ai_energy_benchmarks.config.parser import (
            BenchmarkConfig,
            ConfigParser,
        )

        config = BenchmarkConfig()
        config.backend.type = "invalid_backend"

        with pytest.raises(ValueError, match="Invalid backend type"):
            ConfigParser.validate_config(config)


class TestReasoningParametersSupport:
    """Test that reasoning parameter patterns used by AIEnergyScore work."""

    def test_scenario_enable_thinking_true(self):
        """ScenarioConfig handles enable_thinking=True pattern."""
        from ai_energy_benchmarks.config.parser import ScenarioConfig

        # Pattern for Qwen, DeepSeek, EXAONE models
        cfg = ScenarioConfig(
            dataset_name="test",
            reasoning=True,
            reasoning_params={"enable_thinking": True},
        )

        assert cfg.reasoning_params is not None
        assert cfg.reasoning_params["enable_thinking"] is True

    def test_scenario_enable_thinking_false(self):
        """ScenarioConfig handles enable_thinking=False pattern."""
        from ai_energy_benchmarks.config.parser import ScenarioConfig

        cfg = ScenarioConfig(
            dataset_name="test",
            reasoning=True,
            reasoning_params={"enable_thinking": False},
        )

        assert cfg.reasoning_params is not None
        assert cfg.reasoning_params["enable_thinking"] is False

    def test_scenario_reasoning_effort(self):
        """ScenarioConfig handles reasoning_effort pattern (gpt-oss)."""
        from ai_energy_benchmarks.config.parser import ScenarioConfig

        for effort in ["high", "medium", "low"]:
            cfg = ScenarioConfig(
                dataset_name="test",
                reasoning=True,
                reasoning_params={"reasoning_effort": effort},
            )
            assert cfg.reasoning_params is not None
            assert cfg.reasoning_params["reasoning_effort"] == effort

    def test_scenario_thinking_budget(self):
        """ScenarioConfig handles thinking_budget parameter."""
        from ai_energy_benchmarks.config.parser import ScenarioConfig

        cfg = ScenarioConfig(
            dataset_name="test",
            reasoning=True,
            reasoning_params={"enable_thinking": True, "thinking_budget": 1000},
        )

        assert cfg.reasoning_params is not None
        assert cfg.reasoning_params["thinking_budget"] == 1000

    def test_scenario_generic_reasoning_true(self):
        """ScenarioConfig handles generic reasoning=True pattern (Phi, Gemma)."""
        from ai_energy_benchmarks.config.parser import ScenarioConfig

        cfg = ScenarioConfig(
            dataset_name="test",
            reasoning=True,
            reasoning_params={"reasoning": True},
        )

        assert cfg.reasoning_params is not None
        assert cfg.reasoning_params["reasoning"] is True


class TestFormatterPatternMatching:
    """Test that formatter pattern matching works for AIEnergyScore models."""

    @pytest.mark.parametrize(
        "model_id,expected_type",
        [
            ("openai/gpt-oss-20b", "HarmonyFormatter"),
            ("openai/gpt-oss-120b", "HarmonyFormatter"),
            ("HuggingFaceTB/SmolLM3-3B", "SystemPromptFormatter"),
            ("deepseek-ai/DeepSeek-R1", "PrefixFormatter"),
            ("deepseek-ai/DeepSeek-V3", "PrefixFormatter"),
            ("nvidia/nemotron-340b", "SystemPromptFormatter"),
            ("tencent/hunyuan-large", "SystemPromptFormatter"),
            ("microsoft/Phi-4-reasoning-plus", "ParameterFormatter"),
            ("google/gemma-2-9b", "ParameterFormatter"),
        ],
    )
    def test_model_formatter_mapping(self, model_id, expected_type):
        """Known models get expected formatter types."""
        from ai_energy_benchmarks.formatters.harmony import HarmonyFormatter
        from ai_energy_benchmarks.formatters.parameter import ParameterFormatter
        from ai_energy_benchmarks.formatters.prefix import PrefixFormatter
        from ai_energy_benchmarks.formatters.registry import FormatterRegistry
        from ai_energy_benchmarks.formatters.system_prompt import SystemPromptFormatter

        type_map = {
            "HarmonyFormatter": HarmonyFormatter,
            "SystemPromptFormatter": SystemPromptFormatter,
            "ParameterFormatter": ParameterFormatter,
            "PrefixFormatter": PrefixFormatter,
        }

        registry = FormatterRegistry()
        formatter = registry.get_formatter(model_id)

        assert isinstance(formatter, type_map[expected_type])

    @pytest.mark.parametrize(
        "model_id",
        [
            "Qwen/Qwen-2.5",
            "Qwen/Qwen3-235B-A22B",
            "Qwen/QwQ-32B",
        ],
    )
    def test_qwen_models_return_none_formatter(self, model_id):
        """Qwen models should return None (they use chat template with enable_thinking)."""
        from ai_energy_benchmarks.formatters.registry import FormatterRegistry

        registry = FormatterRegistry()
        formatter = registry.get_formatter(model_id)

        # Qwen uses chat template, not a formatter
        assert formatter is None

        # But model should be in registry
        config = registry._find_model_config(model_id)
        assert config is not None


class TestGenerateKwargsSupport:
    """Test generate_kwargs handling as used by AIEnergyScore."""

    def test_scenario_max_new_tokens(self):
        """ScenarioConfig.generate_kwargs supports max_new_tokens."""
        from ai_energy_benchmarks.config.parser import ScenarioConfig

        # Pattern from run_ai_energy_benchmark.py
        cfg = ScenarioConfig(
            dataset_name="test",
            generate_kwargs={"max_new_tokens": 8192, "min_new_tokens": 1},
        )

        assert cfg.generate_kwargs["max_new_tokens"] == 8192
        assert cfg.generate_kwargs["min_new_tokens"] == 1

    def test_scenario_fixed_tokens(self):
        """ScenarioConfig supports fixed token generation (non-reasoning mode)."""
        from ai_energy_benchmarks.config.parser import ScenarioConfig

        # Pattern from batch_runner.py for non-reasoning mode
        cfg = ScenarioConfig(
            dataset_name="test",
            reasoning=False,
            generate_kwargs={"max_new_tokens": 10, "min_new_tokens": 10},
        )

        assert cfg.generate_kwargs["max_new_tokens"] == 10
        assert cfg.generate_kwargs["min_new_tokens"] == 10


class TestPackageLevelImports:
    """Test package-level imports as used by AIEnergyScore.

    AIEnergyScore tests check for ai_energy_benchmarks availability with:
    - import ai_energy_benchmarks
    - importlib.util.find_spec("ai_energy_benchmarks")
    """

    def test_simple_package_import(self):
        """Test simple 'import ai_energy_benchmarks' works."""
        import importlib

        ai_energy_benchmarks = importlib.import_module("ai_energy_benchmarks")
        assert ai_energy_benchmarks is not None

    def test_package_has_version(self):
        """Test package has __version__ attribute."""
        import importlib

        ai_energy_benchmarks = importlib.import_module("ai_energy_benchmarks")
        assert hasattr(ai_energy_benchmarks, "__version__")
        assert isinstance(ai_energy_benchmarks.__version__, str)

    def test_package_exports_benchmark_runner(self):
        """Test BenchmarkRunner can be imported from package root."""
        from ai_energy_benchmarks import BenchmarkRunner

        assert BenchmarkRunner is not None

    def test_package_exports_benchmark_config(self):
        """Test BenchmarkConfig can be imported from package root."""
        from ai_energy_benchmarks import BenchmarkConfig

        assert BenchmarkConfig is not None

    def test_findspec_works(self):
        """Test importlib.util.find_spec works for the package."""
        import importlib.util

        spec = importlib.util.find_spec("ai_energy_benchmarks")
        assert spec is not None


class TestDictStyleAccess:
    """Test dictionary-style access patterns used by AIEnergyScore.

    AIEnergyScore accesses config values using dict-style notation, e.g.:
    - benchmark_config.scenario.generate_kwargs["max_new_tokens"]
    """

    def test_generate_kwargs_dict_access(self):
        """Test generate_kwargs supports dict-style access."""
        from ai_energy_benchmarks.config.parser import ScenarioConfig

        cfg = ScenarioConfig(
            dataset_name="test",
            generate_kwargs={"max_new_tokens": 100, "min_new_tokens": 50},
        )

        # Dict-style access should work
        assert cfg.generate_kwargs["max_new_tokens"] == 100
        assert cfg.generate_kwargs["min_new_tokens"] == 50

    def test_generate_kwargs_dict_modification(self):
        """Test generate_kwargs can be modified after creation."""
        from ai_energy_benchmarks.config.parser import ScenarioConfig

        cfg = ScenarioConfig(
            dataset_name="test",
            generate_kwargs={"max_new_tokens": 100},
        )

        # Should be able to modify
        cfg.generate_kwargs["min_new_tokens"] = 50
        assert cfg.generate_kwargs["min_new_tokens"] == 50

    def test_reasoning_params_dict_access(self):
        """Test reasoning_params supports dict-style access."""
        from ai_energy_benchmarks.config.parser import ScenarioConfig

        cfg = ScenarioConfig(
            dataset_name="test",
            reasoning_params={"enable_thinking": True, "thinking_budget": 1000},
        )

        assert cfg.reasoning_params is not None
        assert cfg.reasoning_params["enable_thinking"] is True
        assert cfg.reasoning_params["thinking_budget"] == 1000

    def test_reasoning_params_get_method(self):
        """Test reasoning_params.get() works (used in AIEnergyScore)."""
        from ai_energy_benchmarks.config.parser import ScenarioConfig

        cfg = ScenarioConfig(
            dataset_name="test",
            reasoning_params={"reasoning_effort": "high"},
        )

        # .get() should work
        assert cfg.reasoning_params is not None
        assert cfg.reasoning_params.get("reasoning_effort") == "high"
        assert cfg.reasoning_params.get("nonexistent") is None
        assert cfg.reasoning_params.get("nonexistent", "default") == "default"


class TestBackendAttributeAccess:
    """Test backend attribute access patterns used by AIEnergyScore.

    batch_runner.py accesses:
    - runner.backend
    - runner.backend.model
    - runner.backend.tokenizer
    """

    def test_runner_backend_has_model_attribute(self):
        """Test runner.backend has model attribute for cleanup."""
        from ai_energy_benchmarks.config.parser import BenchmarkConfig
        from ai_energy_benchmarks.runner import BenchmarkRunner

        with tempfile.TemporaryDirectory() as tmpdir:
            config = BenchmarkConfig()
            config.backend.type = "vllm"
            config.backend.model = "test-model"
            config.backend.endpoint = "http://localhost:8000/v1"
            config.reporter.output_file = os.path.join(tmpdir, "results.csv")
            config.metrics.enabled = False

            runner = BenchmarkRunner(config)

            # Backend should have model attribute (may be None initially)
            assert hasattr(runner.backend, "model")

    def test_pytorch_backend_has_expected_attributes(self):
        """Test PyTorch backend has expected attributes."""
        from ai_energy_benchmarks.backends.pytorch import PyTorchBackend

        backend = PyTorchBackend(
            model="gpt2",
            device="cuda",
            device_ids=[0],
        )

        # Should have these attributes (used by batch_runner.py for cleanup)
        assert hasattr(backend, "model")
        assert hasattr(backend, "tokenizer")
        assert hasattr(backend, "get_endpoint_info")


class TestBackendInfo:
    """Test backend info methods used by runner."""

    def test_vllm_get_endpoint_info(self):
        """Test VLLMBackend.get_endpoint_info returns dict."""
        from ai_energy_benchmarks.backends.vllm import VLLMBackend

        backend = VLLMBackend(
            endpoint="http://localhost:8000/v1",
            model="test-model",
        )

        info = backend.get_endpoint_info()

        assert isinstance(info, dict)
        assert "model" in info
        assert "endpoint" in info


class TestScenarioConfigDefaults:
    """Test that ScenarioConfig defaults match what AIEnergyScore expects."""

    def test_default_generate_kwargs(self):
        """Test default generate_kwargs are set."""
        from ai_energy_benchmarks.config.parser import ScenarioConfig

        cfg = ScenarioConfig(dataset_name="test")

        # Should have defaults
        assert cfg.generate_kwargs is not None
        assert isinstance(cfg.generate_kwargs, dict)
        assert "max_new_tokens" in cfg.generate_kwargs
        assert "min_new_tokens" in cfg.generate_kwargs

    def test_default_reasoning_is_false(self):
        """Test reasoning defaults to False."""
        from ai_energy_benchmarks.config.parser import ScenarioConfig

        cfg = ScenarioConfig(dataset_name="test")
        assert cfg.reasoning is False

    def test_default_reasoning_params_is_none(self):
        """Test reasoning_params defaults to None."""
        from ai_energy_benchmarks.config.parser import ScenarioConfig

        cfg = ScenarioConfig(dataset_name="test")
        assert cfg.reasoning_params is None


class TestErrorHandling:
    """Test error handling for invalid configurations."""

    def test_invalid_backend_type_raises(self):
        """Test that invalid backend type raises ValueError."""
        from ai_energy_benchmarks.config.parser import BenchmarkConfig, ConfigParser

        config = BenchmarkConfig()
        config.backend.type = "invalid_type"

        with pytest.raises(ValueError, match="Invalid backend type"):
            ConfigParser.validate_config(config)

    def test_vllm_without_endpoint_raises(self):
        """Test that vLLM without endpoint raises ValueError."""
        from ai_energy_benchmarks.config.parser import BenchmarkConfig, ConfigParser

        config = BenchmarkConfig()
        config.backend.type = "vllm"
        config.backend.endpoint = None

        with pytest.raises(ValueError, match="vLLM backend requires endpoint"):
            ConfigParser.validate_config(config)

    def test_zero_samples_raises(self):
        """Test that zero samples raises ValueError."""
        from ai_energy_benchmarks.config.parser import BenchmarkConfig, ConfigParser

        config = BenchmarkConfig()
        config.scenario.num_samples = 0

        with pytest.raises(ValueError, match="num_samples must be"):
            ConfigParser.validate_config(config)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
