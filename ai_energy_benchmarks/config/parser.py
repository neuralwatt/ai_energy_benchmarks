"""Configuration parser for Hydra-based configs."""

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from omegaconf import DictConfig, OmegaConf


@dataclass
class BackendConfig:
    """Backend configuration."""

    type: str = "vllm"
    device: str = "cuda"
    device_ids: List[int] = field(default_factory=lambda: [0])
    model: str = "openai/gpt-oss-120b"
    endpoint: Optional[str] = "http://localhost:8000/v1"
    task: str = "text-generation"


@dataclass
class ScenarioConfig:
    """Scenario configuration."""

    dataset_name: str = "AIEnergyScore/text_generation"
    text_column_name: str = "text"
    num_samples: int = 10
    truncation: bool = True
    reasoning: bool = False
    reasoning_params: Optional[Dict[str, Any]] = None
    input_shapes: Dict[str, Any] = field(default_factory=lambda: {"batch_size": 1})
    generate_kwargs: Dict[str, Any] = field(
        default_factory=lambda: {"max_new_tokens": 100, "min_new_tokens": 50}
    )


@dataclass
class MetricsConfig:
    """Metrics configuration."""

    type: str = "codecarbon"
    enabled: bool = True
    project_name: str = "ai_energy_benchmarks"
    output_dir: str = "./emissions"
    country_iso_code: str = "USA"
    region: Optional[str] = None


@dataclass
class ReporterConfig:
    """Reporter configuration."""

    type: str = "csv"
    output_file: str = "./results/benchmark_results.csv"


@dataclass
class BenchmarkConfig:
    """Main benchmark configuration."""

    name: str = "benchmark"
    backend: BackendConfig = field(default_factory=BackendConfig)
    scenario: ScenarioConfig = field(default_factory=ScenarioConfig)
    metrics: MetricsConfig = field(default_factory=MetricsConfig)
    reporter: ReporterConfig = field(default_factory=ReporterConfig)
    output_dir: str = "./benchmark_output"


class ConfigParser:
    """Parse and validate Hydra-based benchmark configurations."""

    @staticmethod
    def load_config(config_path: str) -> BenchmarkConfig:
        """Load configuration from YAML file.

        Args:
            config_path: Path to configuration file

        Returns:
            BenchmarkConfig object
        """
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")

        # Load YAML with OmegaConf
        cfg = OmegaConf.load(config_path)
        if not isinstance(cfg, DictConfig):
            raise TypeError("Expected DictConfig from configuration file")

        # Create structured config
        config = ConfigParser._build_config(cfg)

        return config

    @staticmethod
    def load_config_with_overrides(
        config_path: str, overrides: Optional[Dict[str, Any]] = None
    ) -> BenchmarkConfig:
        """Load configuration with CLI-style overrides.

        Args:
            config_path: Path to configuration file
            overrides: Dictionary of override values

        Returns:
            BenchmarkConfig object
        """
        cfg = OmegaConf.load(config_path)

        if not isinstance(cfg, DictConfig):
            raise TypeError("Expected DictConfig from configuration file")

        if overrides:
            override_cfg = OmegaConf.create(overrides)
            cfg = OmegaConf.merge(cfg, override_cfg)
            if not isinstance(cfg, DictConfig):
                raise TypeError("Merged configuration must be a DictConfig")

        return ConfigParser._build_config(cfg)

    @staticmethod
    def _build_config(cfg: DictConfig) -> BenchmarkConfig:
        """Build BenchmarkConfig from OmegaConf config.

        Args:
            cfg: OmegaConf configuration

        Returns:
            BenchmarkConfig object
        """
        # Extract sections
        backend_cfg = BackendConfig(
            type=cfg.get("backend", {}).get("type", "vllm"),
            device=cfg.get("backend", {}).get("device", "cuda"),
            device_ids=cfg.get("backend", {}).get("device_ids", [0]),
            model=cfg.get("backend", {}).get("model", "openai/gpt-oss-120b"),
            endpoint=cfg.get("backend", {}).get("endpoint", "http://localhost:8000/v1"),
            task=cfg.get("backend", {}).get("task", "text-generation"),
        )

        scenario_cfg = ScenarioConfig(
            dataset_name=cfg.get("scenario", {}).get(
                "dataset_name", "AIEnergyScore/text_generation"
            ),
            text_column_name=cfg.get("scenario", {}).get("text_column_name", "text"),
            num_samples=cfg.get("scenario", {}).get("num_samples", 10),
            truncation=cfg.get("scenario", {}).get("truncation", True),
            reasoning=cfg.get("scenario", {}).get("reasoning", False),
            reasoning_params=cfg.get("scenario", {}).get("reasoning_params"),
            input_shapes=cfg.get("scenario", {}).get("input_shapes", {"batch_size": 1}),
            generate_kwargs=cfg.get("scenario", {}).get(
                "generate_kwargs", {"max_new_tokens": 100, "min_new_tokens": 50}
            ),
        )

        metrics_cfg = MetricsConfig(
            type=cfg.get("metrics", {}).get("type", "codecarbon"),
            enabled=cfg.get("metrics", {}).get("enabled", True),
            project_name=cfg.get("metrics", {}).get("project_name", "ai_energy_benchmarks"),
            output_dir=cfg.get("metrics", {}).get("output_dir", "./emissions"),
            country_iso_code=cfg.get("metrics", {}).get("country_iso_code", "USA"),
            region=cfg.get("metrics", {}).get("region"),
        )

        reporter_cfg = ReporterConfig(
            type=cfg.get("reporter", {}).get("type", "csv"),
            output_file=cfg.get("reporter", {}).get(
                "output_file", "./results/benchmark_results.csv"
            ),
        )

        return BenchmarkConfig(
            name=cfg.get("name", "benchmark"),
            backend=backend_cfg,
            scenario=scenario_cfg,
            metrics=metrics_cfg,
            reporter=reporter_cfg,
            output_dir=cfg.get("output_dir", "./benchmark_output"),
        )

    @staticmethod
    def validate_config(config: BenchmarkConfig) -> bool:
        """Validate configuration.

        Args:
            config: BenchmarkConfig to validate

        Returns:
            bool: True if valid

        Raises:
            ValueError: If configuration is invalid
        """
        # Validate backend
        if config.backend.type not in ["vllm", "pytorch"]:
            raise ValueError(f"Invalid backend type: {config.backend.type}")

        if config.backend.type == "vllm" and not config.backend.endpoint:
            raise ValueError("vLLM backend requires endpoint configuration")

        # Validate scenario
        if config.scenario.num_samples < 1:
            raise ValueError("num_samples must be >= 1")

        # Validate metrics
        if config.metrics.type not in ["codecarbon"]:
            raise ValueError(f"Invalid metrics type: {config.metrics.type}")

        # Validate reporter
        if config.reporter.type not in ["csv"]:
            raise ValueError(f"Invalid reporter type: {config.reporter.type}")

        return True
