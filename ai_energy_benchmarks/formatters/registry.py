"""Registry for reasoning formatters."""

import re
from pathlib import Path
from typing import Any, Dict, Optional, cast

import yaml  # type: ignore[import-untyped]

from ai_energy_benchmarks.formatters.base import ReasoningFormatter
from ai_energy_benchmarks.formatters.harmony import HarmonyFormatter
from ai_energy_benchmarks.formatters.parameter import ParameterFormatter
from ai_energy_benchmarks.formatters.prefix import PrefixFormatter
from ai_energy_benchmarks.formatters.system_prompt import SystemPromptFormatter


class FormatterRegistry:
    """Registry for reasoning formatters."""

    def __init__(self, config_path: Optional[str] = None):
        """Initialize registry.

        Args:
            config_path: Path to reasoning_formats.yaml
        """
        if config_path is None:
            # Default to bundled config
            config_path_obj = Path(__file__).parent.parent / "config" / "reasoning_formats.yaml"
        else:
            config_path_obj = Path(config_path)

        self.config_path = config_path_obj
        self.config = self._load_config()
        self._formatters: Dict[str, Optional[ReasoningFormatter]] = {}

    def _load_config(self) -> Dict[str, Any]:
        """Load YAML configuration.

        Returns:
            Configuration dictionary

        Raises:
            FileNotFoundError: If config file doesn't exist
        """
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config not found: {self.config_path}")

        with open(self.config_path) as f:
            config = yaml.safe_load(f)
            if not isinstance(config, dict):
                return {}
            return config

    def get_formatter(self, model_name: str) -> Optional[ReasoningFormatter]:
        """Get formatter for a given model.

        Args:
            model_name: Model identifier (e.g., "openai/gpt-oss-20b")

        Returns:
            ReasoningFormatter instance or None
        """
        # Check cache
        if model_name in self._formatters:
            return self._formatters[model_name]

        # Find matching config
        model_config = self._find_model_config(model_name)
        if not model_config:
            # Cache None result
            self._formatters[model_name] = None
            return None

        # Create formatter
        formatter = self._create_formatter(model_config)
        self._formatters[model_name] = formatter
        return formatter

    def _find_model_config(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Find config for model by pattern matching.

        Args:
            model_name: Model identifier

        Returns:
            Config dict or None
        """
        model_lower = model_name.lower()

        # Check for exact match first
        if "models" in self.config and model_name in self.config["models"]:
            return cast(Dict[str, Any], self.config["models"][model_name])

        # Check model families
        families = self.config.get("families", {})
        for _family_name, family_config in families.items():
            patterns = family_config.get("patterns", [])
            for pattern in patterns:
                if re.search(pattern, model_lower):
                    # Check for model-specific override
                    overrides = family_config.get("model_overrides", {})
                    for override_pattern, override_config in overrides.items():
                        if re.search(override_pattern, model_lower):
                            # Merge family config with override
                            merged = {**family_config, **override_config}
                            return cast(Dict[str, Any], merged)
                    return cast(Dict[str, Any], family_config)

        return None

    def _create_formatter(self, config: Dict[str, Any]) -> Optional[ReasoningFormatter]:
        """Create formatter from config.

        Args:
            config: Formatter configuration

        Returns:
            ReasoningFormatter instance or None

        Raises:
            ValueError: If formatter type is unknown
        """
        formatter_type = config.get("type")

        if formatter_type == "harmony":
            return HarmonyFormatter()

        elif formatter_type == "system_prompt":
            return SystemPromptFormatter(
                enable_flag=config.get("enable_flag", "/think"),
                disable_flag=config.get("disable_flag", "/no_think"),
                default_enabled=config.get("default_enabled", False),
            )

        elif formatter_type == "parameter":
            return ParameterFormatter()

        elif formatter_type == "prefix":
            return PrefixFormatter(prefix=config.get("prefix", ""), suffix=config.get("suffix", ""))

        elif formatter_type is None or formatter_type == "null":
            # No formatter for this model
            return None

        else:
            raise ValueError(f"Unknown formatter type: {formatter_type}")
