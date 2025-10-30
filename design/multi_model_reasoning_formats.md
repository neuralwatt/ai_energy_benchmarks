# Multi-Model Reasoning Format Support - Design Document

**Version:** 1.0
**Date:** 2025-10-19
**Authors:** AI Energy Benchmarks Team
**Status:** Draft

---

## Executive Summary

### Problem Statement

Current reasoning format handling in `ai_energy_benchmarks` and `AIEnergyScore` is model-specific and hardcoded. Each new reasoning-capable model requires custom code changes across multiple files:

- **gpt-oss models**: Hardcoded Harmony formatting in both PyTorch and vLLM backends
- **DeepSeek models**: Partial support with `<think>` prefix approach
- **SmolLM models**: Not yet supported (requires `/think` and `/no_think` system flags)
- **Other models**: Scattered across `model_config_parser.py` with inconsistent handling

This approach is **not scalable** and violates the principle that `ai_energy_benchmarks` should be a generic AI benchmarking library without model-specific logic bleeding in.

### Proposed Solution

Create a **unified reasoning format abstraction layer** that:

1. **Decouples** reasoning format logic from backend implementations
2. **Centralizes** format definitions in a YAML registry
3. **Provides** pluggable formatter classes for extensibility
4. **Maintains** backward compatibility with deprecation warnings
5. **Keeps** `ai_energy_benchmarks` generic and model-agnostic

---

## Current State Analysis

### Existing Implementation

#### 1. Harmony Formatting (gpt-oss models)

**Location:** `ai_energy_benchmarks/backends/{vllm.py, pytorch.py}`

```python
# Hardcoded in both backends
def format_harmony_prompt(self, text: str, reasoning_effort: str = "high") -> str:
    harmony_prompt = (
        "<|start|>system<|message|>"
        "You are a helpful AI assistant.\n"
        f"Reasoning: {reasoning_effort}\n"
        "# Valid channels: analysis, commentary, final"
        "<|end|>\n"
        f"<|start|>user<|message|>{text}<|end|>"
    )
    return harmony_prompt

# Auto-detection based on model name
self.use_harmony = "gpt-oss" in model.lower()
```

**Issues:**
- Duplicated code across backends
- Hardcoded format logic
- No extensibility for other formats

#### 2. AIEnergyScore Model Parser

**Location:** `AIEnergyScore/model_config_parser.py` (lines 132-211)

```python
def _parse_chat_template(self, config: ModelConfig) -> None:
    # gpt-oss models
    if "gpt-oss" in model_id_lower:
        config.use_harmony = True
        if "reasoning_effort: high" in template:
            config.reasoning_params = {"reasoning_effort": "high"}

    # DeepSeek models
    elif "deepseek" in model_id_lower:
        if "<think>" in template:
            config.prompt_prefix = "<think>"

    # Qwen models
    elif "qwen" in model_id_lower:
        if "enable_thinking=true" in template:
            config.reasoning_params = {"enable_thinking": True}

    # ... 80+ lines of similar if/elif logic
```

**Issues:**
- Massive if/elif chain for 10+ model families
- Tight coupling to model names
- Difficult to test and maintain
- No separation of concerns

#### 3. Missing Support

- **SmolLM3**: System prompt flags (`/think`, `/no_think`)
- **Hunyuan**: `/think` prefix
- **Nemotron**: `/no_think` to disable thinking
- **Future models**: Requires code changes

---

## Requirements

### Functional Requirements

1. **FR-1:** Support multiple reasoning format strategies:
   - Harmony formatting (gpt-oss)
   - System prompt flags (SmolLM, Hunyuan, Nemotron)
   - Parameter-based (DeepSeek, Qwen, EXAONE)
   - Prefix/suffix tags (DeepSeek `<think>`)

2. **FR-2:** Registry-based configuration
   - YAML file mapping model families to formats
   - Hybrid granularity: family defaults + per-model overrides
   - Easy to add new models without code changes

3. **FR-3:** Backend-agnostic design
   - Works with both PyTorch and vLLM backends
   - Consistent behavior across backends
   - No backend-specific format logic

4. **FR-4:** Backward compatibility
   - Existing configs continue to work
   - Deprecation warnings for old approaches
   - Clear migration path

### Non-Functional Requirements

1. **NFR-1:** `ai_energy_benchmarks` must remain generic
   - No training-specific logic
   - No NeuralWatt-specific code
   - Pure benchmarking library

2. **NFR-2:** Testability
   - Unit tests for each formatter
   - Integration tests with real models
   - Mock-friendly design

3. **NFR-3:** Code quality
   - Pass ruff, ruff-format, mypy, black
   - Type hints throughout
   - Clear documentation

---

## Architecture Design

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    ai_energy_benchmarks                  │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  ┌──────────────┐         ┌──────────────┐             │
│  │ PyTorch      │         │ vLLM         │             │
│  │ Backend      │         │ Backend      │             │
│  └──────┬───────┘         └──────┬───────┘             │
│         │                        │                      │
│         └────────┬───────────────┘                      │
│                  │                                      │
│         ┌────────▼────────┐                            │
│         │  Formatter       │                            │
│         │  Registry        │◄───┐                      │
│         └────────┬────────┘     │                      │
│                  │               │                      │
│    ┌─────────────┼───────────────┼────────────┐       │
│    │             │               │             │       │
│ ┌──▼──┐  ┌──────▼─────┐  ┌──────▼──┐  ┌──────▼──┐   │
│ │Harmony│ │SystemPrompt│ │Parameter│ │ Prefix  │   │
│ │Format │ │Formatter   │ │Formatter│ │Formatter│   │
│ └───────┘ └────────────┘ └─────────┘ └─────────┘   │
│                                                          │
│  ┌──────────────────────────────────────┐              │
│  │   reasoning_formats.yaml             │              │
│  │   (Registry Configuration)           │              │
│  └──────────────────────────────────────┘              │
└─────────────────────────────────────────────────────────┘
                         │
                         │ imports
                         │
┌────────────────────────▼─────────────────────────────────┐
│                    AIEnergyScore                          │
├───────────────────────────────────────────────────────────┤
│  ┌──────────────────────────────────────┐                │
│  │  model_config_parser.py              │                │
│  │  (Uses formatters from library)      │                │
│  └──────────────────────────────────────┘                │
└───────────────────────────────────────────────────────────┘
```

### Core Components

#### 1. ReasoningFormatter (Base Class)

```python
# ai_energy_benchmarks/formatters/base.py

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

class ReasoningFormatter(ABC):
    """Base class for reasoning format strategies."""

    @abstractmethod
    def format_prompt(
        self,
        prompt: str,
        reasoning_params: Optional[Dict[str, Any]] = None
    ) -> str:
        """Format prompt with reasoning instructions.

        Args:
            prompt: Original user prompt
            reasoning_params: Optional reasoning parameters (effort, flags, etc.)

        Returns:
            Formatted prompt string
        """
        pass

    @abstractmethod
    def get_generation_params(
        self,
        reasoning_params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Get backend-specific generation parameters.

        Args:
            reasoning_params: Optional reasoning parameters

        Returns:
            Dict of parameters to pass to model.generate() or API
        """
        pass

    def supports_backend(self, backend_type: str) -> bool:
        """Check if formatter supports given backend.

        Args:
            backend_type: 'pytorch' or 'vllm'

        Returns:
            True if supported
        """
        return True  # Default: support all backends
```

#### 2. Concrete Formatters

##### HarmonyFormatter (gpt-oss models)

```python
# ai_energy_benchmarks/formatters/harmony.py

from typing import Any, Dict, Optional
from .base import ReasoningFormatter

class HarmonyFormatter(ReasoningFormatter):
    """Harmony formatting for gpt-oss models.

    Format:
        <|start|>system<|message|>You are a helpful AI assistant.
        Reasoning: {effort}
        # Valid channels: analysis, commentary, final<|end|>
        <|start|>user<|message|>{prompt}<|end|>
    """

    def format_prompt(
        self,
        prompt: str,
        reasoning_params: Optional[Dict[str, Any]] = None
    ) -> str:
        effort = "high"  # Default
        if reasoning_params and "reasoning_effort" in reasoning_params:
            effort = reasoning_params["reasoning_effort"]

        return (
            "<|start|>system<|message|>"
            "You are a helpful AI assistant.\n"
            f"Reasoning: {effort}\n"
            "# Valid channels: analysis, commentary, final"
            "<|end|>\n"
            f"<|start|>user<|message|>{prompt}<|end|>"
        )

    def get_generation_params(
        self,
        reasoning_params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Harmony uses prompt formatting only, no extra params."""
        return {}
```

##### SystemPromptFormatter (SmolLM, Hunyuan, Nemotron)

```python
# ai_energy_benchmarks/formatters/system_prompt.py

from typing import Any, Dict, Optional
from .base import ReasoningFormatter

class SystemPromptFormatter(ReasoningFormatter):
    """System prompt flag-based formatting.

    Models: SmolLM3, Hunyuan, Nemotron

    Format:
        - SmolLM: /think or /no_think in system prompt
        - Hunyuan: /think in system prompt
        - Nemotron: /no_think to disable (thinking on by default)
    """

    def __init__(
        self,
        enable_flag: str = "/think",
        disable_flag: Optional[str] = "/no_think",
        default_enabled: bool = False
    ):
        """Initialize formatter with model-specific flags.

        Args:
            enable_flag: Flag to enable thinking (e.g., "/think")
            disable_flag: Flag to disable thinking (e.g., "/no_think")
            default_enabled: Whether thinking is on by default
        """
        self.enable_flag = enable_flag
        self.disable_flag = disable_flag
        self.default_enabled = default_enabled

    def format_prompt(
        self,
        prompt: str,
        reasoning_params: Optional[Dict[str, Any]] = None
    ) -> str:
        # Determine if thinking should be enabled
        enable_thinking = self.default_enabled

        if reasoning_params:
            if "enable_thinking" in reasoning_params:
                enable_thinking = reasoning_params["enable_thinking"]
            elif "reasoning_effort" in reasoning_params:
                # Map reasoning_effort to enable/disable
                effort = reasoning_params["reasoning_effort"]
                enable_thinking = effort in ["high", "medium"]

        # Add appropriate flag to system prompt
        if enable_thinking and not self.default_enabled:
            flag = self.enable_flag
        elif not enable_thinking and self.default_enabled:
            flag = self.disable_flag if self.disable_flag else ""
        else:
            flag = ""

        if flag:
            # Prepend flag to prompt
            return f"{flag}\n{prompt}"
        return prompt

    def get_generation_params(
        self,
        reasoning_params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """System prompt formatting only, no extra params."""
        return {}
```

##### ParameterFormatter (DeepSeek, Qwen, EXAONE)

```python
# ai_energy_benchmarks/formatters/parameter.py

from typing import Any, Dict, Optional
from .base import ReasoningFormatter

class ParameterFormatter(ReasoningFormatter):
    """Parameter-based reasoning control.

    Models: DeepSeek, Qwen, EXAONE

    Passes reasoning parameters to model.generate() or API:
        - enable_thinking: bool
        - thinking_budget: int (DeepSeek)
        - cot_depth: int (potential future use)
    """

    def format_prompt(
        self,
        prompt: str,
        reasoning_params: Optional[Dict[str, Any]] = None
    ) -> str:
        """No prompt modification for parameter-based formatters."""
        return prompt

    def get_generation_params(
        self,
        reasoning_params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Return reasoning params to pass to generation."""
        if not reasoning_params:
            return {}

        # Filter out our internal params and pass through model-specific ones
        gen_params = {}

        for key in ["enable_thinking", "thinking_budget", "cot_depth"]:
            if key in reasoning_params:
                gen_params[key] = reasoning_params[key]

        return gen_params
```

##### PrefixFormatter (DeepSeek `<think>`)

```python
# ai_energy_benchmarks/formatters/prefix.py

from typing import Any, Dict, Optional
from .base import ReasoningFormatter

class PrefixFormatter(ReasoningFormatter):
    """Prefix/suffix-based formatting.

    Models: DeepSeek (with <think> tag)

    Format:
        <think>{prompt}
    """

    def __init__(self, prefix: str = "", suffix: str = ""):
        """Initialize with prefix/suffix strings.

        Args:
            prefix: String to prepend to prompt
            suffix: String to append to prompt
        """
        self.prefix = prefix
        self.suffix = suffix

    def format_prompt(
        self,
        prompt: str,
        reasoning_params: Optional[Dict[str, Any]] = None
    ) -> str:
        # Check if thinking is enabled
        enable_thinking = True
        if reasoning_params and "enable_thinking" in reasoning_params:
            enable_thinking = reasoning_params["enable_thinking"]

        if enable_thinking:
            return f"{self.prefix}{prompt}{self.suffix}"
        return prompt

    def get_generation_params(
        self,
        reasoning_params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Prefix formatting only, no extra params."""
        return {}
```

#### 3. Formatter Registry

```python
# ai_energy_benchmarks/formatters/registry.py

from pathlib import Path
from typing import Dict, Optional
import yaml
import re

from .base import ReasoningFormatter
from .harmony import HarmonyFormatter
from .system_prompt import SystemPromptFormatter
from .parameter import ParameterFormatter
from .prefix import PrefixFormatter

class FormatterRegistry:
    """Registry for reasoning formatters."""

    def __init__(self, config_path: Optional[str] = None):
        """Initialize registry.

        Args:
            config_path: Path to reasoning_formats.yaml
        """
        if config_path is None:
            # Default to bundled config
            config_path = Path(__file__).parent.parent / "config" / "reasoning_formats.yaml"

        self.config_path = Path(config_path)
        self.config = self._load_config()
        self._formatters: Dict[str, ReasoningFormatter] = {}

    def _load_config(self) -> dict:
        """Load YAML configuration."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config not found: {self.config_path}")

        with open(self.config_path) as f:
            return yaml.safe_load(f)

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
            return None

        # Create formatter
        formatter = self._create_formatter(model_config)
        self._formatters[model_name] = formatter
        return formatter

    def _find_model_config(self, model_name: str) -> Optional[dict]:
        """Find config for model by pattern matching.

        Args:
            model_name: Model identifier

        Returns:
            Config dict or None
        """
        model_lower = model_name.lower()

        # Check for exact match first
        if model_name in self.config.get("models", {}):
            return self.config["models"][model_name]

        # Check model families
        families = self.config.get("families", {})
        for family_name, family_config in families.items():
            patterns = family_config.get("patterns", [])
            for pattern in patterns:
                if re.search(pattern, model_lower):
                    # Check for model-specific override
                    overrides = family_config.get("model_overrides", {})
                    for override_pattern, override_config in overrides.items():
                        if re.search(override_pattern, model_lower):
                            # Merge family config with override
                            merged = {**family_config, **override_config}
                            return merged
                    return family_config

        return None

    def _create_formatter(self, config: dict) -> ReasoningFormatter:
        """Create formatter from config.

        Args:
            config: Formatter configuration

        Returns:
            ReasoningFormatter instance
        """
        formatter_type = config.get("type")

        if formatter_type == "harmony":
            return HarmonyFormatter()

        elif formatter_type == "system_prompt":
            return SystemPromptFormatter(
                enable_flag=config.get("enable_flag", "/think"),
                disable_flag=config.get("disable_flag", "/no_think"),
                default_enabled=config.get("default_enabled", False)
            )

        elif formatter_type == "parameter":
            return ParameterFormatter()

        elif formatter_type == "prefix":
            return PrefixFormatter(
                prefix=config.get("prefix", ""),
                suffix=config.get("suffix", "")
            )

        else:
            raise ValueError(f"Unknown formatter type: {formatter_type}")
```

---

## Registry Format (YAML Schema)

### File Location
`ai_energy_benchmarks/config/reasoning_formats.yaml`

### Schema

```yaml
# Model families with pattern matching
families:
  # gpt-oss models (Harmony formatting)
  gpt-oss:
    patterns:
      - "gpt-oss"
      - "openai/gpt-oss"
    type: harmony
    description: "OpenAI GPT-OSS models using Harmony format"

  # SmolLM models (System prompt flags)
  smollm:
    patterns:
      - "smollm"
      - "huggingfacetb/smollm"
    type: system_prompt
    enable_flag: "/think"
    disable_flag: "/no_think"
    default_enabled: false
    description: "SmolLM3 models using /think and /no_think flags"

    # Per-model overrides
    model_overrides:
      "smollm3-3b":
        # SmolLM3-3B specific config (if different from family)
        enable_flag: "/think"

  # DeepSeek models (Prefix + Parameter)
  deepseek:
    patterns:
      - "deepseek"
      - "deepseek-ai"
    type: prefix
    prefix: "<think>"
    suffix: ""
    description: "DeepSeek-R1 models using <think> prefix"

    # Alternative for parameter-based DeepSeek models
    model_overrides:
      "deepseek-r1.*parameter":
        type: parameter

  # Qwen models (Parameter-based)
  qwen:
    patterns:
      - "qwen"
    type: parameter
    description: "Qwen models using enable_thinking parameter"

  # Hunyuan models (System prompt)
  hunyuan:
    patterns:
      - "hunyuan"
    type: system_prompt
    enable_flag: "/think"
    disable_flag: null
    default_enabled: false
    description: "Hunyuan models using /think flag"

  # Nemotron models (System prompt, default ON)
  nemotron:
    patterns:
      - "nemotron"
      - "nvidia/nemotron"
    type: system_prompt
    enable_flag: null
    disable_flag: "/no_think"
    default_enabled: true
    description: "Nemotron models (thinking on by default, /no_think to disable)"

  # EXAONE models (Parameter-based)
  exaone:
    patterns:
      - "exaone"
    type: parameter
    description: "EXAONE models using enable_thinking parameter"

# Explicit model overrides (highest priority)
models:
  "openai/gpt-oss-20b":
    type: harmony

  "openai/gpt-oss-120b":
    type: harmony

  "HuggingFaceTB/SmolLM3-3B":
    type: system_prompt
    enable_flag: "/think"
    disable_flag: "/no_think"
    default_enabled: false

# Default formatter for unknown models
default:
  type: null  # No reasoning formatting by default
```

---

## Implementation Details

### Integration with Backends

#### Modified vLLM Backend

```python
# ai_energy_benchmarks/backends/vllm.py

from ai_energy_benchmarks.formatters.registry import FormatterRegistry

class VLLMBackend(Backend):
    def __init__(
        self,
        endpoint: str,
        model: str,
        timeout: int = 300,
        use_harmony: Optional[bool] = None,  # DEPRECATED
    ):
        self.endpoint = endpoint.rstrip("/v1").rstrip("/")
        self.model = model
        self.timeout = timeout

        # DEPRECATED: Backward compatibility
        if use_harmony is not None:
            import warnings
            warnings.warn(
                "use_harmony parameter is deprecated. "
                "Reasoning format is now auto-detected via FormatterRegistry. "
                "This parameter will be removed in v2.0.",
                DeprecationWarning,
                stacklevel=2
            )
            self._legacy_use_harmony = use_harmony
        else:
            self._legacy_use_harmony = None

        # Initialize formatter registry
        self.formatter_registry = FormatterRegistry()
        self.formatter = self.formatter_registry.get_formatter(model)

    def run_inference(
        self,
        prompt: str,
        max_tokens: int = 100,
        temperature: float = 0.7,
        reasoning_params: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        start_time = time.time()

        # Format prompt using registry formatter
        formatted_prompt = prompt
        extra_gen_params = {}

        if self.formatter:
            formatted_prompt = self.formatter.format_prompt(prompt, reasoning_params)
            extra_gen_params = self.formatter.get_generation_params(reasoning_params)
        elif self._legacy_use_harmony:
            # DEPRECATED: Legacy Harmony formatting
            formatted_prompt = self._legacy_format_harmony(prompt, reasoning_params)

        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": formatted_prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        # Add formatter-provided parameters
        if extra_gen_params:
            payload["extra_body"] = extra_gen_params

        # ... rest of inference logic

    def _legacy_format_harmony(self, prompt: str, reasoning_params: Optional[Dict[str, Any]]) -> str:
        """DEPRECATED: Legacy Harmony formatting."""
        import warnings
        warnings.warn(
            "Legacy Harmony formatting is deprecated. "
            "Use FormatterRegistry instead.",
            DeprecationWarning,
            stacklevel=2
        )
        # ... old harmony logic
```

#### Modified PyTorch Backend

Similar refactoring for `pytorch.py`:

```python
# ai_energy_benchmarks/backends/pytorch.py

from ai_energy_benchmarks.formatters.registry import FormatterRegistry

class PyTorchBackend(Backend):
    def __init__(
        self,
        model: str,
        device: str = "cuda",
        device_ids: Optional[List[int]] = None,
        torch_dtype: str = "auto",
        device_map: str = "auto",
        max_memory: Optional[Dict[str, Any]] = None,
        use_harmony: Optional[bool] = None,  # DEPRECATED
    ):
        # Similar deprecation handling as vLLM
        # Initialize formatter registry
        self.formatter_registry = FormatterRegistry()
        self.formatter = self.formatter_registry.get_formatter(model)

    def run_inference(
        self,
        prompt: str,
        max_tokens: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        reasoning_params: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        # Format prompt using registry formatter
        formatted_prompt = prompt
        extra_gen_params = {}

        if self.formatter:
            formatted_prompt = self.formatter.format_prompt(prompt, reasoning_params)
            extra_gen_params = self.formatter.get_generation_params(reasoning_params)

        # Tokenize and generate
        # ... existing logic with extra_gen_params merged into gen_kwargs
```

### Integration with AIEnergyScore

#### Modified model_config_parser.py

```python
# AIEnergyScore/model_config_parser.py

from ai_energy_benchmarks.formatters.registry import FormatterRegistry

class ModelConfigParser:
    def __init__(self, csv_path: str):
        self.csv_path = Path(csv_path)
        self.formatter_registry = FormatterRegistry()

    def _parse_chat_template(self, config: ModelConfig) -> None:
        """Parse chat template using ai_energy_benchmarks formatters."""
        template = config.chat_template.lower()
        model_id_lower = config.model_id.lower()

        # Skip if N/A or empty
        if not template or "n/a" in template:
            return

        # Get formatter from registry
        formatter = self.formatter_registry.get_formatter(config.model_id)

        if formatter:
            # Extract reasoning parameters from template
            reasoning_params = self._extract_reasoning_params(template)
            config.reasoning_params = reasoning_params

            # Set flags based on formatter type
            from ai_energy_benchmarks.formatters.harmony import HarmonyFormatter
            if isinstance(formatter, HarmonyFormatter):
                config.use_harmony = True

        else:
            # DEPRECATED: Fallback to old hardcoded logic
            import warnings
            warnings.warn(
                f"Model {config.model_id} not found in FormatterRegistry. "
                f"Using deprecated hardcoded format detection. "
                f"Please add model to reasoning_formats.yaml.",
                DeprecationWarning,
                stacklevel=2
            )
            self._legacy_parse_chat_template(config)

    def _extract_reasoning_params(self, template: str) -> Optional[Dict[str, Any]]:
        """Extract reasoning parameters from template string."""
        params = {}

        # Extract reasoning_effort
        if "reasoning_effort: high" in template or "reasoning: high" in template:
            params["reasoning_effort"] = "high"
        elif "reasoning_effort: medium" in template or "reasoning: medium" in template:
            params["reasoning_effort"] = "medium"
        elif "reasoning_effort: low" in template or "reasoning: low" in template:
            params["reasoning_effort"] = "low"

        # Extract enable_thinking
        if "enable_thinking=true" in template:
            params["enable_thinking"] = True
        elif "enable_thinking=false" in template:
            params["enable_thinking"] = False

        return params if params else None

    def _legacy_parse_chat_template(self, config: ModelConfig) -> None:
        """DEPRECATED: Old hardcoded logic for backward compatibility."""
        # Keep existing lines 132-211 logic here
        # ... gpt-oss, deepseek, qwen, etc.
```

---

## Migration Strategy

### Phase 1: Foundation (Week 1)
**Status:** No breaking changes, fully backward compatible

- [ ] Create `ai_energy_benchmarks/formatters/` package
- [ ] Implement base `ReasoningFormatter` class
- [ ] Implement concrete formatters (Harmony, SystemPrompt, Parameter, Prefix)
- [ ] Create `FormatterRegistry` with YAML loading
- [ ] Create `reasoning_formats.yaml` config
- [ ] Add unit tests for all formatters

### Phase 2: Integration (Week 2)
**Status:** Deprecation warnings, backward compatible

- [ ] Integrate `FormatterRegistry` into vLLM backend
- [ ] Integrate `FormatterRegistry` into PyTorch backend
- [ ] Keep old `format_harmony_prompt()` methods with deprecation warnings
- [ ] Add deprecation warning for `use_harmony` parameter
- [ ] Integration tests with real models (gpt-oss, SmolLM)

### Phase 3: AIEnergyScore Migration (Week 3)
**Status:** Deprecation warnings, backward compatible

- [ ] Update `model_config_parser.py` to use `FormatterRegistry`
- [ ] Keep old `_parse_chat_template()` logic with deprecation warnings
- [ ] Add new models to `reasoning_formats.yaml`
- [ ] Test with existing CSV configurations

### Phase 4: Cleanup (Future - v2.0)
**Status:** Breaking changes

- [ ] Remove deprecated `format_harmony_prompt()` methods
- [ ] Remove `use_harmony` parameter
- [ ] Remove old `_parse_chat_template()` logic
- [ ] Update documentation

---

## Testing Strategy

### Unit Tests

```python
# tests/test_formatters.py

import pytest
from ai_energy_benchmarks.formatters.harmony import HarmonyFormatter
from ai_energy_benchmarks.formatters.system_prompt import SystemPromptFormatter
from ai_energy_benchmarks.formatters.registry import FormatterRegistry

class TestHarmonyFormatter:
    def test_format_prompt_high_effort(self):
        formatter = HarmonyFormatter()
        prompt = "Explain quantum computing"
        result = formatter.format_prompt(
            prompt,
            reasoning_params={"reasoning_effort": "high"}
        )

        assert "<|start|>system<|message|>" in result
        assert "Reasoning: high" in result
        assert prompt in result

    def test_format_prompt_default_effort(self):
        formatter = HarmonyFormatter()
        result = formatter.format_prompt("Test prompt")
        assert "Reasoning: high" in result  # Default

class TestSystemPromptFormatter:
    def test_smollm_think_flag(self):
        formatter = SystemPromptFormatter(
            enable_flag="/think",
            disable_flag="/no_think",
            default_enabled=False
        )

        # Enable thinking
        result = formatter.format_prompt(
            "Test prompt",
            reasoning_params={"enable_thinking": True}
        )
        assert result.startswith("/think")

        # Disable thinking (no flag needed if default is False)
        result = formatter.format_prompt(
            "Test prompt",
            reasoning_params={"enable_thinking": False}
        )
        assert not result.startswith("/think")

    def test_nemotron_default_on(self):
        formatter = SystemPromptFormatter(
            enable_flag=None,
            disable_flag="/no_think",
            default_enabled=True
        )

        # Thinking on by default (no flag)
        result = formatter.format_prompt("Test prompt")
        assert not result.startswith("/no_think")

        # Disable thinking
        result = formatter.format_prompt(
            "Test prompt",
            reasoning_params={"enable_thinking": False}
        )
        assert result.startswith("/no_think")

class TestFormatterRegistry:
    def test_get_gpt_oss_formatter(self):
        registry = FormatterRegistry()
        formatter = registry.get_formatter("openai/gpt-oss-20b")
        assert isinstance(formatter, HarmonyFormatter)

    def test_get_smollm_formatter(self):
        registry = FormatterRegistry()
        formatter = registry.get_formatter("HuggingFaceTB/SmolLM3-3B")
        assert isinstance(formatter, SystemPromptFormatter)

    def test_unknown_model_returns_none(self):
        registry = FormatterRegistry()
        formatter = registry.get_formatter("some/unknown-model")
        assert formatter is None
```

### Integration Tests

```python
# tests/integration/test_reasoning_integration.py

import pytest
from ai_energy_benchmarks.backends.vllm import VLLMBackend
from ai_energy_benchmarks.backends.pytorch import PyTorchBackend

@pytest.mark.integration
class TestReasoningIntegration:
    def test_gpt_oss_harmony_formatting(self):
        """Test that gpt-oss models use Harmony formatting automatically."""
        backend = VLLMBackend(
            endpoint="http://localhost:8000",
            model="openai/gpt-oss-20b"
        )

        # Formatter should be auto-detected
        assert backend.formatter is not None

        # Format a prompt
        formatted = backend.formatter.format_prompt(
            "Test prompt",
            reasoning_params={"reasoning_effort": "high"}
        )
        assert "<|start|>system<|message|>" in formatted

    def test_smollm_system_prompt_formatting(self):
        """Test that SmolLM models use /think flags."""
        backend = PyTorchBackend(
            model="HuggingFaceTB/SmolLM3-3B",
            device="cuda"
        )

        assert backend.formatter is not None

        formatted = backend.formatter.format_prompt(
            "Test prompt",
            reasoning_params={"enable_thinking": True}
        )
        assert formatted.startswith("/think")
```

### Pre-Commit Checks

Ensure all changes pass:

```bash
# Run before committing to ai_energy_benchmarks
cd ai_energy_benchmarks
.venv/bin/pre-commit run --all-files

# Specific checks
.venv/bin/ruff check ai_energy_benchmarks/
.venv/bin/ruff format --check ai_energy_benchmarks/
.venv/bin/mypy ai_energy_benchmarks/
```

---

## Example Usage

### Example 1: Using ai_energy_benchmarks with SmolLM

```python
from ai_energy_benchmarks.backends.pytorch import PyTorchBackend

# Initialize backend (formatter auto-detected)
backend = PyTorchBackend(
    model="HuggingFaceTB/SmolLM3-3B",
    device="cuda"
)

# Run with thinking enabled
result = backend.run_inference(
    prompt="Explain quantum entanglement in simple terms",
    max_tokens=200,
    reasoning_params={"enable_thinking": True}
)

# The prompt is automatically formatted with /think flag
# Output will include reasoning trace
```

### Example 2: Using AIEnergyScore with Updated Parser

```python
from AIEnergyScore.model_config_parser import ModelConfigParser

# Parse CSV with model configs
parser = ModelConfigParser("AI Energy Score (Oct 2025) - Models.csv")
configs = parser.parse()

# SmolLM configs automatically get proper formatter
smollm_configs = [c for c in configs if "SmolLM" in c.model_id]
for config in smollm_configs:
    print(f"Model: {config.model_id}")
    print(f"Reasoning params: {config.reasoning_params}")
    # Formatter is auto-detected, no hardcoded logic needed
```

### Example 3: Adding a New Model to Registry

```yaml
# ai_energy_benchmarks/config/reasoning_formats.yaml

families:
  # ... existing families

  # New model family
  new-reasoning-model:
    patterns:
      - "company/new-reasoning"
    type: system_prompt
    enable_flag: "/reason"
    disable_flag: "/no_reason"
    default_enabled: false
    description: "New reasoning model using /reason flags"
```

No code changes needed! The formatter is automatically available.

---

## Open Questions & Future Considerations

### Open Questions

1. **Q: Should we support multiple formatters per model?**
   - Example: DeepSeek could use both `<think>` prefix AND parameter-based control
   - **Proposed:** Start with single formatter per model, add composite formatters if needed

2. **Q: How to handle tokenizer.apply_chat_template() for PyTorch?**
   - SmolLM example shows `tokenizer.apply_chat_template(..., enable_thinking=False)`
   - **Proposed:** Add `TokenizerFormatter` class for models that support this

3. **Q: Should reasoning_formats.yaml be in ai_energy_benchmarks or separate package?**
   - **Proposed:** Keep in ai_energy_benchmarks for now, consider separate package if widely adopted

### Future Enhancements

1. **Composite Formatters:** Chain multiple formatters (e.g., prefix + parameter)
2. **Dynamic Config:** Load reasoning_formats.yaml from remote URL
3. **Formatter Plugins:** Allow third-party formatters via entry points
4. **Auto-Detection:** Infer formatter from model card/config.json
5. **Validation:** Schema validation for reasoning_formats.yaml
6. **Telemetry:** Track which formatters are used in benchmarks

---

## Appendices

### Appendix A: Model Reasoning Format Matrix

| Model Family | Reasoning Format | Enable Method | Disable Method | Notes |
|--------------|------------------|---------------|----------------|-------|
| gpt-oss | Harmony | Reasoning: {effort} in system | N/A | Always on |
| SmolLM3 | System Prompt | /think | /no_think | Default off |
| DeepSeek-R1 | Prefix | `<think>` prefix | No prefix | Tag-based |
| Qwen | Parameter | enable_thinking=true | enable_thinking=false | API param |
| Hunyuan | System Prompt | /think | N/A | Default off |
| Nemotron | System Prompt | N/A (default on) | /no_think | Default on |
| EXAONE | Parameter | enable_thinking=true | enable_thinking=false | API param |

### Appendix B: References

1. **OpenAI Harmony Format:** https://github.com/openai/harmony
2. **SmolLM Documentation:** Hugging Face model cards
3. **DeepSeek-R1 Paper:** arXiv (thinking tokens)
4. **ai_energy_benchmarks README:** /mnt/storage/src/ai_energy_benchmarks/README.md
5. **CLAUDE.md Project Instructions:** /mnt/storage/src/CLAUDE.md

---

## Document History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2025-10-19 | AI Energy Team | Initial design document |

---

**End of Document**
