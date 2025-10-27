"""Unit tests for reasoning formatters."""

import pytest

from ai_energy_benchmarks.formatters.harmony import HarmonyFormatter
from ai_energy_benchmarks.formatters.parameter import ParameterFormatter
from ai_energy_benchmarks.formatters.prefix import PrefixFormatter
from ai_energy_benchmarks.formatters.registry import FormatterRegistry
from ai_energy_benchmarks.formatters.system_prompt import SystemPromptFormatter


class TestHarmonyFormatter:
    """Tests for HarmonyFormatter."""

    def test_format_prompt_high_effort(self):
        formatter = HarmonyFormatter()
        prompt = "Explain quantum computing"
        result = formatter.format_prompt(prompt, reasoning_params={"reasoning_effort": "high"})

        assert "<|start|>system<|message|>" in result
        assert "Reasoning: high" in result
        assert prompt in result
        assert "<|start|>user<|message|>" in result

    def test_format_prompt_medium_effort(self):
        formatter = HarmonyFormatter()
        prompt = "Test prompt"
        result = formatter.format_prompt(prompt, reasoning_params={"reasoning_effort": "medium"})

        assert "Reasoning: medium" in result
        assert prompt in result

    def test_format_prompt_default_effort(self):
        formatter = HarmonyFormatter()
        result = formatter.format_prompt("Test prompt")
        assert "Reasoning: high" in result  # Default

    def test_get_generation_params(self):
        formatter = HarmonyFormatter()
        params = formatter.get_generation_params()
        assert params == {}  # Harmony uses prompt formatting only


class TestSystemPromptFormatter:
    """Tests for SystemPromptFormatter."""

    def test_smollm_think_flag(self):
        formatter = SystemPromptFormatter(
            enable_flag="/think", disable_flag="/no_think", default_enabled=False
        )

        # Enable thinking
        result = formatter.format_prompt("Test prompt", reasoning_params={"enable_thinking": True})
        assert result.startswith("/think")

        # Disable thinking (no flag needed if default is False)
        result = formatter.format_prompt("Test prompt", reasoning_params={"enable_thinking": False})
        assert not result.startswith("/think")
        assert not result.startswith("/no_think")

    def test_nemotron_default_on(self):
        formatter = SystemPromptFormatter(
            enable_flag=None, disable_flag="/no_think", default_enabled=True
        )

        # Thinking on by default (no flag)
        result = formatter.format_prompt("Test prompt")
        assert not result.startswith("/no_think")
        assert result == "Test prompt"

        # Disable thinking
        result = formatter.format_prompt("Test prompt", reasoning_params={"enable_thinking": False})
        assert result.startswith("/no_think")

    def test_reasoning_effort_mapping(self):
        formatter = SystemPromptFormatter(
            enable_flag="/think", disable_flag="/no_think", default_enabled=False
        )

        # High effort should enable thinking
        result = formatter.format_prompt(
            "Test prompt", reasoning_params={"reasoning_effort": "high"}
        )
        assert result.startswith("/think")

        # Low effort should not enable thinking (maps to False)
        result = formatter.format_prompt(
            "Test prompt", reasoning_params={"reasoning_effort": "low"}
        )
        assert not result.startswith("/think")

    def test_get_generation_params(self):
        formatter = SystemPromptFormatter()
        params = formatter.get_generation_params()
        assert params == {}  # System prompt formatting only


class TestParameterFormatter:
    """Tests for ParameterFormatter."""

    def test_format_prompt_no_modification(self):
        formatter = ParameterFormatter()
        original = "Test prompt"
        result = formatter.format_prompt(original)
        assert result == original

    def test_get_generation_params_enable_thinking(self):
        formatter = ParameterFormatter()
        result = formatter.get_generation_params(
            reasoning_params={"enable_thinking": True, "thinking_budget": 100}
        )

        assert result["enable_thinking"] is True
        assert result["thinking_budget"] == 100

    def test_get_generation_params_filters_unknown(self):
        formatter = ParameterFormatter()
        result = formatter.get_generation_params(
            reasoning_params={
                "enable_thinking": True,
                "unknown_param": "should_be_filtered",
            }
        )

        assert "enable_thinking" in result
        assert "unknown_param" not in result

    def test_get_generation_params_reasoning_parameter(self):
        """Test that generic 'reasoning' parameter is passed through (Phi, Gemma)."""
        formatter = ParameterFormatter()
        result = formatter.get_generation_params(reasoning_params={"reasoning": True})

        assert result["reasoning"] is True

    def test_get_generation_params_empty(self):
        formatter = ParameterFormatter()
        result = formatter.get_generation_params()
        assert result == {}


class TestPrefixFormatter:
    """Tests for PrefixFormatter."""

    def test_format_with_prefix(self):
        formatter = PrefixFormatter(prefix="<think>", suffix="")
        result = formatter.format_prompt("Test prompt", reasoning_params={"enable_thinking": True})

        assert result == "<think>Test prompt"

    def test_format_without_thinking(self):
        formatter = PrefixFormatter(prefix="<think>", suffix="")
        result = formatter.format_prompt("Test prompt", reasoning_params={"enable_thinking": False})

        assert result == "Test prompt"

    def test_format_with_suffix(self):
        formatter = PrefixFormatter(prefix="<think>", suffix="</think>")
        result = formatter.format_prompt("Test prompt", reasoning_params={"enable_thinking": True})

        assert result == "<think>Test prompt</think>"

    def test_default_enable_thinking(self):
        formatter = PrefixFormatter(prefix="<think>", suffix="")
        result = formatter.format_prompt("Test prompt")
        assert result == "<think>Test prompt"  # Default is True

    def test_get_generation_params(self):
        formatter = PrefixFormatter(prefix="<think>")
        params = formatter.get_generation_params()
        assert params == {}  # Prefix formatting only


class TestFormatterRegistry:
    """Tests for FormatterRegistry."""

    def test_get_gpt_oss_formatter(self):
        registry = FormatterRegistry()
        formatter = registry.get_formatter("openai/gpt-oss-20b")
        assert isinstance(formatter, HarmonyFormatter)

    def test_get_gpt_oss_120b_formatter(self):
        registry = FormatterRegistry()
        formatter = registry.get_formatter("openai/gpt-oss-120b")
        assert isinstance(formatter, HarmonyFormatter)

    def test_get_smollm_formatter(self):
        registry = FormatterRegistry()
        formatter = registry.get_formatter("HuggingFaceTB/SmolLM3-3B")
        assert isinstance(formatter, SystemPromptFormatter)

    def test_get_deepseek_formatter(self):
        registry = FormatterRegistry()
        formatter = registry.get_formatter("deepseek-ai/DeepSeek-R1")
        assert isinstance(formatter, PrefixFormatter)

    def test_get_qwen_formatter(self):
        """Qwen models now use chat template (type: null in config), not ParameterFormatter."""
        registry = FormatterRegistry()
        formatter = registry.get_formatter("Qwen/Qwen-2.5")
        # Qwen models should NOT have a formatter (they use chat template with enable_thinking)
        assert formatter is None

    def test_get_phi_formatter(self):
        registry = FormatterRegistry()
        formatter = registry.get_formatter("microsoft/Phi-4-reasoning-plus")
        assert isinstance(formatter, ParameterFormatter)

    def test_get_gemma_formatter(self):
        registry = FormatterRegistry()
        formatter = registry.get_formatter("google/gemma-2-9b")
        assert isinstance(formatter, ParameterFormatter)

    def test_unknown_model_returns_none(self):
        registry = FormatterRegistry()
        formatter = registry.get_formatter("some/unknown-model")
        assert formatter is None

    def test_formatter_caching(self):
        registry = FormatterRegistry()
        formatter1 = registry.get_formatter("openai/gpt-oss-20b")
        formatter2 = registry.get_formatter("openai/gpt-oss-20b")
        # Should return the same cached instance
        assert formatter1 is formatter2

    def test_pattern_matching_case_insensitive(self):
        registry = FormatterRegistry()
        formatter = registry.get_formatter("OpenAI/GPT-OSS-20B")
        assert isinstance(formatter, HarmonyFormatter)

    def test_nemotron_default_on(self):
        registry = FormatterRegistry()
        formatter = registry.get_formatter("nvidia/nemotron-340b")
        assert isinstance(formatter, SystemPromptFormatter)
        # Verify default_enabled is True
        assert formatter.default_enabled is True

    def test_hunyuan_formatter(self):
        registry = FormatterRegistry()
        formatter = registry.get_formatter("hunyuan-large")
        assert isinstance(formatter, SystemPromptFormatter)
        # Verify enable_flag is /think
        assert formatter.enable_flag == "/think"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
