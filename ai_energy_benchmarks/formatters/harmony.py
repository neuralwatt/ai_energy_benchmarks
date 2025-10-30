"""Harmony formatting for gpt-oss models."""

from typing import Any, Dict, Optional

from ai_energy_benchmarks.formatters.base import ReasoningFormatter


class HarmonyFormatter(ReasoningFormatter):
    """Harmony formatting for gpt-oss models.

    Format:
        <|start|>system<|message|>You are a helpful AI assistant.
        Reasoning: {effort}
        # Valid channels: analysis, commentary, final<|end|>
        <|start|>user<|message|>{prompt}<|end|>
    """

    def format_prompt(self, prompt: str, reasoning_params: Optional[Dict[str, Any]] = None) -> str:
        """Format prompt with Harmony structure.

        Args:
            prompt: Original user prompt
            reasoning_params: Optional dict with 'reasoning_effort' key

        Returns:
            Harmony-formatted prompt string
        """
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
        self, reasoning_params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Harmony uses prompt formatting only, no extra params.

        Args:
            reasoning_params: Optional reasoning parameters (unused)

        Returns:
            Empty dict (no additional generation parameters)
        """
        return {}
