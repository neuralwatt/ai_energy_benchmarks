"""Parameter-based reasoning control for DeepSeek, Qwen, EXAONE."""

from typing import Any, Dict, Optional

from ai_energy_benchmarks.formatters.base import ReasoningFormatter


class ParameterFormatter(ReasoningFormatter):
    """Parameter-based reasoning control.

    Models: DeepSeek, Qwen, EXAONE, Phi, Gemma

    Passes reasoning parameters to model.generate() or API:
        - enable_thinking: bool
        - thinking_budget: int (DeepSeek)
        - cot_depth: int (potential future use)
        - reasoning: bool (Phi, Gemma)
    """

    def format_prompt(self, prompt: str, reasoning_params: Optional[Dict[str, Any]] = None) -> str:
        """No prompt modification for parameter-based formatters.

        Args:
            prompt: Original user prompt
            reasoning_params: Optional reasoning parameters (unused for formatting)

        Returns:
            Unmodified prompt
        """
        return prompt

    def get_generation_params(
        self, reasoning_params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Return reasoning params to pass to generation.

        Args:
            reasoning_params: Optional reasoning parameters

        Returns:
            Dict of parameters to pass to model generation
        """
        if not reasoning_params:
            return {}

        # Filter out our internal params and pass through model-specific ones
        gen_params = {}

        # Known reasoning parameter keys
        known_keys = [
            "enable_thinking",
            "thinking_budget",
            "cot_depth",
            "reasoning",  # Generic parameter for Phi, Gemma, etc.
        ]

        for key in known_keys:
            if key in reasoning_params:
                gen_params[key] = reasoning_params[key]

        return gen_params
