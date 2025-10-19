"""Prefix/suffix-based formatting for DeepSeek with <think> tag."""

from typing import Any, Dict, Optional

from ai_energy_benchmarks.formatters.base import ReasoningFormatter


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

    def format_prompt(self, prompt: str, reasoning_params: Optional[Dict[str, Any]] = None) -> str:
        """Format prompt with prefix/suffix.

        Args:
            prompt: Original user prompt
            reasoning_params: Optional dict with 'enable_thinking'

        Returns:
            Prompt with prefix/suffix if thinking is enabled
        """
        # Check if thinking is enabled
        enable_thinking = True
        if reasoning_params and "enable_thinking" in reasoning_params:
            enable_thinking = reasoning_params["enable_thinking"]

        if enable_thinking:
            return f"{self.prefix}{prompt}{self.suffix}"
        return prompt

    def get_generation_params(
        self, reasoning_params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Prefix formatting only, no extra params.

        Args:
            reasoning_params: Optional reasoning parameters (unused)

        Returns:
            Empty dict (no additional generation parameters)
        """
        return {}
