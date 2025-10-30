"""System prompt flag-based formatting for SmolLM, Hunyuan, Nemotron."""

from typing import Any, Dict, Optional

from ai_energy_benchmarks.formatters.base import ReasoningFormatter


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
        enable_flag: Optional[str] = "/think",
        disable_flag: Optional[str] = "/no_think",
        default_enabled: bool = False,
    ):
        """Initialize formatter with model-specific flags.

        Args:
            enable_flag: Flag to enable thinking (e.g., "/think"), None if not used
            disable_flag: Flag to disable thinking (e.g., "/no_think"), None if not used
            default_enabled: Whether thinking is on by default
        """
        self.enable_flag = enable_flag
        self.disable_flag = disable_flag
        self.default_enabled = default_enabled

    def format_prompt(self, prompt: str, reasoning_params: Optional[Dict[str, Any]] = None) -> str:
        """Format prompt with system flags.

        Args:
            prompt: Original user prompt
            reasoning_params: Optional dict with 'enable_thinking' or 'reasoning_effort'

        Returns:
            Prompt with appropriate flag prepended if needed
        """
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
            flag = self.enable_flag if self.enable_flag else ""
        elif not enable_thinking and self.default_enabled:
            flag = self.disable_flag if self.disable_flag else ""
        else:
            flag = ""

        if flag:
            # Prepend flag to prompt
            return f"{flag}\n{prompt}"
        return prompt

    def get_generation_params(
        self, reasoning_params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """System prompt formatting only, no extra params.

        Args:
            reasoning_params: Optional reasoning parameters (unused)

        Returns:
            Empty dict (no additional generation parameters)
        """
        return {}
