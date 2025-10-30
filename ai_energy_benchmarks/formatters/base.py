"""Base class for reasoning format strategies."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


class ReasoningFormatter(ABC):
    """Base class for reasoning format strategies."""

    @abstractmethod
    def format_prompt(self, prompt: str, reasoning_params: Optional[Dict[str, Any]] = None) -> str:
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
        self, reasoning_params: Optional[Dict[str, Any]] = None
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
