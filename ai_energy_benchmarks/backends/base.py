"""Base interface for inference backends."""

from abc import ABC, abstractmethod
from typing import Any, Dict


class Backend(ABC):
    """Interface for inference backends (PyTorch, vLLM, etc.)."""

    @abstractmethod
    def validate_environment(self) -> bool:
        """Check if backend is available and properly configured.

        Returns:
            bool: True if backend is ready, False otherwise
        """
        pass

    @abstractmethod
    def get_endpoint_info(self) -> Dict[str, Any]:
        """Get backend endpoint information.

        Returns:
            Dict with backend configuration and status
        """
        pass

    @abstractmethod
    def health_check(self) -> bool:
        """Check if backend is healthy and ready.

        Returns:
            bool: True if backend is healthy, False otherwise
        """
        pass

    @abstractmethod
    def run_inference(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Run inference on a single prompt.

        Args:
            prompt: Input text prompt
            **kwargs: Additional generation parameters

        Returns:
            Dict with response text and metadata
        """
        pass
