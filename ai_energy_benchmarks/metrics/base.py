"""Base interface for metrics collectors."""

from abc import ABC, abstractmethod
from typing import Any, Dict


class MetricsCollector(ABC):
    """Base interface for all metrics collectors."""

    @abstractmethod
    def start(self) -> None:
        """Start metrics collection."""
        pass

    @abstractmethod
    def stop(self) -> Dict[str, Any]:
        """Stop collection and return metrics.

        Returns:
            Dict with collected metrics
        """
        pass

    @abstractmethod
    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current metrics without stopping.

        Returns:
            Dict with current metrics
        """
        pass

    @abstractmethod
    def get_metadata(self) -> Dict[str, Any]:
        """Get collector metadata (name, version, capabilities).

        Returns:
            Dict with metadata
        """
        pass
