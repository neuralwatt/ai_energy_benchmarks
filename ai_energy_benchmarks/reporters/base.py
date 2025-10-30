"""Base interface for results reporters."""

from abc import ABC, abstractmethod
from typing import Any, Dict


class Reporter(ABC):
    """Results reporter interface."""

    @abstractmethod
    def report(self, results: Dict[str, Any]) -> None:
        """Report benchmark results.

        Args:
            results: Benchmark results to report
        """
        pass

    @abstractmethod
    def validate(self) -> bool:
        """Validate reporter configuration.

        Returns:
            bool: True if reporter is properly configured, False otherwise
        """
        pass
