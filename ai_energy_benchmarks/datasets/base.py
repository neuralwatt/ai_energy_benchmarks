"""Base interface for dataset loaders."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List


class Dataset(ABC):
    """Interface for dataset loaders."""

    @abstractmethod
    def load(self, config: Dict[str, Any]) -> List[str]:
        """Load prompts from dataset.

        Args:
            config: Dataset configuration

        Returns:
            List of prompt strings
        """
        pass

    @abstractmethod
    def validate(self) -> bool:
        """Validate dataset availability.

        Returns:
            bool: True if dataset is available, False otherwise
        """
        pass
