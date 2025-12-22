"""Load profile configuration for genai-perf benchmarks."""

from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class LoadProfileConfig:
    """Configuration for a single load profile phase.

    Attributes:
        name: Profile name (e.g., 'light', 'moderate', 'heavy', 'stress')
        description: Human-readable description of the profile
        concurrency: Number of concurrent requests
        request_count: Number of requests to send
        input_token_range: Min and max input tokens as tuple
        output_token_range: Min and max output tokens as tuple
        cache_strategy: Cache strategy for prompt generation ('minimal', 'low', 'realistic')
    """

    name: str
    description: str = ""
    concurrency: int = 1
    request_count: int = 100
    input_token_range: Tuple[int, int] = (100, 500)
    output_token_range: Tuple[int, int] = (100, 500)
    cache_strategy: str = "minimal"

    def to_dict(self) -> dict:
        """Convert to dictionary for genai-perf command building."""
        return {
            "name": self.name,
            "description": self.description,
            "concurrency": self.concurrency,
            "request_count": self.request_count,
            "input_token_range": list(self.input_token_range),
            "output_token_range": list(self.output_token_range),
            "cache_strategy": self.cache_strategy,
        }


@dataclass
class MultiPhaseProfile:
    """Configuration for multi-phase profiles like 'train'.

    Attributes:
        name: Profile name (e.g., 'train', 'pattern')
        description: Human-readable description
        phases: List of LoadProfileConfig phases to execute sequentially
        cache_strategy: Default cache strategy for all phases
    """

    name: str
    description: str = ""
    phases: List[LoadProfileConfig] = field(default_factory=list)
    cache_strategy: str = "minimal"

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "description": self.description,
            "type": "multi-phase",
            "phases": [phase.to_dict() for phase in self.phases],
            "cache_strategy": self.cache_strategy,
        }


__all__ = ["LoadProfileConfig", "MultiPhaseProfile"]
