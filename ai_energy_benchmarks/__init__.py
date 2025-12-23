"""AI Energy Benchmarks

A modular benchmarking framework for AI energy measurements.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("ai_energy_benchmarks")
except PackageNotFoundError:
    # Package not installed, fall back to reading VERSION.txt
    from pathlib import Path

    _version_file = Path(__file__).parent.parent / "VERSION.txt"
    if _version_file.exists():
        __version__ = _version_file.read_text().strip()
    else:
        __version__ = "0.0.0"  # Development fallback
__author__ = "Neuralwatt"
__license__ = "MIT"

# Import key classes for convenience
from ai_energy_benchmarks.config.parser import BenchmarkConfig, ConfigParser
from ai_energy_benchmarks.runner import BenchmarkRunner

__all__ = ["BenchmarkRunner", "BenchmarkConfig", "ConfigParser", "__version__"]
