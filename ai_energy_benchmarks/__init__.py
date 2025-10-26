"""AI Energy Benchmarks

A modular benchmarking framework for AI energy measurements.
"""

__version__ = "0.0.1rc1"
__author__ = "Neuralwatt"
__license__ = "MIT"

# Import key classes for convenience
from ai_energy_benchmarks.config.parser import BenchmarkConfig, ConfigParser
from ai_energy_benchmarks.runner import BenchmarkRunner

__all__ = ["BenchmarkRunner", "BenchmarkConfig", "ConfigParser", "__version__"]
