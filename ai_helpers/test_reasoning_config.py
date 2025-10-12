#!/usr/bin/env python3
"""Simple test to validate reasoning configuration parsing."""

import sys
from pathlib import Path

# Add ai_energy_benchmarks to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ai_energy_benchmarks.config.parser import (
    BenchmarkConfig,
    BackendConfig,
    ScenarioConfig,
    MetricsConfig,
    ReporterConfig,
)


def test_scenario_config_with_reasoning():
    """Test that ScenarioConfig accepts reasoning parameters."""
    print("Testing ScenarioConfig with reasoning parameters...")

    scenario = ScenarioConfig(
        dataset_name="EnergyStarAI/text_generation",
        text_column_name="text",
        num_samples=10,
        reasoning=True,
        reasoning_params={"reasoning_effort": "high"},
        generate_kwargs={"max_new_tokens": 100},
    )

    assert scenario.reasoning == True, "reasoning should be True"
    assert scenario.reasoning_params is not None, "reasoning_params should not be None"
    assert scenario.reasoning_params["reasoning_effort"] == "high", (
        "reasoning_effort should be 'high'"
    )

    print("✓ ScenarioConfig with reasoning parameters works correctly")


def test_full_benchmark_config():
    """Test full BenchmarkConfig with reasoning."""
    print("\nTesting full BenchmarkConfig with reasoning...")

    backend_cfg = BackendConfig(
        type="pytorch",
        model="openai/gpt-oss-20b",
        device="cuda",
        device_ids=[0],
    )

    scenario_cfg = ScenarioConfig(
        dataset_name="EnergyStarAI/text_generation",
        text_column_name="text",
        num_samples=10,
        reasoning=True,
        reasoning_params={"reasoning_effort": "medium"},
        generate_kwargs={"max_new_tokens": 100},
    )

    metrics_cfg = MetricsConfig(
        enabled=True,
        type="codecarbon",
        project_name="test_reasoning",
        output_dir="./emissions",
    )

    reporter_cfg = ReporterConfig(type="csv", output_file="./results.csv")

    config = BenchmarkConfig(
        name="test_reasoning",
        backend=backend_cfg,
        scenario=scenario_cfg,
        metrics=metrics_cfg,
        reporter=reporter_cfg,
        output_dir="./output",
    )

    assert config.scenario.reasoning == True
    assert config.scenario.reasoning_params["reasoning_effort"] == "medium"

    print("✓ Full BenchmarkConfig with reasoning works correctly")


def test_reasoning_disabled():
    """Test config with reasoning disabled."""
    print("\nTesting config with reasoning disabled...")

    scenario = ScenarioConfig(
        dataset_name="EnergyStarAI/text_generation",
        text_column_name="text",
        num_samples=10,
        reasoning=False,
        reasoning_params=None,
        generate_kwargs={"max_new_tokens": 100},
    )

    assert scenario.reasoning == False
    assert scenario.reasoning_params is None

    print("✓ Config with reasoning disabled works correctly")


def main():
    print("=" * 60)
    print("REASONING CONFIGURATION TESTS")
    print("=" * 60)

    try:
        test_scenario_config_with_reasoning()
        test_full_benchmark_config()
        test_reasoning_disabled()

        print("\n" + "=" * 60)
        print("ALL TESTS PASSED ✓")
        print("=" * 60)
        return 0

    except AssertionError as e:
        print(f"\nTEST FAILED: {e}")
        return 1
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
