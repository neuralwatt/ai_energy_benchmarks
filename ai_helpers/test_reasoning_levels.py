#!/usr/bin/env python3
"""
Test script to validate reasoning levels work consistently across:
1. ai_energy_benchmark (PyTorch backend)
2. ai_energy_benchmark (vLLM backend)
3. optimum-benchmark (AIEnergyScore wrapper)

This script runs benchmarks with different reasoning effort levels (low, medium, high)
and compares the results to ensure consistency.
"""

import json
import sys
import os
import subprocess
from pathlib import Path
from typing import Dict, Any, List

# Add ai_energy_benchmarks to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ai_energy_benchmarks.config.parser import (
    BenchmarkConfig, BackendConfig, ScenarioConfig,
    MetricsConfig, ReporterConfig
)
from ai_energy_benchmarks.runner import BenchmarkRunner


def run_ai_energy_benchmark_pytorch(reasoning_effort: str, output_dir: Path) -> Dict[str, Any]:
    """Run ai_energy_benchmark with PyTorch backend and specified reasoning effort."""
    print(f"\n{'='*60}")
    print(f"Running ai_energy_benchmark (PyTorch) with reasoning effort: {reasoning_effort}")
    print(f"{'='*60}\n")

    output_path = output_dir / f"pytorch_{reasoning_effort}"
    output_path.mkdir(parents=True, exist_ok=True)

    backend_cfg = BackendConfig(
        type="pytorch",
        model="openai/gpt-oss-20b",
        device="cuda",
        device_ids=[0],
    )

    scenario_cfg = ScenarioConfig(
        dataset_name="EnergyStarAI/text_generation",
        text_column_name="text",
        #dataset_name="scottcha/reasoning_text_generation",
        #text_column_name="prompt",
        num_samples=5,  # Small sample for testing
        reasoning=True,
        reasoning_params={
            "reasoning_effort": reasoning_effort,
            "use_prompt_based_reasoning": True  # Enable prompt-based reasoning for gpt-oss-20b
        },
        generate_kwargs={
            "max_new_tokens": 2000,  # Allow more tokens for reasoning
            "min_new_tokens": 50
        },
    )

    metrics_cfg = MetricsConfig(
        enabled=True,
        type="codecarbon",
        project_name=f"reasoning_test_pytorch_{reasoning_effort}",
        output_dir=str(output_path / "emissions"),
        country_iso_code="USA",
        region="california",
    )

    reporter_cfg = ReporterConfig(
        type="csv",
        output_file=str(output_path / "benchmark_results.csv"),
    )

    bench_config = BenchmarkConfig(
        name=f"reasoning_test_pytorch_{reasoning_effort}",
        backend=backend_cfg,
        scenario=scenario_cfg,
        metrics=metrics_cfg,
        reporter=reporter_cfg,
        output_dir=str(output_path),
    )

    runner = BenchmarkRunner(bench_config)

    if not runner.validate():
        print(f"ERROR: Benchmark validation failed for PyTorch {reasoning_effort}")
        return {"error": "validation_failed"}

    results = runner.run()

    # Save results
    result_file = output_path / "benchmark_report.json"
    with open(result_file, 'w') as f:
        json.dump(results, f, indent=2)

    return results


def run_ai_energy_benchmark_vllm(reasoning_effort: str, output_dir: Path, endpoint: str) -> Dict[str, Any]:
    """Run ai_energy_benchmark with vLLM backend and specified reasoning effort."""
    print(f"\n{'='*60}")
    print(f"Running ai_energy_benchmark (vLLM) with reasoning effort: {reasoning_effort}")
    print(f"{'='*60}\n")

    output_path = output_dir / f"vllm_{reasoning_effort}"
    output_path.mkdir(parents=True, exist_ok=True)

    backend_cfg = BackendConfig(
        type="vllm",
        model="openai/gpt-oss-20b",
        endpoint=endpoint,
    )

    scenario_cfg = ScenarioConfig(
        dataset_name="EnergyStarAI/text_generation",
        text_column_name="text",
        num_samples=10,
        reasoning=True,
        reasoning_params={"reasoning_effort": reasoning_effort},
        generate_kwargs={
            "max_new_tokens": 100,
            "min_new_tokens": 50
        },
    )

    metrics_cfg = MetricsConfig(
        enabled=True,
        type="codecarbon",
        project_name=f"reasoning_test_vllm_{reasoning_effort}",
        output_dir=str(output_path / "emissions"),
        country_iso_code="USA",
        region="california",
    )

    reporter_cfg = ReporterConfig(
        type="csv",
        output_file=str(output_path / "benchmark_results.csv"),
    )

    bench_config = BenchmarkConfig(
        name=f"reasoning_test_vllm_{reasoning_effort}",
        backend=backend_cfg,
        scenario=scenario_cfg,
        metrics=metrics_cfg,
        reporter=reporter_cfg,
        output_dir=str(output_path),
    )

    runner = BenchmarkRunner(bench_config)

    if not runner.validate():
        print(f"ERROR: Benchmark validation failed for vLLM {reasoning_effort}")
        return {"error": "validation_failed"}

    results = runner.run()

    # Save results
    result_file = output_path / "benchmark_report.json"
    with open(result_file, 'w') as f:
        json.dump(results, f, indent=2)

    return results


def run_optimum_benchmark(reasoning_effort: str, output_dir: Path) -> Dict[str, Any]:
    """Run optimum-benchmark (AIEnergyScore) with specified reasoning effort."""
    print(f"\n{'='*60}")
    print(f"Running optimum-benchmark (AIEnergyScore) with reasoning effort: {reasoning_effort}")
    print(f"{'='*60}\n")

    output_path = output_dir / f"optimum_{reasoning_effort}"
    output_path.mkdir(parents=True, exist_ok=True)

    # Use the AIEnergyScore wrapper script
    wrapper_script = Path("/home/scott/src/AIEnergyScore/run_ai_energy_benchmark.py")
    config_name = f"text_generation_gptoss_reasoning_{reasoning_effort}"

    env = os.environ.copy()
    env['RESULTS_DIR'] = str(output_path)
    env['BENCHMARK_BACKEND'] = 'pytorch'

    try:
        result = subprocess.run(
            [sys.executable, str(wrapper_script), f"--config-name={config_name}"],
            cwd=str(wrapper_script.parent),
            env=env,
            capture_output=True,
            text=True,
            timeout=600
        )

        if result.returncode != 0:
            print(f"ERROR: optimum-benchmark failed for {reasoning_effort}")
            print(f"STDOUT: {result.stdout}")
            print(f"STDERR: {result.stderr}")
            return {"error": "execution_failed", "stdout": result.stdout, "stderr": result.stderr}

        # Load results
        result_file = output_path / "benchmark_report.json"
        if result_file.exists():
            with open(result_file) as f:
                return json.load(f)
        else:
            return {"error": "no_results_file"}

    except subprocess.TimeoutExpired:
        print(f"ERROR: optimum-benchmark timed out for {reasoning_effort}")
        return {"error": "timeout"}
    except Exception as e:
        print(f"ERROR: Exception running optimum-benchmark: {e}")
        return {"error": str(e)}


def compare_results(all_results: Dict[str, Dict[str, Any]]) -> None:
    """Compare results across different backends and reasoning levels."""
    print(f"\n{'='*60}")
    print("COMPARISON RESULTS")
    print(f"{'='*60}\n")

    # Define reasoning efforts
    efforts = ["low", "medium", "high"]
    backends = ["pytorch", "vllm", "optimum"]

    # Print header
    print(f"{'Reasoning Effort':<20} | {'Backend':<15} | {'Success':<10} | {'Avg Latency (s)':<18} | {'Energy (Wh)':<15} | {'Tokens/s':<12}")
    print("-" * 110)

    # Print results for each combination
    for effort in efforts:
        for backend in backends:
            key = f"{backend}_{effort}"
            if key in all_results:
                result = all_results[key]
                if "error" in result:
                    print(f"{effort:<20} | {backend:<15} | {'FAILED':<10} | {'-':<18} | {'-':<15} | {'-':<12}")
                else:
                    summary = result.get('summary', {})
                    energy = result.get('energy', {})
                    success = summary.get('successful_prompts', 0)
                    avg_lat = summary.get('avg_latency_seconds', 0)
                    energy_wh = energy.get('energy_wh', 0)
                    throughput = summary.get('throughput_tokens_per_second', 0)
                    print(f"{effort:<20} | {backend:<15} | {success:<10} | {avg_lat:<18.2f} | {energy_wh:<15.2f} | {throughput:<12.2f}")
            else:
                print(f"{effort:<20} | {backend:<15} | {'SKIPPED':<10} | {'-':<18} | {'-':<15} | {'-':<12}")

    # Analysis: Check if reasoning effort impacts latency/energy
    print(f"\n{'='*60}")
    print("ANALYSIS")
    print(f"{'='*60}\n")

    for backend in backends:
        print(f"\n{backend.upper()} Backend:")
        latencies = []
        energies = []
        for effort in efforts:
            key = f"{backend}_{effort}"
            if key in all_results and "error" not in all_results[key]:
                result = all_results[key]
                summary = result.get('summary', {})
                energy = result.get('energy', {})
                latencies.append((effort, summary.get('avg_latency_seconds', 0)))
                energies.append((effort, energy.get('energy_wh', 0)))

        if latencies:
            print("  Latencies:")
            for effort, lat in latencies:
                print(f"    {effort}: {lat:.2f}s")

            # Check for variation
            lat_values = [l[1] for l in latencies]
            if max(lat_values) - min(lat_values) > 0.5:
                print(f"  ✓ Latency varies with reasoning effort (range: {min(lat_values):.2f}s - {max(lat_values):.2f}s)")
            else:
                print(f"  ⚠ Latency does NOT vary significantly with reasoning effort")

        if energies:
            print("  Energies:")
            for effort, eng in energies:
                print(f"    {effort}: {eng:.2f} Wh")

            # Check for variation
            eng_values = [e[1] for e in energies]
            if max(eng_values) - min(eng_values) > 1.0:
                print(f"  ✓ Energy varies with reasoning effort (range: {min(eng_values):.2f}Wh - {max(eng_values):.2f}Wh)")
            else:
                print(f"  ⚠ Energy does NOT vary significantly with reasoning effort")


def main():
    """Main test execution."""
    print("="*60)
    print("REASONING LEVELS TEST")
    print("Testing reasoning parameters across benchmark engines")
    print("="*60)

    # Setup output directory
    output_dir = Path("./test_results/reasoning_levels")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get vLLM endpoint from environment or use default
    vllm_endpoint = os.getenv("VLLM_ENDPOINT", "http://localhost:8000/v1")

    # Store all results
    all_results = {}

    # Test each reasoning effort level with each backend
    for effort in ["low", "medium", "high"]:
        # Test PyTorch backend
        try:
            result = run_ai_energy_benchmark_pytorch(effort, output_dir)
            all_results[f"pytorch_{effort}"] = result
        except Exception as e:
            print(f"ERROR: PyTorch backend failed for {effort}: {e}")
            all_results[f"pytorch_{effort}"] = {"error": str(e)}

        # Test vLLM backend (if endpoint is available)
        if vllm_endpoint:
            try:
                result = run_ai_energy_benchmark_vllm(effort, output_dir, vllm_endpoint)
                all_results[f"vllm_{effort}"] = result
            except Exception as e:
                print(f"ERROR: vLLM backend failed for {effort}: {e}")
                all_results[f"vllm_{effort}"] = {"error": str(e)}
        else:
            print(f"SKIPPING vLLM backend test (no endpoint configured)")

        # Test optimum-benchmark
        try:
            result = run_optimum_benchmark(effort, output_dir)
            all_results[f"optimum_{effort}"] = result
        except Exception as e:
            print(f"ERROR: optimum-benchmark failed for {effort}: {e}")
            all_results[f"optimum_{effort}"] = {"error": str(e)}

    # Compare results
    compare_results(all_results)

    # Save all results
    summary_file = output_dir / "test_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\nTest results saved to: {summary_file}")
    print("\nDone!")


if __name__ == "__main__":
    main()
