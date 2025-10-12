#!/usr/bin/env python3
"""
Simplified benchmark runner that works without optional dependencies.
Uses manual prompts instead of HuggingFace datasets.
"""

import sys
import os
import time

sys.path.insert(0, ".")

from ai_energy_benchmarks.backends.vllm import VLLMBackend
from ai_energy_benchmarks.reporters.csv_reporter import CSVReporter


def run_simple_benchmark(
    model="openai/gpt-oss-120b",
    endpoint="http://localhost:8000/v1",
    num_samples=10,
    output_file="./results/benchmark_results.csv",
):
    """Run a simple benchmark without HuggingFace datasets or CodeCarbon.

    Args:
        model: Model name
        endpoint: vLLM endpoint URL
        num_samples: Number of prompts to run
        output_file: Output CSV file path
    """

    print("=" * 70)
    print(" AI ENERGY BENCHMARKS - SIMPLE RUNNER")
    print("=" * 70)
    print()

    # Create output directory
    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)

    # Test prompts (replaces HuggingFace datasets for now)
    all_prompts = [
        "What is machine learning?",
        "Explain photosynthesis in simple terms.",
        "What is the capital of France?",
        "How does a computer work?",
        "What is climate change?",
        "Explain quantum computing.",
        "What is artificial intelligence?",
        "How do airplanes fly?",
        "What is the internet?",
        "Explain DNA in simple terms.",
        "What causes earthquakes?",
        "How does electricity work?",
        "What is the solar system?",
        "Explain gravity.",
        "What are atoms?",
        "How do vaccines work?",
        "What is democracy?",
        "Explain evolution.",
        "What is renewable energy?",
        "How does the brain work?",
    ]

    # Select requested number of prompts
    prompts = all_prompts[:num_samples]

    print(f"Configuration:")
    print(f"  Model: {model}")
    print(f"  Endpoint: {endpoint}")
    print(f"  Prompts: {len(prompts)}")
    print(f"  Output: {output_file}")
    print()

    # Initialize backend
    print("Initializing vLLM backend...")
    backend = VLLMBackend(endpoint=endpoint, model=model)

    # Validate
    print("Validating environment...")
    if not backend.health_check():
        print("✗ Backend health check failed")
        print("  Make sure vLLM server is running:")
        print(f"    vllm serve {model} --port 8000")
        return 1

    if not backend.validate_environment():
        print("✗ Backend validation failed")
        print("  Check that the model is loaded in vLLM")
        return 1

    print("✓ Backend ready")
    print()

    # Initialize reporter
    reporter = CSVReporter(output_file)

    # Run benchmark
    print(f"Running benchmark on {len(prompts)} prompts...")
    print("-" * 70)

    start_time = time.time()
    results = []

    for i, prompt in enumerate(prompts):
        print(f"  Prompt {i + 1}/{len(prompts)}: {prompt[:50]:50s}", end="", flush=True)

        result = backend.run_inference(prompt, max_tokens=100, temperature=0.7)

        if result["success"]:
            results.append(result)
            print(f" ✓ {result['total_tokens']:3d} tok, {result['latency_seconds']:.3f}s")
        else:
            print(f" ✗ {result['error']}")

    end_time = time.time()

    # Calculate statistics
    successful = [r for r in results if r.get("success", False)]
    failed = len(prompts) - len(successful)

    total_tokens = sum(r.get("total_tokens", 0) for r in successful)
    total_prompt_tokens = sum(r.get("prompt_tokens", 0) for r in successful)
    total_completion_tokens = sum(r.get("completion_tokens", 0) for r in successful)
    avg_latency = (
        sum(r.get("latency_seconds", 0) for r in successful) / len(successful) if successful else 0
    )
    total_duration = end_time - start_time
    throughput = total_tokens / total_duration if total_duration > 0 else 0

    # Create summary
    summary = {
        "benchmark_name": "simple_benchmark",
        "backend": "vllm",
        "model": model,
        "endpoint": endpoint,
        "total_prompts": len(prompts),
        "successful_prompts": len(successful),
        "failed_prompts": failed,
        "total_duration_seconds": total_duration,
        "avg_latency_seconds": avg_latency,
        "total_tokens": total_tokens,
        "total_prompt_tokens": total_prompt_tokens,
        "total_completion_tokens": total_completion_tokens,
        "throughput_tokens_per_second": throughput,
    }

    # Save results
    print()
    print("Saving results...")
    reporter.report(summary)

    # Print summary
    print()
    print("=" * 70)
    print(" BENCHMARK RESULTS")
    print("=" * 70)
    print(f"Total prompts:       {summary['total_prompts']}")
    print(f"Successful:          {summary['successful_prompts']}")
    print(f"Failed:              {summary['failed_prompts']}")
    print(f"Duration:            {summary['total_duration_seconds']:.2f}s")
    print(f"Avg latency:         {summary['avg_latency_seconds']:.3f}s")
    print(f"Total tokens:        {summary['total_tokens']}")
    print(f"  Prompt tokens:     {summary['total_prompt_tokens']}")
    print(f"  Completion tokens: {summary['total_completion_tokens']}")
    print(f"Throughput:          {summary['throughput_tokens_per_second']:.2f} tok/s")
    print("=" * 70)
    print()
    print(f"✓ Results saved to: {output_file}")
    print()

    return 0


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run simple benchmark without optional dependencies"
    )
    parser.add_argument("--model", default="openai/gpt-oss-120b", help="Model name")
    parser.add_argument("--endpoint", default="http://localhost:8000/v1", help="vLLM endpoint URL")
    parser.add_argument("--num-samples", type=int, default=10, help="Number of prompts to run")
    parser.add_argument(
        "--output", default="./results/simple_benchmark_results.csv", help="Output CSV file"
    )

    args = parser.parse_args()

    sys.exit(
        run_simple_benchmark(
            model=args.model,
            endpoint=args.endpoint,
            num_samples=args.num_samples,
            output_file=args.output,
        )
    )
