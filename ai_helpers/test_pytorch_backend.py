#!/usr/bin/env python3
"""Test PyTorch backend with GPU monitoring."""

import sys

sys.path.insert(0, ".")

from ai_energy_benchmarks.backends.pytorch import PyTorchBackend
from ai_energy_benchmarks.utils.gpu import GPUMonitor
from ai_energy_benchmarks.reporters.csv_reporter import CSVReporter
import os
import time


def print_banner(text):
    """Print a formatted banner."""
    print("\n" + "=" * 80)
    print(f" {text}")
    print("=" * 80 + "\n")


def print_section(text):
    """Print a section header."""
    print(f"\n{'─' * 80}")
    print(f" {text}")
    print("─" * 80)


def main():
    """Test PyTorch backend."""
    print_banner("PyTorch Backend Test with GPU Monitoring")

    # Configuration
    model_name = "microsoft/phi-2"  # Small 2.7B model
    num_prompts = 3
    output_file = "./results/pytorch_backend_test.csv"

    print(f"Configuration:")
    print(f"  Model: {model_name}")
    print(f"  Prompts: {num_prompts}")
    print(f"  Output: {output_file}")

    # Create output directory
    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)

    # Test prompts
    test_prompts = [
        "What is the capital of France?",
        "Explain machine learning in one sentence.",
        "Count from 1 to 5.",
    ][:num_prompts]

    # Step 1: Check GPU baseline
    print_section("Step 1: GPU Baseline State")
    baseline = GPUMonitor.get_gpu_stats(0)
    if baseline:
        print(
            f"Memory: {baseline.memory_used_mb:.0f} MB / {baseline.memory_total_mb:.0f} MB ({baseline.memory_percent:.1f}%)"
        )
        print(f"Utilization: {baseline.utilization_percent:.1f}%")
        if baseline.power_draw_w:
            print(f"Power: {baseline.power_draw_w:.1f}W")
    else:
        print("✗ Unable to query GPU")
        return 1

    # Step 2: Initialize backend
    print_section("Step 2: Initialize PyTorch Backend")
    print(f"Loading model: {model_name}")
    print("This may take 1-2 minutes for first load...")

    backend = PyTorchBackend(
        model=model_name, device="cuda", device_ids=[0], torch_dtype="auto", device_map="auto"
    )

    # Validate environment
    print("\nValidating environment...", end=" ")
    if not backend.validate_environment():
        print("✗ FAILED")
        return 1
    print("✓ PASS")

    # Step 3: Model loading with GPU monitoring
    print_section("Step 3: Model Loading")

    def load_model():
        return backend.health_check()

    load_result = GPUMonitor.monitor_during_operation(load_model, gpu_id=0, interval=0.5)

    if not load_result["success"]:
        print(f"✗ Model loading failed: {load_result['error']}")
        return 1

    print("✓ Model loaded successfully")
    print(f"  Load time: {load_result['duration']:.2f}s")

    # Check GPU after loading
    after_load = GPUMonitor.get_gpu_stats(0)
    if after_load:
        print(f"\nGPU State After Model Load:")
        print(
            f"  Memory: {after_load.memory_used_mb:.0f} MB / {after_load.memory_total_mb:.0f} MB ({after_load.memory_percent:.1f}%)"
        )
        print(f"  Memory increase: {after_load.memory_used_mb - baseline.memory_used_mb:.0f} MB")
        if after_load.power_draw_w:
            print(f"  Power: {after_load.power_draw_w:.1f}W")

    # Step 4: Run inferences with GPU monitoring
    print_section("Step 4: Running Inferences with GPU Monitoring")

    results = []
    gpu_samples = []

    for i, prompt in enumerate(test_prompts):
        print(f"\nPrompt {i + 1}/{len(test_prompts)}: '{prompt}'")
        print("Running inference with GPU monitoring...")

        def run_inference():
            return backend.run_inference(prompt, max_tokens=30)

        monitored = GPUMonitor.monitor_during_operation(run_inference, gpu_id=0, interval=0.1)

        if not monitored["success"]:
            print(f"  ✗ Failed: {monitored['error']}")
            continue

        result = monitored["result"]
        gpu_stats = monitored["gpu_stats"]

        results.append(result)
        gpu_samples.append(gpu_stats)

        # Display results
        if result["success"]:
            print(f"  ✓ Success")
            print(f"    Response: {result['text'][:80]}...")
            print(
                f"    Tokens: {result['total_tokens']} ({result['prompt_tokens']} + {result['completion_tokens']})"
            )
            print(f"    Latency: {result['latency_seconds']:.3f}s")
            print(
                f"    GPU Utilization: {gpu_stats['avg_utilization_percent']:.1f}% avg, {gpu_stats['max_utilization_percent']:.1f}% peak"
            )
            print(
                f"    GPU Memory: {gpu_stats['avg_memory_mb']:.0f} MB avg, {gpu_stats['max_memory_mb']:.0f} MB peak"
            )
            if gpu_stats["avg_power_w"]:
                print(
                    f"    Power: {gpu_stats['avg_power_w']:.1f}W avg, {gpu_stats['max_power_w']:.1f}W peak"
                )
            print(f"    GPU Active: {'✓ YES' if gpu_stats['gpu_active'] else '✗ NO'}")
        else:
            print(f"  ✗ Failed: {result['error']}")

    # Step 5: Aggregate results
    print_section("Step 5: Results Summary")

    successful = [r for r in results if r.get("success", False)]

    if not successful:
        print("✗ No successful inferences")
        return 1

    total_tokens = sum(r["total_tokens"] for r in successful)
    total_duration = sum(r["latency_seconds"] for r in successful)
    avg_latency = total_duration / len(successful)
    throughput = total_tokens / total_duration if total_duration > 0 else 0

    # GPU stats
    any_gpu_active = any(s["gpu_active"] for s in gpu_samples)
    avg_gpu_util = sum(s["avg_utilization_percent"] for s in gpu_samples) / len(gpu_samples)
    max_gpu_util = max(s["max_utilization_percent"] for s in gpu_samples)
    avg_memory = sum(s["avg_memory_mb"] for s in gpu_samples) / len(gpu_samples)
    max_memory = max(s["max_memory_mb"] for s in gpu_samples)

    print(f"Successful inferences: {len(successful)}/{len(test_prompts)}")
    print(f"Total tokens: {total_tokens}")
    print(f"Average latency: {avg_latency:.3f}s")
    print(f"Throughput: {throughput:.2f} tokens/second")
    print(f"\nGPU Statistics:")
    print(f"  Average utilization: {avg_gpu_util:.1f}%")
    print(f"  Peak utilization: {max_gpu_util:.1f}%")
    print(f"  Average memory: {avg_memory:.0f} MB")
    print(f"  Peak memory: {max_memory:.0f} MB")
    print(f"  GPU actively used: {'✓ YES' if any_gpu_active else '✗ NO'}")

    # Save to CSV
    reporter = CSVReporter(output_file)
    summary = {
        "backend": "pytorch",
        "model": model_name,
        "total_prompts": len(test_prompts),
        "successful_prompts": len(successful),
        "total_tokens": total_tokens,
        "avg_latency_seconds": avg_latency,
        "throughput_tokens_per_second": throughput,
        "avg_gpu_utilization_percent": avg_gpu_util,
        "max_gpu_utilization_percent": max_gpu_util,
        "avg_memory_mb": avg_memory,
        "max_memory_mb": max_memory,
        "gpu_active": any_gpu_active,
    }
    reporter.report(summary)
    print(f"\n✓ Results saved to: {output_file}")

    # Step 6: Cleanup
    print_section("Step 6: Cleanup")
    backend.cleanup()
    print("✓ Model unloaded and GPU memory freed")

    time.sleep(1)
    after_cleanup = GPUMonitor.get_gpu_stats(0)
    if after_cleanup:
        print(f"\nGPU State After Cleanup:")
        print(
            f"  Memory: {after_cleanup.memory_used_mb:.0f} MB ({after_cleanup.memory_percent:.1f}%)"
        )
        print(f"  Memory freed: {after_load.memory_used_mb - after_cleanup.memory_used_mb:.0f} MB")

    # Final verdict
    print_banner("TEST RESULT")
    if any_gpu_active and len(successful) == len(test_prompts):
        print("✓ PyTorch Backend Test PASSED")
        print(f"  - All {len(test_prompts)} inferences successful")
        print(f"  - GPU activity confirmed")
        print(f"  - Throughput: {throughput:.2f} tok/s")
        return 0
    else:
        print("✗ PyTorch Backend Test FAILED")
        if not any_gpu_active:
            print("  - GPU activity not detected")
        if len(successful) != len(test_prompts):
            print(f"  - Only {len(successful)}/{len(test_prompts)} inferences successful")
        return 1


if __name__ == "__main__":
    sys.exit(main())
