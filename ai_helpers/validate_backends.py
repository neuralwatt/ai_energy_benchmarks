#!/usr/bin/env python3
"""Comprehensive validation of both PyTorch and vLLM backends with GPU monitoring."""

import sys
import time

sys.path.insert(0, ".")

from ai_energy_benchmarks.backends.vllm import VLLMBackend
from ai_energy_benchmarks.backends.pytorch import PyTorchBackend
from ai_energy_benchmarks.utils.gpu import GPUMonitor
from ai_energy_benchmarks.reporters.csv_reporter import CSVReporter
import os


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


def validate_vllm_backend():
    """Validate vLLM backend with GPU monitoring."""
    print_banner("VALIDATING vLLM BACKEND")

    # Check GPU availability
    print_section("Step 1: GPU Availability Check")
    if not GPUMonitor.check_gpu_available(0):
        print("✗ GPU not available or cannot be queried")
        return False

    GPUMonitor.print_gpu_info(0)

    # Initialize backend
    print_section("Step 2: Initialize vLLM Backend")
    backend = VLLMBackend(endpoint="http://localhost:8000/v1", model="openai/gpt-oss-120b")

    # Validate
    print("Health check...", end=" ")
    if backend.health_check():
        print("✓ PASS")
    else:
        print("✗ FAIL")
        return False

    print("Environment validation...", end=" ")
    if backend.validate_environment():
        print("✓ PASS")
    else:
        print("✗ FAIL")
        return False

    # Run inference with GPU monitoring
    print_section("Step 3: Run Inference with GPU Monitoring")
    test_prompt = "Explain what machine learning is in one sentence."

    print(f"Prompt: '{test_prompt}'")
    print("Monitoring GPU during inference...")

    def run_inference():
        return backend.run_inference(test_prompt, max_tokens=50)

    monitored_result = GPUMonitor.monitor_during_operation(run_inference, gpu_id=0, interval=0.2)

    if not monitored_result["success"]:
        print(f"✗ Inference failed: {monitored_result['error']}")
        return False

    result = monitored_result["result"]
    gpu_stats = monitored_result["gpu_stats"]

    # Display results
    print(f"\n✓ Inference successful")
    print(f"  Response: {result['text'][:100]}...")
    print(
        f"  Tokens: {result['total_tokens']} ({result['prompt_tokens']} prompt + {result['completion_tokens']} completion)"
    )
    print(f"  Latency: {result['latency_seconds']:.3f}s")

    print(f"\nGPU Statistics:")
    print(f"  Samples collected: {gpu_stats['samples']}")
    print(f"  Average utilization: {gpu_stats['avg_utilization_percent']:.1f}%")
    print(f"  Peak utilization: {gpu_stats['max_utilization_percent']:.1f}%")
    print(f"  Average memory: {gpu_stats['avg_memory_mb']:.0f} MB")
    print(f"  Peak memory: {gpu_stats['max_memory_mb']:.0f} MB")
    if gpu_stats["avg_power_w"] is not None:
        print(f"  Average power: {gpu_stats['avg_power_w']:.1f}W")
        print(f"  Peak power: {gpu_stats['max_power_w']:.1f}W")

    # Validate GPU was actually used
    print(
        f"\n{'✓' if gpu_stats['gpu_active'] else '✗'} GPU Activity Detected: {gpu_stats['gpu_active']}"
    )
    if not gpu_stats["gpu_active"]:
        print("  Warning: GPU utilization was very low. Inference may not be using GPU.")

    return gpu_stats["gpu_active"]


def validate_pytorch_backend():
    """Validate PyTorch backend with GPU monitoring."""
    print_banner("VALIDATING PyTorch BACKEND")

    # Check if PyTorch is available
    print_section("Step 1: Environment Check")
    try:
        import torch
        import transformers

        print(f"✓ PyTorch version: {torch.__version__}")
        print(f"✓ Transformers version: {transformers.__version__}")
        print(f"✓ CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"✓ CUDA version: {torch.version.cuda}")
            print(f"✓ GPU count: {torch.cuda.device_count()}")
            print(f"✓ Current device: {torch.cuda.current_device()}")
            print(f"✓ Device name: {torch.cuda.get_device_name(0)}")
    except ImportError as e:
        print(f"✗ Missing dependency: {e}")
        print("  Install with: pip install torch transformers")
        return False

    # Check GPU before loading model
    print_section("Step 2: GPU State Before Model Load")
    GPUMonitor.print_gpu_info(0)

    # Initialize backend with a small model
    print_section("Step 3: Initialize PyTorch Backend")
    print("Using small model: microsoft/phi-2 (2.7B parameters)")
    print("Loading model... (this may take a minute)")

    backend = PyTorchBackend(
        model="microsoft/phi-2",
        device="cuda",
        device_ids=[0],
        torch_dtype="auto",
        device_map="auto",
    )

    # Validate environment
    print("\nValidating environment...", end=" ")
    if backend.validate_environment():
        print("✓ PASS")
    else:
        print("✗ FAIL")
        return False

    # Check GPU after model load
    print_section("Step 4: GPU State After Model Load")
    GPUMonitor.print_gpu_info(0)

    # Run inference with GPU monitoring
    print_section("Step 5: Run Inference with GPU Monitoring")
    test_prompt = "What is the capital of France?"

    print(f"Prompt: '{test_prompt}'")
    print("Monitoring GPU during inference...")

    def run_inference():
        return backend.run_inference(test_prompt, max_tokens=30)

    monitored_result = GPUMonitor.monitor_during_operation(
        run_inference,
        gpu_id=0,
        interval=0.1,  # More frequent sampling
    )

    if not monitored_result["success"]:
        print(f"✗ Inference failed: {monitored_result['error']}")
        # Cleanup even on failure
        backend.cleanup()
        return False

    result = monitored_result["result"]
    gpu_stats = monitored_result["gpu_stats"]

    # Display results
    print(f"\n✓ Inference successful")
    print(f"  Response: {result['text'][:100]}...")
    print(
        f"  Tokens: {result['total_tokens']} ({result['prompt_tokens']} prompt + {result['completion_tokens']} completion)"
    )
    print(f"  Latency: {result['latency_seconds']:.3f}s")

    print(f"\nGPU Statistics:")
    print(f"  Samples collected: {gpu_stats['samples']}")
    print(f"  Average utilization: {gpu_stats['avg_utilization_percent']:.1f}%")
    print(f"  Peak utilization: {gpu_stats['max_utilization_percent']:.1f}%")
    print(f"  Average memory: {gpu_stats['avg_memory_mb']:.0f} MB")
    print(f"  Peak memory: {gpu_stats['max_memory_mb']:.0f} MB")
    if gpu_stats["avg_power_w"] is not None:
        print(f"  Average power: {gpu_stats['avg_power_w']:.1f}W")
        print(f"  Peak power: {gpu_stats['max_power_w']:.1f}W")

    # Validate GPU was actually used
    print(
        f"\n{'✓' if gpu_stats['gpu_active'] else '✗'} GPU Activity Detected: {gpu_stats['gpu_active']}"
    )
    if not gpu_stats["gpu_active"]:
        print("  Warning: GPU utilization was very low. Inference may not be using GPU.")

    # Cleanup
    print_section("Step 6: Cleanup")
    backend.cleanup()
    print("✓ Model unloaded and GPU memory freed")

    time.sleep(1)  # Give GPU time to update stats
    print("\nGPU State After Cleanup:")
    GPUMonitor.print_gpu_info(0)

    return gpu_stats["gpu_active"]


def main():
    """Run comprehensive validation."""
    print_banner("AI ENERGY BENCHMARKS - BACKEND VALIDATION WITH GPU MONITORING")

    results = {"vllm": None, "pytorch": None}

    # Validate vLLM backend
    try:
        results["vllm"] = validate_vllm_backend()
    except Exception as e:
        print(f"\n✗ vLLM validation failed with exception: {e}")
        import traceback

        traceback.print_exc()
        results["vllm"] = False

    # Validate PyTorch backend
    try:
        results["pytorch"] = validate_pytorch_backend()
    except Exception as e:
        print(f"\n✗ PyTorch validation failed with exception: {e}")
        import traceback

        traceback.print_exc()
        results["pytorch"] = False

    # Summary
    print_banner("VALIDATION SUMMARY")

    print("Backend Results:")
    print(f"  vLLM:    {'✓ PASS' if results['vllm'] else '✗ FAIL'} (GPU active: {results['vllm']})")
    print(
        f"  PyTorch: {'✓ PASS' if results['pytorch'] else '✗ FAIL'} (GPU active: {results['pytorch']})"
    )

    # Overall status
    all_passed = all(results.values())
    print(f"\nOverall Status: {'✓ ALL TESTS PASSED' if all_passed else '✗ SOME TESTS FAILED'}")

    if all_passed:
        print("\n✓ Both backends are functional and using GPU for inference")
        print("✓ Ready for Phase 1 development")
        return 0
    else:
        print("\n✗ Some backends failed validation")
        print("  Review errors above for details")
        return 1


if __name__ == "__main__":
    sys.exit(main())
