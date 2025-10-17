#!/usr/bin/env python3
"""Test script for multi-GPU configuration and availability checking.

This script helps validate multi-GPU setup before running benchmarks.
"""

import sys

from ai_energy_benchmarks.utils import GPUMonitor


def main():
    """Test multi-GPU configuration."""
    print("=" * 60)
    print("Multi-GPU Configuration Test")
    print("=" * 60)

    # Check available GPUs
    print("\n1. Checking GPU availability...")

    # Try to detect number of GPUs
    try:
        import torch

        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            print(f"   PyTorch detected {gpu_count} GPU(s)")
            gpu_ids = list(range(gpu_count))
        else:
            print("   Warning: CUDA not available via PyTorch")
            gpu_ids = [0]  # Try GPU 0 anyway
    except ImportError:
        print("   PyTorch not installed, trying nvidia-smi...")
        gpu_ids = [0]  # Default to single GPU

    print(f"   Testing GPUs: {gpu_ids}")

    # Check multi-GPU availability
    print("\n2. Testing GPU accessibility...")
    availability = GPUMonitor.check_multi_gpu_available(gpu_ids)

    print(f"   Requested GPUs: {availability['requested_gpus']}")
    print(f"   Available GPUs: {availability['available_gpus']}")
    print(f"   Unavailable GPUs: {availability['unavailable_gpus']}")

    if availability["all_available"]:
        print("   ✓ All requested GPUs are available!")
    else:
        print("   ✗ Some GPUs are unavailable")
        for gpu_id, error in availability["errors"].items():
            print(f"     GPU {gpu_id}: {error}")

    # Get detailed GPU stats
    print("\n3. Collecting GPU statistics...")
    GPUMonitor.print_multi_gpu_info(availability["available_gpus"])

    # Test configuration recommendations
    print("\n4. Configuration Recommendations:")
    print("-" * 60)

    num_available = len(availability["available_gpus"])

    if num_available == 0:
        print("   No GPUs available for benchmarking.")
        print("   Please check CUDA installation and GPU drivers.")
        return 1
    elif num_available == 1:
        print("   Single GPU detected.")
        print("   Recommended config:")
        print("   ```yaml")
        print("   backend:")
        print("     type: pytorch")
        print("     device: cuda")
        print("     device_ids: [0]")
        print("     device_map: auto")
        print("   ```")
    else:
        print(f"   {num_available} GPUs detected - Multi-GPU setup available!")
        print("   Recommended config for large models:")
        print("   ```yaml")
        print("   backend:")
        print("     type: pytorch")
        print("     device: cuda")
        print(f"     device_ids: {availability['available_gpus']}")
        print("     device_map: auto  # Automatically balance across GPUs")
        print("     torch_dtype: auto")
        print("   ```")

        # Calculate suggested max_memory
        stats = GPUMonitor.get_multi_gpu_stats(availability["available_gpus"])
        if stats and any(s is not None for s in stats.values()):
            print("\n   Optional memory limits (leave ~10% buffer):")
            print("   ```yaml")
            print("   max_memory:")
            for gpu_id, gpu_stat in stats.items():
                if gpu_stat is not None:
                    # Suggest 90% of total memory
                    suggested_gb = int(gpu_stat.memory_total_mb * 0.9 / 1024)
                    print(f'     {gpu_id}: "{suggested_gb}GB"')
            print("   ```")

    print("\n" + "=" * 60)
    print("Test complete!")
    print("=" * 60)

    if availability["all_available"]:
        print("\nYou can now run multi-GPU benchmarks with:")
        print("  ./run_benchmark.sh configs/pytorch_multigpu.yaml")
        return 0
    else:
        print("\nPlease resolve GPU availability issues before benchmarking.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
