#!/usr/bin/env python3
"""Verify POC installation and dependencies."""

import sys
import importlib.util


def check_import(module_name, package_name=None):
    """Check if a module can be imported."""
    if package_name is None:
        package_name = module_name

    spec = importlib.util.find_spec(module_name)
    if spec is None:
        print(f"✗ {package_name} - NOT INSTALLED")
        return False
    else:
        print(f"✓ {package_name} - installed")
        return True


def main():
    """Run installation verification."""
    print("=" * 50)
    print("AI Energy Benchmarks - Installation Verification")
    print("=" * 50)
    print()

    # Check core dependencies
    print("Checking core dependencies:")
    all_ok = True

    all_ok &= check_import("requests")
    all_ok &= check_import("omegaconf")
    all_ok &= check_import("yaml", "pyyaml")

    print()
    print("Checking optional dependencies:")

    check_import("datasets", "HuggingFace datasets")
    check_import("codecarbon")

    print()
    print("Checking development dependencies:")

    check_import("pytest")
    check_import("pytest_cov", "pytest-cov")

    print()
    print("Checking package installation:")

    try:
        import ai_energy_benchmarks

        print(f"✓ ai_energy_benchmarks - version {ai_energy_benchmarks.__version__}")
    except ImportError:
        print("✗ ai_energy_benchmarks - NOT INSTALLED")
        print("  Run: pip install -e .")
        all_ok = False

    print()
    print("Checking package components:")

    if check_import("ai_energy_benchmarks.backends.vllm"):
        try:
            from ai_energy_benchmarks.backends.vllm import VLLMBackend

            print("  ✓ VLLMBackend class available")
        except ImportError:
            print("  ✗ VLLMBackend class not available")
            all_ok = False

    if check_import("ai_energy_benchmarks.config.parser"):
        try:
            from ai_energy_benchmarks.config.parser import ConfigParser

            print("  ✓ ConfigParser class available")
        except ImportError:
            print("  ✗ ConfigParser class not available")
            all_ok = False

    if check_import("ai_energy_benchmarks.runner"):
        try:
            from ai_energy_benchmarks.runner import BenchmarkRunner

            print("  ✓ BenchmarkRunner class available")
        except ImportError:
            print("  ✗ BenchmarkRunner class not available")
            all_ok = False

    print()
    print("=" * 50)

    if all_ok:
        print("✓ Installation verification PASSED")
        print()
        print("Next steps:")
        print("  1. Start vLLM server: vllm serve openai/gpt-oss-120b")
        print("  2. Run benchmark: ./run_benchmark.sh configs/gpt_oss_120b.yaml")
        return 0
    else:
        print("✗ Installation verification FAILED")
        print()
        print("Install missing dependencies:")
        print("  pip install -e .")
        print("  pip install -e '.[dev]'")
        return 1


if __name__ == "__main__":
    sys.exit(main())
