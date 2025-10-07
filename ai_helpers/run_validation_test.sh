#!/bin/bash
# Run ai_energy_benchmarks validation test using PyTorch backend
# This matches the optimum-benchmark text_generation.yaml configuration

set -e

echo "========================================="
echo "ai_energy_benchmarks Validation Test"
echo "Using custom PyTorch backend"
echo "========================================="
echo ""

# Configuration
CONFIG_FILE="configs/pytorch_validation.yaml"

echo "Configuration: ${CONFIG_FILE}"
echo ""

# Check if virtual environment is activated
if [ -z "$VIRTUAL_ENV" ]; then
    echo "Activating virtual environment..."
    source .venv/bin/activate
fi

echo "Python environment: $(which python)"
echo "Python version: $(python --version)"
echo ""
echo "Running benchmark..."
echo ""

# Run benchmark
python -c "
from ai_energy_benchmarks.runner import run_benchmark_from_config
import sys

try:
    results = run_benchmark_from_config('${CONFIG_FILE}')
    print('\n✓ Benchmark completed successfully')
    sys.exit(0)
except Exception as e:
    print(f'\n✗ Benchmark failed: {e}')
    import traceback
    traceback.print_exc()
    sys.exit(1)
"

echo ""
echo "========================================="
echo "Benchmark Complete"
echo "========================================="
echo ""
echo "Results saved to:"
echo "  - results/pytorch_validation_results.csv"
echo "  - emissions/pytorch_validation/"
echo ""
