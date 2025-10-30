#!/bin/bash
# Main benchmark runner script for POC

set -e

# Default configuration
CONFIG_FILE="${1:-configs/gpt_oss_120b.yaml}"

echo "========================================="
echo "AI Energy Benchmarks - POC"
echo "========================================="
echo ""
echo "Configuration: $CONFIG_FILE"
echo ""

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Configuration file not found: $CONFIG_FILE"
    exit 1
fi

# Run benchmark
python -c "
from ai_energy_benchmarks.runner import run_benchmark_from_config
import sys

try:
    results = run_benchmark_from_config('$CONFIG_FILE')
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
