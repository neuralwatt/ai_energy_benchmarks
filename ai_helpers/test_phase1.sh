#!/bin/bash
# Phase 1 Testing Script
# Tests the wheel build and installation process

set -e

echo "=========================================="
echo "Phase 1: Internal Wheel Distribution Test"
echo "=========================================="
echo ""

# Step 1: Build wheel
echo "Step 1: Building wheel..."
cd /home/scott/src/ai_energy_benchmarks
./build_wheel.sh
echo ""

# Step 2: Verify wheel exists
echo "Step 2: Verifying wheel artifacts..."
if [ -f "dist/ai_energy_benchmarks-0.0.2-py3-none-any.whl" ]; then
    echo "✓ Wheel found: dist/ai_energy_benchmarks-0.0.2-py3-none-any.whl"
    ls -lh dist/
else
    echo "✗ Wheel not found!"
    exit 1
fi
echo ""

# Step 3: Create test environment
echo "Step 3: Creating test environment..."
rm -rf test_phase1_env
python3 -m venv test_phase1_env
source test_phase1_env/bin/activate
echo "✓ Test environment created"
echo ""

# Step 4: Install wheel
echo "Step 4: Installing wheel..."
pip install dist/ai_energy_benchmarks-*.whl > /dev/null 2>&1
echo "✓ Wheel installed successfully"
echo ""

# Step 5: Test imports
echo "Step 5: Testing imports..."
python3 << 'EOF'
try:
    # Test core module imports
    from ai_energy_benchmarks.runner import BenchmarkRunner
    from ai_energy_benchmarks.backends.base import Backend
    # Import submodules to verify they're packaged correctly
    import ai_energy_benchmarks.backends.vllm
    import ai_energy_benchmarks.backends.pytorch
    import ai_energy_benchmarks.config.parser
    import ai_energy_benchmarks.datasets.huggingface
    import ai_energy_benchmarks.metrics.codecarbon
    import ai_energy_benchmarks.reporters.csv_reporter
    print("✓ All critical imports successful")
except ImportError as e:
    print(f"✗ Import failed: {e}")
    exit(1)
EOF
echo ""

# Step 6: Verify package metadata
echo "Step 6: Verifying package metadata..."
pip show ai-energy-benchmarks | grep -E "(Name|Version|Summary|Author)"
echo ""

# Step 7: Cleanup
echo "Step 7: Cleaning up..."
deactivate
rm -rf test_phase1_env
echo "✓ Test environment cleaned up"
echo ""

echo "=========================================="
echo "Phase 1 Testing: ALL TESTS PASSED ✓"
echo "=========================================="
echo ""
echo "Summary:"
echo "  - Wheel built successfully"
echo "  - Clean installation verified"
echo "  - All imports functional"
echo "  - Package metadata correct"
echo ""
echo "Ready for AIEnergyScore integration!"
