#!/bin/bash
# Build wheel for ai_energy_benchmarks package

set -e

echo "Building ai_energy_benchmarks wheel..."

# Activate virtual environment if it exists
if [ -d ".venv" ]; then
    echo "Activating virtual environment..."
    source .venv/bin/activate
fi

# Check if build module is installed
if ! python3 -c "import build" 2>/dev/null; then
    echo "Installing build module..."
    pip install build
fi

# Clean previous builds
rm -rf dist/ build/ *.egg-info

# Build wheel and source distribution
python3 -m build

echo "✓ Wheel built successfully"
echo ""
echo "Wheel location: dist/ai_energy_benchmarks-$(cat VERSION.txt)-py3-none-any.whl"
echo ""
echo "Install with:"
echo "  pip install dist/ai_energy_benchmarks-*.whl"
echo ""
echo "Or install with extras:"
echo "  pip install 'dist/ai_energy_benchmarks-*.whl[pytorch]'"
echo "  pip install 'dist/ai_energy_benchmarks-*.whl[all]'"
