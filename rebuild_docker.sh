#!/bin/bash
# Rebuild Docker image with latest ai_energy_benchmarks changes
#
# This script:
# 1. Builds a new wheel with the latest code changes
# 2. Rebuilds the Docker image to include the new wheel

set -e

echo "========================================="
echo "Rebuilding Docker with latest changes"
echo "========================================="
echo ""

# Step 1: Build wheel
echo "Step 1: Building ai_energy_benchmarks wheel..."
cd /mnt/storage/src/ai_energy_benchmarks
./build_wheel.sh
echo ""

# Step 2: Rebuild Docker image
echo "Step 2: Rebuilding Docker image..."
cd /home/scott/src
./AIEnergyScore/build.sh
echo ""

echo "========================================="
echo "✓ Docker image rebuilt successfully!"
echo "========================================="
echo ""
echo "You can now run your tests with the updated code."
echo ""
