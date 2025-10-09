#!/bin/bash
# Wrapper script to run reasoning tests with proper venv activation

cd /home/scott/src/ai_energy_benchmarks

# Activate venv
source .venv/bin/activate

# Run the test
python3 ai_helpers/test_reasoning_levels.py "$@"
