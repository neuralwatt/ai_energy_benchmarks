# Running Benchmarks Without Optional Dependencies

**Problem:** The full benchmark runner (`run_benchmark.sh`) requires optional dependencies that aren't installed:
- `datasets` (HuggingFace datasets)
- `omegaconf` (Hydra configuration)
- `codecarbon` (Energy metrics)

**Solution:** Use the simplified runner that works with minimal dependencies.

## Quick Start (Works Now)

### Run Simple Benchmark

```bash
cd /home/scott/src/ai_energy_benchmarks

# Run with defaults (10 prompts)
python3 run_simple_benchmark.py

# Run with custom number of prompts
python3 run_simple_benchmark.py --num-samples 5

# Run with custom output file
python3 run_simple_benchmark.py --output ./results/my_benchmark.csv

# Full options
python3 run_simple_benchmark.py \
  --model openai/gpt-oss-120b \
  --endpoint http://localhost:8000/v1 \
  --num-samples 20 \
  --output ./results/benchmark.csv
```

### What It Does

✅ **Works without optional dependencies**
✅ Tests vLLM backend with real inference
✅ Uses built-in test prompts (no dataset download needed)
✅ Saves results to CSV
✅ Reports throughput and latency metrics

### Example Output

```
======================================================================
 BENCHMARK RESULTS
======================================================================
Total prompts:       5
Successful:          5
Failed:              0
Duration:            2.50s
Avg latency:         0.501s
Total tokens:        806
Throughput:          322.01 tok/s
======================================================================
```

## Installing Full Dependencies

To use the full benchmark runner with all features:

### Option 1: Install with pip (if available)

```bash
# Install from requirements.txt
pip install -r requirements.txt

# Or install individually
pip install datasets>=2.14.0 omegaconf>=2.3.0 codecarbon>=2.3.0
```

### Option 2: Install with conda

```bash
conda install -c conda-forge datasets omegaconf codecarbon
```

### Option 3: Install in a virtual environment

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run full benchmark
./run_benchmark.sh configs/gpt_oss_120b.yaml
```

### Option 4: Use Docker

The Docker image includes all dependencies:

```bash
docker build -f Dockerfile.poc -t ai_energy_benchmarks:poc .
docker run --gpus all --network host ai_energy_benchmarks:poc
```

## Feature Comparison

| Feature | Simple Runner | Full Runner |
|---------|---------------|-------------|
| **Dependencies** | requests, pyyaml only | datasets, omegaconf, codecarbon |
| **Prompts** | Built-in test prompts | HuggingFace datasets |
| **Configuration** | Command-line args | YAML config files |
| **Energy Metrics** | ❌ No | ✅ CodeCarbon |
| **CSV Output** | ✅ Yes | ✅ Yes |
| **Works Now** | ✅ Yes | ⏳ Needs dependencies |

## After Installing Dependencies

Once dependencies are installed, you can use the full runner:

```bash
# Full benchmark with all features
./run_benchmark.sh configs/gpt_oss_120b.yaml

# Or with Python
python3 -c "
from ai_energy_benchmarks.runner import run_benchmark_from_config
results = run_benchmark_from_config('configs/gpt_oss_120b.yaml')
"
```

## Current Validation Results

Using the simple runner, we confirmed:

✅ **vLLM Backend Works**
- 5 prompts completed successfully
- 0 failures
- Average latency: 0.501s
- Throughput: 322 tokens/second

✅ **GPU Activity Confirmed** (from earlier validation)
- Memory: 91.6% utilization (model on GPU)
- Power: 15W → 127W during inference
- Significant GPU load verified

## Files Created

- **`run_simple_benchmark.py`** - Simple runner without optional deps
- **`requirements.txt`** - Updated with all dependencies
- **`RUNNING_WITHOUT_DEPENDENCIES.md`** - This guide

## Summary

**Current Status:**
- ✅ Simple benchmark works now (no installation needed)
- ✅ vLLM backend validated with GPU confirmation
- ⏳ Full benchmark needs dependencies installed

**Recommendation:**
- Use `run_simple_benchmark.py` for immediate testing
- Install dependencies when ready for full features (datasets, energy metrics)
