# Getting Started with AI Energy Benchmarks (POC)

This guide will help you get started with the POC implementation of the AI Energy Benchmarks framework.

## Prerequisites

### System Requirements

- Python 3.10 or higher
- NVIDIA GPU with CUDA support
- Docker and Docker Compose (for containerized deployment)
- 8GB+ RAM
- 50GB+ disk space

### Software Requirements

- vLLM server (for inference backend)
- Git
- pip or conda

## Installation

### Option 1: Standard Python Installation

```bash
# Clone the repository
cd /home/scott/src/ai_energy_benchmarks

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install package
pip install -e .

# Verify installation
python -c "import ai_energy_benchmarks; print(ai_energy_benchmarks.__version__)"
```

### Option 2: Development Installation

```bash
# Install with development dependencies
pip install -e ".[dev]"

# Install pre-commit hooks (optional)
pre-commit install
```

### Option 3: Docker Installation

```bash
# Build Docker image
docker build -f Dockerfile.poc -t ai_energy_benchmarks:poc .

# Verify image
docker images | grep ai_energy_benchmarks
```

## Running Your First Benchmark

### Step 1: Start vLLM Server

Open a terminal and start the vLLM server:

```bash
# Install vLLM if not already installed
pip install vllm

# Start vLLM with a model
vllm serve openai/gpt-oss-120b \
  --port 8000 \
  --gpu-memory-utilization 0.9

# Wait for "Application startup complete" message
```

### Step 2: Configure Benchmark

Create or modify the configuration file (`configs/gpt_oss_120b.yaml`):

```yaml
name: my_first_benchmark

backend:
  type: vllm
  endpoint: "http://localhost:8000/v1"
  model: openai/gpt-oss-120b

scenario:
  dataset_name: AIEnergyScore/text_generation
  num_samples: 5  # Start small for testing

metrics:
  enabled: true
  output_dir: "./emissions"

reporter:
  output_file: "./results/my_results.csv"
```

### Step 3: Run Benchmark

```bash
# Using shell script
./run_benchmark.sh configs/gpt_oss_120b.yaml

# Or using Python directly
python -c "
from ai_energy_benchmarks.runner import run_benchmark_from_config
results = run_benchmark_from_config('configs/gpt_oss_120b.yaml')
print('Benchmark completed!')
"
```

### Step 4: View Results

Results are saved in:
- `./results/my_results.csv` - Performance and energy metrics
- `./emissions/` - CodeCarbon emissions reports

Example CSV output:

```csv
timestamp,name,backend,model,total_prompts,successful_prompts,energy_wh,emissions_g_co2eq
2025-10-06T12:00:00,my_first_benchmark,vllm,openai/gpt-oss-120b,5,5,42.5,12.3
```

## Docker Deployment

### Using Docker Compose

```bash
# Set environment variables
export VLLM_ENDPOINT=http://host.docker.internal:8000/v1
export CONFIG_FILE=configs/gpt_oss_120b.yaml

# Run benchmark
docker compose -f docker-compose.poc.yml up

# View results
cat results/gpt_oss_120b_results.csv
```

### Using Docker Run

```bash
docker run --gpus all \
  -v $(pwd)/configs:/app/configs:ro \
  -v $(pwd)/results:/app/results \
  -v $(pwd)/emissions:/app/emissions \
  --network host \
  ai_energy_benchmarks:poc \
  ./run_benchmark.sh configs/gpt_oss_120b.yaml
```

## Common Issues

### vLLM Connection Errors

**Error**: `Backend validation failed`

**Solution**:
- Ensure vLLM is running: `curl http://localhost:8000/health`
- Check endpoint in config matches vLLM port
- Verify model is loaded in vLLM

### CodeCarbon Installation

**Error**: `codecarbon not installed`

**Solution**:
```bash
pip install codecarbon
```

### Permission Errors

**Error**: `Permission denied` for results/emissions directories

**Solution**:
```bash
mkdir -p results emissions benchmark_output
chmod 755 results emissions benchmark_output
```

### Docker GPU Access

**Error**: `GPU not accessible in Docker`

**Solution**:
- Install nvidia-container-toolkit
- Verify: `docker run --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi`

## Next Steps

- Review [Configuration Guide](./configuration.md)
- Explore [Backend Documentation](./backends.md)
- See [Examples](../examples/)
- Run tests: `pytest tests/`

## Getting Help

- Check the [README](../README.poc.md)
- Review test files in `tests/` for usage examples
- Open an issue on GitHub
