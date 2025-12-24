# AI Energy Benchmarks

A modular benchmarking framework for measuring AI model energy consumption and carbon emissions across different inference backends.

This current release has tested support for pytorch backend and initial support for vllm backend.  Some features outlined may still be in development so please contact the maintainers if you have a questions.

## Overview

AI Energy Benchmarks provides a flexible, backend-agnostic framework for measuring the energy footprint of AI models during inference. The framework supports multiple backends and integrates with CodeCarbon for accurate emissions tracking.

**Key Features:**
- **Multiple Backends**: PyTorch for model comparison, vLLM for production deployment testing
- **Energy Tracking**: Integrated CodeCarbon metrics for energy consumption and CO₂ emissions
- **Flexible Configuration**: YAML-based configuration following Hydra/OmegaConf patterns
- **Dataset Integration**: Built-in support for HuggingFace datasets
- **Reasoning Format Support**: Automatic detection and formatting for reasoning-capable models (gpt-oss, DeepSeek, SmolLM, Qwen, etc.)
- **Multi-GPU Support**: Comprehensive multi-GPU support for large models
- **Modular Design**: Easy to extend with new backends, metrics, or reporters
- **Docker Support**: Containerized deployment for reproducible benchmarks

---

## Understanding Backends

The framework is built around a **backend-agnostic architecture** with two primary backends, each serving different use cases:

### PyTorch Backend: Model Comparison & Research

**Purpose**: Direct model inference for comparing different models head-to-head

**Key Characteristics:**
- ✅ Direct model loading from HuggingFace or local paths
- ✅ Full control over model configuration (quantization, device mapping, etc.)
- ✅ Multi-GPU support with automatic model sharding
- ✅ Measures raw model performance without serving overhead
- ✅ Ideal for controlled experiments

**Best For:**
- Comparing energy efficiency of different models (e.g., GPT-2 vs Llama vs Mistral)
- Testing model variants (quantized, pruned, distilled models)
- Research and development workflows
- Evaluating model optimizations
- Multi-model head-to-head comparisons

**Example Use Case:**
```bash
# Compare energy efficiency of a small model (Phi-2) vs large model (Llama-2-70B)
./run_benchmark.sh configs/pytorch_test.yaml        # Uses microsoft/phi-2 (2.7B)
./run_benchmark.sh configs/pytorch_multigpu.yaml   # Uses meta-llama/Llama-2-70b-hf
```

### vLLM Backend: Production Deployment Testing

**Purpose**: Connect to existing vLLM serving infrastructure to measure production workloads

**Key Characteristics:**
- ✅ Connects to running vLLM servers via HTTP
- ✅ Measures real production serving patterns
- ✅ Includes serving infrastructure overhead
- ✅ Tests production-like configurations
- ✅ No model loading required (uses existing server)

**Best For:**
- Benchmarking production vLLM deployments
- Measuring serving infrastructure efficiency
- Testing production workload patterns
- Optimizing deployment configurations
- Production capacity planning

**Example Use Case:**
```bash
# Start vLLM server (production config)
vllm serve openai/gpt-oss-120b --port 8000

# Benchmark the deployment
./run_benchmark.sh configs/gpt_oss_120b.yaml
```

### Choosing the Right Backend

| Use Case | Backend | Why |
|----------|---------|-----|
| Compare GPT-4 vs Llama 3 energy efficiency | **PyTorch** | Direct model comparison in controlled environment |
| Measure production vLLM deployment | **vLLM** | Real-world serving metrics with infrastructure overhead |
| Test quantized vs full-precision models | **PyTorch** | Need control over model loading and configuration |
| Benchmark serving infrastructure | **vLLM** | Production-like conditions and serving patterns |
| Multi-model evaluation (5+ models) | **PyTorch** | Easy model switching without server restarts |
| Production optimization and tuning | **vLLM** | Actual deployment metrics and configurations |
| Research paper experiments | **PyTorch** | Reproducible, controlled benchmarking |
| Capacity planning for production | **vLLM** | Real-world throughput and latency patterns |

**Important Note on Comparisons:**
- Results between PyTorch and vLLM backends are **not directly comparable**
- PyTorch measures raw model performance
- vLLM includes serving infrastructure overhead (batching, scheduling, HTTP, etc.)
- Always use the **same backend** for fair model comparisons

---

## Quick Start

### Prerequisites

**System Requirements:**
- Python 3.10 or higher
- NVIDIA GPU with CUDA support (for GPU benchmarks)
- Docker (optional, for containerized deployment)
- 8GB+ RAM
- 50GB+ disk space for models

**Software Dependencies:**
- For vLLM backend: vLLM server
- For PyTorch backend: PyTorch and transformers
- CodeCarbon (for emissions tracking)

### Installation

#### Option 1: Standard Installation

```bash
# Clone or navigate to the repository
cd ai_energy_benchmarks

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Basic installation (vLLM backend only)
pip install -e .

# With PyTorch backend support
pip install -e ".[pytorch]"

# With all dependencies (development + testing)
pip install -e ".[all]"

# Verify installation
python -c "import ai_energy_benchmarks; print('Installation successful!')"
```

#### Option 2: Install from PyPI (Production/Docker)

For production deployments or Docker images, install directly from PyPI:

```bash
# Basic installation (vLLM backend only)
pip install ai_energy_benchmarks

# With PyTorch backend support
pip install ai_energy_benchmarks[pytorch]

# With all dependencies (development + testing)
pip install ai_energy_benchmarks[all]
```

### Your First Benchmark

Choose your path based on your use case:

#### Option A: PyTorch Backend (Model Comparison)

No server setup required - direct model inference:

```bash
# Run benchmark with PyTorch backend
./run_benchmark.sh configs/pytorch_test.yaml

# View results
cat results/pytorch_test_results.csv
```

**What happened:**
1. Downloaded microsoft/phi-2 model from HuggingFace (2.7B parameters)
2. Ran inference on 3 test prompts from AIEnergyScore/text_generation dataset
3. Measured energy consumption and emissions (disabled in test config)
4. Saved results to CSV

#### Option B: vLLM Backend (Production Deployment)

Requires running vLLM server first:

```bash
# Terminal 1: Start vLLM server
vllm serve openai/gpt-oss-120b \
  --port 8000 \
  --gpu-memory-utilization 0.9

# Wait for "Application startup complete" message

# Terminal 2: Run benchmark
./run_benchmark.sh configs/gpt_oss_120b.yaml

# View results
cat results/gpt_oss_120b_results.csv
```

**What happened:**
1. Connected to running vLLM server
2. Sent prompts via HTTP API
3. Measured end-to-end serving performance
4. Tracked energy and emissions

### Understanding Your Results

Results are saved in CSV format with metrics like:

```csv
timestamp,name,backend,model,total_prompts,successful_prompts,energy_wh,emissions_g_co2eq,avg_latency_s
2025-10-27T12:00:00,pytorch_backend_test,pytorch,microsoft/phi-2,3,3,0.15,0.04,1.23
```

Key metrics:
- **energy_wh**: Energy consumed in watt-hours
- **emissions_g_co2eq**: CO₂ emissions in grams
- **total_prompts**: Number of prompts processed
- **avg_latency_s**: Average response time

---

## Usage Modes

The framework supports multiple ways to run benchmarks:

### 1. Shell Script Mode (Recommended)

Simplest way to run benchmarks:

```bash
# Run with default config (gpt_oss_120b.yaml - requires vLLM server)
./run_benchmark.sh

# Run with specific config
./run_benchmark.sh configs/pytorch_test.yaml

# Run with custom config path
./run_benchmark.sh /path/to/my/config.yaml
```

### 2. Python API Mode

Programmatic access for integration:

```python
from ai_energy_benchmarks.runner import run_benchmark_from_config

# Basic usage
results = run_benchmark_from_config('configs/pytorch_test.yaml')
print(f"Energy consumed: {results['summary']['total_energy_wh']} Wh")

# With configuration overrides
overrides = {
    'scenario': {
        'num_samples': 20  # Override num_samples
    },
    'backend': {
        'model': 'gpt2-medium'  # Override model
    }
}
results = run_benchmark_from_config('configs/base.yaml', overrides=overrides)
```

### 3. Docker Compose Mode

For containerized deployments:

**Standard Compose** (with integrated Ollama server):
```bash
# Set environment variables
export AI_MODEL=llama3.2
export GPU_MODEL=h100

# Run benchmark
docker compose up

# View results
cat benchmark_output/results.csv
```

**POC Compose** (with external vLLM server):
```bash
# Start vLLM server on host first
vllm serve openai/gpt-oss-120b --port 8000

# Set environment
export VLLM_ENDPOINT=http://host.docker.internal:8000/v1
export CONFIG_FILE=configs/gpt_oss_120b.yaml

# Run benchmark
docker compose -f docker-compose.poc.yml up

# View results
cat results/gpt_oss_120b_results.csv
```

### 4. Docker Run Mode

Direct Docker container execution:

```bash
# Build image
docker build -t ai-energy-benchmark .

# Run benchmark
docker run --gpus all \
  -v $(pwd)/configs:/app/configs:ro \
  -v $(pwd)/results:/app/results \
  -v $(pwd)/emissions:/app/emissions \
  --network host \
  ai-energy-benchmark \
  ./run_benchmark.sh configs/pytorch_test.yaml
```

---

## Configuration

Benchmarks are configured using YAML files. The framework follows a Hydra/OmegaConf-inspired configuration pattern.

### Configuration Structure

Complete example showing all sections:

```yaml
name: my_benchmark

backend:
  type: pytorch  # or vllm
  # ... backend-specific settings

scenario:
  dataset_name: AIEnergyScore/text_generation
  num_samples: 100
  # ... scenario settings

metrics:
  type: codecarbon
  enabled: true
  # ... metrics settings

reporter:
  type: csv
  output_file: "./results/results.csv"

output_dir: ./benchmark_output
```

### Backend Configuration

#### PyTorch Backend - For Model Comparison

**When to use:** Comparing models, research, development, controlled experiments

**Single GPU Configuration:**

```yaml
backend:
  type: pytorch
  model: gpt2  # HuggingFace model name or local path
  device: cuda
  device_ids: [0]  # Use GPU 0
  task: text-generation
```

**Supported Models:**
- Goal is to support most top models on hugging face.
- Small models: `gpt2`, `gpt2-medium`, `facebook/opt-125m`
- Medium models: `facebook/opt-1.3b`, `EleutherAI/gpt-neo-1.3B`
- Large models: `meta-llama/Llama-2-7b-hf`, `mistralai/Mistral-7B-v0.1`
- Very large models (multi-GPU): `meta-llama/Llama-2-70b-hf`, `tiiuae/falcon-180B`

**Multi-GPU Configuration** (for large models):

```yaml
backend:
  type: pytorch
  model: meta-llama/Llama-2-70b-hf
  device: cuda
  device_ids: [0, 1, 2, 3]  # Use 4 GPUs
  device_map: auto  # Automatically distribute model across GPUs
  torch_dtype: auto  # Auto-select optimal dtype (float16/bfloat16)

  # Optional: Limit memory per GPU to prevent OOM
  max_memory:
    0: "20GB"
    1: "20GB"
    2: "20GB"
    3: "20GB"
```

**Device Map Strategies:**

| Strategy | Description | Best For |
|----------|-------------|----------|
| `auto` | Automatically balance layers across GPUs | **Recommended** - works for most models |
| `balanced` | Evenly distribute layers | Models with uniform layer sizes |
| `balanced_low_0` | Balance across GPUs, minimize GPU 0 | When GPU 0 runs other processes |
| `sequential` | Fill GPUs sequentially (0 first, then 1, etc.) | Debugging or specific hardware configs |

**Advanced PyTorch Options:**

```yaml
backend:
  type: pytorch
  model: meta-llama/Llama-2-13b-hf
  device: cuda
  device_ids: [0, 1]

  # Model loading options
  torch_dtype: float16  # or bfloat16, float32
  load_in_8bit: false  # Enable 8-bit quantization
  load_in_4bit: false  # Enable 4-bit quantization
  trust_remote_code: false  # Allow custom model code

  # Memory management
  device_map: auto
  max_memory:
    0: "24GB"
    1: "24GB"

  # Performance tuning
  use_cache: true  # Enable KV cache
  pad_token_id: 0  # Set padding token
```

**Use Cases:**
- ✅ Compare energy efficiency of different model sizes
- ✅ Test quantized vs full-precision models
- ✅ Evaluate model variants (base vs instruction-tuned)
- ✅ Research experiments with controlled variables
- ✅ Multi-model benchmarking

#### vLLM Backend - For Production Deployments

**When to use:** Production benchmarks, serving infrastructure testing, deployment analysis

**Configuration:**

```yaml
backend:
  type: vllm
  endpoint: "http://localhost:8000/v1"
  model: openai/gpt-oss-120b  # Must match vLLM server model
```

**vLLM Server Setup:**

```bash
# Basic vLLM server
vllm serve openai/gpt-oss-120b --port 8000

# Production-like configuration
vllm serve nvidia/Llama-3.3-70B-Instruct-FP8 \
  --port 8000 \
  --tensor-parallel-size 4 \
  --gpu-memory-utilization 0.9 \
  --max-num-seqs 256 \
  --dtype float16

# With specific GPU devices
vllm serve meta-llama/Llama-2-70b-hf \
  --port 8000 \
  --tensor-parallel-size 2 \
  --pipeline-parallel-size 2
```

**Docker Network Configuration:**

When benchmarking from Docker container to host vLLM server:

```yaml
backend:
  type: vllm
  endpoint: "http://host.docker.internal:8000/v1"  # Docker → host
  model: openai/gpt-oss-120b
```

**Use Cases:**
- ✅ Benchmark production vLLM deployments
- ✅ Measure serving infrastructure efficiency
- ✅ Test production workload patterns
- ✅ Optimize vLLM configuration parameters
- ✅ Capacity planning for production

**Important Notes:**
- vLLM server must be running before benchmark starts
- Model name in config must match the server's loaded model
- Endpoint must be accessible from benchmark environment
- Results include serving overhead (batching, scheduling, HTTP)

---

### GenAI-Perf Load Profiling (ai-energy-profile CLI)

The `ai-energy-profile` CLI provides a streamlined interface for running load profiles using NVIDIA's genai-perf tool against vLLM or OpenAI-compatible endpoints.

**Installation:**
```bash
# Basic installation
pip install -e .

# With profiling dependencies (pandas for result formatting)
pip install -e ".[profiling]"

# Or from PyPI
pip install ai_energy_benchmarks[profiling]
```

**Basic Usage:**
```bash
# Run a light load test (20 requests)
ai-energy-profile --profile light --model my-model

# Run against a custom endpoint
ai-energy-profile --profile moderate --model my-model --endpoint http://my-server:8000/v1

# Run with reproducible inputs using a seed
ai-energy-profile --profile heavy --model my-model --seed 42
```

**Available Profiles:**

| Profile | Requests | Concurrency | Description |
|---------|----------|-------------|-------------|
| `light` | 20 | 2 | Light load - 10-20% GPU utilization |
| `moderate` | 40 | 4 | Moderate load - 40-50% GPU utilization |
| `heavy` | 80 | 8 | Heavy load - 70-80% GPU utilization |
| `stress` | 240 | 24 | Stress test - 90-100% GPU utilization |
| `multiphase` | 78 | varies | Multi-phase workload with variability |
| `pattern` | varies | varies | Multi-phase pattern test |
| `power_test` | varies | varies | Extended phases for power measurement |

**Authentication (--api-key):**

For authenticated API endpoints, use the `--api-key` flag:

```bash
# Connect to an authenticated API endpoint
ai-energy-profile --profile light \
  --model meta-llama/Llama-3.3-70B-Instruct \
  --endpoint https://api.neuralwatt.com/v1 \
  --api-key YOUR_API_KEY
```

The API key is passed as a Bearer token in the `Authorization` header.

**Endpoint Types (--endpoint-type):**

Different APIs support different endpoint types:

| Endpoint Type | API Path | Use For |
|---------------|----------|---------|
| `chat` (default) | `/v1/chat/completions` | OpenAI-compatible chat APIs |
| `completions` | `/v1/completions` | Legacy completions APIs |

```bash
# Use chat completions endpoint (default)
ai-energy-profile --profile light --model my-model --endpoint-type chat

# Use legacy completions endpoint
ai-energy-profile --profile light --model my-model --endpoint-type completions
```

**All CLI Options:**

```bash
ai-energy-profile --help

Options:
  --profile {light,moderate,heavy,stress,multiphase,pattern,power_test}
                        Load profile to use (default: moderate)
  --endpoint ENDPOINT   API endpoint URL (default: http://localhost:8000/v1)
  --model MODEL         Model name (required)
  --output-dir DIR      Output directory for results (default: ./benchmark_output)
  --seed SEED           Random seed for reproducible inputs and outputs
  --api-key API_KEY     API key for authenticated endpoints (Bearer token)
  --endpoint-type {chat,completions}
                        Endpoint type (default: chat)
  --run-id-suffix SUFFIX
                        Suffix to append to episode RunId for differentiation
  --prompts-file FILE   Path to custom prompts file
```

**Example: Testing Against Remote API:**

```bash
# Test against NeuralWatt API with authentication
ai-energy-profile --profile light \
  --model Qwen/Qwen3-Coder-480B-A35B-Instruct \
  --endpoint https://api.neuralwatt.com/v1 \
  --api-key sk-your-api-key-here \
  --seed 42
```

**Example: Multi-Phase Workload:**

```bash
# Run multi-phase profile (light → moderate → stress)
ai-energy-profile --profile multiphase \
  --model my-model \
  --endpoint http://localhost:8000/v1
```

---

### Scenario Configuration

Controls the benchmark workload and generation parameters:

```yaml
scenario:
  # Dataset configuration
  dataset_name: AIEnergyScore/text_generation  # HuggingFace dataset
  text_column_name: text  # Column containing prompts
  num_samples: 100  # Number of prompts to process
  truncation: true  # Truncate long prompts

  # Input configuration
  input_shapes:
    batch_size: 1  # Batch size for inference

  # Generation parameters
  generate_kwargs:
    max_new_tokens: 100  # Maximum tokens to generate
    min_new_tokens: 50   # Minimum tokens to generate
    temperature: 0.7     # Sampling temperature
    top_p: 0.9          # Nucleus sampling threshold
    top_k: 50           # Top-k sampling
    do_sample: true     # Enable sampling (vs greedy)
```

**Common Datasets:**
- `AIEnergyScore/text_generation` - General text generation prompts
- `openai/gsm8k` - Math reasoning tasks
- `tatsu-lab/alpaca` - Instruction following
- Your custom dataset on HuggingFace

**Workload Profiles:**

Light workload (testing):
```yaml
scenario:
  num_samples: 10
  generate_kwargs:
    max_new_tokens: 50
```

Medium workload:
```yaml
scenario:
  num_samples: 100
  generate_kwargs:
    max_new_tokens: 100
```

Heavy workload (production-like):
```yaml
scenario:
  num_samples: 1000
  generate_kwargs:
    max_new_tokens: 200
```

### Reasoning Parameters

The framework includes a unified reasoning format system that automatically detects and formats prompts for reasoning-capable models.

#### Supported Model Families

| Model Family | Format Type | Configuration | Example Models |
|--------------|-------------|---------------|----------------|
| **gpt-oss** (OpenAI) | Harmony | `reasoning_effort: high/medium/low` | `openai/gpt-oss-20b`, `openai/gpt-oss-120b` |
| **SmolLM3** (HuggingFace) | System Prompt | `enable_thinking: true` | `HuggingFaceTB/SmolLM3-3B` |
| **DeepSeek-R1** | Prefix + Parameter | `enable_thinking: true`, `thinking_budget: 1000` | `deepseek-ai/DeepSeek-R1` |
| **Qwen** (Alibaba) | Parameter | `enable_thinking: true` | `Qwen/Qwen2.5-72B-Instruct` |
| **Hunyuan** (Tencent) | System Prompt | `enable_thinking: true` | `tencent/Hunyuan-1.8B-Instruct` |
| **Nemotron** (NVIDIA) | System Prompt | `disable_thinking: true` to disable | `nvidia/Nemotron-*` (default enabled) |
| **EXAONE** (LG) | Parameter | `enable_thinking: true` | `LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct` |
| **Phi** (Microsoft) | Parameter | `enable_thinking: true` | `microsoft/phi-*` |
| **Gemma** (Google) | Parameter | `enable_thinking: true` | `google/gemma-*` |

#### Reasoning Configuration Examples

**gpt-oss Models (Harmony Format):**

```yaml
backend:
  type: vllm
  endpoint: "http://localhost:8000/v1"
  model: openai/gpt-oss-20b

scenario:
  reasoning_params:
    reasoning_effort: high  # Options: low, medium, high
```

**SmolLM3 (System Prompt):**

```yaml
backend:
  type: pytorch
  model: HuggingFaceTB/SmolLM3-3B
  device: cuda

scenario:
  reasoning_params:
    enable_thinking: true
```

**DeepSeek-R1 (Prefix + Parameter):**

```yaml
backend:
  type: pytorch
  model: deepseek-ai/DeepSeek-R1
  device: cuda

scenario:
  reasoning_params:
    enable_thinking: true
    thinking_budget: 1000  # Token budget for reasoning
```

**Qwen (Parameter-based):**

```yaml
backend:
  type: pytorch
  model: Qwen/Qwen2.5-72B-Instruct
  device: cuda
  device_ids: [0, 1, 2, 3]

scenario:
  reasoning_params:
    enable_thinking: true
```

#### How Reasoning Formats Work

1. **Automatic Detection**: The `FormatterRegistry` detects model type from model name
2. **Format Selection**: Appropriate formatter selected from `ai_energy_benchmarks/config/reasoning_formats.yaml`
3. **Prompt Formatting**: Formatter modifies prompt and/or generation parameters
4. **Backward Compatibility**: Legacy `use_harmony` parameter still works (deprecated)

**Works with both PyTorch and vLLM backends!**

#### Adding New Reasoning Models

To add support for a new reasoning model, simply update `reasoning_formats.yaml`:

```yaml
families:
  new-model-family:
    patterns:
      - "company/new-model"
      - "company/new-model-v2"
    type: system_prompt  # or harmony, parameter, prefix
    enable_flag: "/reason"
    disable_flag: "/no_reason"
    default_enabled: false
    description: "New reasoning model using /reason flags"
```

**No code changes required!** The system automatically picks up the configuration.

### Metrics Configuration

Controls energy and performance metrics collection via CodeCarbon:

```yaml
metrics:
  type: codecarbon
  enabled: true
  project_name: "my_benchmark"
  output_dir: "./emissions"
  country_iso_code: "USA"
  region: null  # or specific region like "california"
```

**Supported Carbon Regions:**

```yaml
# United States
country_iso_code: "USA"
region: null  # US average
# or region: "california", "texas", "new_york", etc.

# Europe
country_iso_code: "FRA"  # France
country_iso_code: "DEU"  # Germany
country_iso_code: "GBR"  # United Kingdom

# Other regions
country_iso_code: "CAN"  # Canada
country_iso_code: "CHN"  # China
country_iso_code: "IND"  # India
```

See [CodeCarbon documentation](https://mlco2.github.io/codecarbon/) for full list.

**Metrics Collected:**
- Energy consumption (kWh)
- CO₂ emissions (kg CO₂eq)
- GPU power draw (W)
- CPU power draw (W)
- RAM power draw (W)
- Carbon intensity of electricity grid (g CO₂/kWh)

### Reporter Configuration

Controls how results are output:

```yaml
reporter:
  type: csv  # Currently only CSV supported
  output_file: "./results/benchmark_results.csv"
```

**CSV Output Columns:**
- `timestamp` - ISO 8601 timestamp
- `name` - Benchmark name
- `backend` - Backend type (pytorch/vllm)
- `model` - Model name
- `total_prompts` - Total prompts processed
- `successful_prompts` - Successfully processed prompts
- `failed_prompts` - Failed prompts
- `energy_wh` - Energy consumed (Wh)
- `emissions_g_co2eq` - CO₂ emissions (g)
- `avg_latency_s` - Average latency (seconds)
- `throughput_prompts_per_sec` - Throughput
- `gpu_stats_*` - Per-GPU metrics (PyTorch backend only)

### Environment Variable Overrides

You can use environment variables in config files:

```yaml
backend:
  endpoint: "${VLLM_ENDPOINT:-http://localhost:8000/v1}"
  model: "${MODEL_NAME:-openai/gpt-oss-120b}"

scenario:
  num_samples: "${NUM_SAMPLES:-100}"
```

Then set environment variables:

```bash
export VLLM_ENDPOINT=http://my-server:8000/v1
export MODEL_NAME=meta-llama/Llama-3-70b
export NUM_SAMPLES=500

./run_benchmark.sh configs/example.yaml
```

Or inline:

```bash
VLLM_ENDPOINT=http://localhost:8001/v1 ./run_benchmark.sh configs/example.yaml
```

---

## Common Workflows

### Model Comparison Workflow (PyTorch Backend)

Compare energy efficiency of different models:

```bash
# Step 1: Create configs for each model
# configs/compare_phi2.yaml
name: phi2_comparison
backend:
  type: pytorch
  model: microsoft/phi-2
  device: cuda
  device_ids: [0]
scenario:
  num_samples: 100

# configs/compare_llama7b.yaml
name: llama7b_comparison
backend:
  type: pytorch
  model: meta-llama/Llama-2-7b-hf
  device: cuda
  device_ids: [0]
scenario:
  num_samples: 100

# Step 2: Run benchmarks
./run_benchmark.sh configs/compare_phi2.yaml
./run_benchmark.sh configs/compare_llama7b.yaml

# Step 3: Compare results
python -c "
import pandas as pd
phi2 = pd.read_csv('results/phi2_results.csv')
llama = pd.read_csv('results/llama7b_results.csv')
print('Phi-2 (2.7B) Energy:', phi2['energy_wh'].iloc[0], 'Wh')
print('Llama-7B Energy:', llama['energy_wh'].iloc[0], 'Wh')
"
```

**Multi-model comparison script:**

```bash
# Compare multiple models in one go
for model in "microsoft/phi-2" "HuggingFaceTB/SmolLM3-3B" "meta-llama/Llama-2-7b-hf"; do
  echo "Benchmarking $model..."
  BENCHMARK_MODEL=$model ./run_benchmark.sh configs/pytorch_test.yaml
done
```
<!-- 
### Production Deployment Workflow (vLLM Backend)

Benchmark production vLLM deployment:

```bash
# Step 1: Start vLLM server with production config
vllm serve nvidia/Llama-3.3-70B-Instruct-FP8 \
  --port 8000 \
  --tensor-parallel-size 4 \
  --max-num-seqs 256 \
  --gpu-memory-utilization 0.9

# Step 2: Run production workload benchmark
./run_benchmark.sh configs/production_benchmark.yaml

# Step 3: Analyze serving efficiency
cat results/production_results.csv

# Step 4: Optimize and re-test
vllm serve nvidia/Llama-3.3-70B-Instruct-FP8 \
  --port 8000 \
  --tensor-parallel-size 4 \
  --max-num-seqs 512  # Increased batch size
  --gpu-memory-utilization 0.95

./run_benchmark.sh configs/production_benchmark.yaml
```

### Quantization Comparison Workflow

Compare full-precision vs quantized models:

```yaml
# configs/llama_fp16.yaml
backend:
  type: pytorch
  model: meta-llama/Llama-2-7b-hf
  torch_dtype: float16

# configs/llama_8bit.yaml
backend:
  type: pytorch
  model: meta-llama/Llama-2-7b-hf
  load_in_8bit: true

# configs/llama_4bit.yaml
backend:
  type: pytorch
  model: meta-llama/Llama-2-7b-hf
  load_in_4bit: true
```

```bash
./run_benchmark.sh configs/llama_fp16.yaml
./run_benchmark.sh configs/llama_8bit.yaml
./run_benchmark.sh configs/llama_4bit.yaml
```

### Reasoning Model Testing

Test reasoning capabilities across effort levels:

```yaml
# configs/reasoning_test.yaml
backend:
  type: vllm
  endpoint: "http://localhost:8000/v1"
  model: openai/gpt-oss-20b

scenario:
  dataset_name: openai/gsm8k  # Math reasoning dataset
  num_samples: 50
```

```bash
# Test low effort
echo "reasoning_params:\n  reasoning_effort: low" >> configs/reasoning_low.yaml
./run_benchmark.sh configs/reasoning_low.yaml

# Test medium effort
echo "reasoning_params:\n  reasoning_effort: medium" >> configs/reasoning_medium.yaml
./run_benchmark.sh configs/reasoning_medium.yaml

# Test high effort
echo "reasoning_params:\n  reasoning_effort: high" >> configs/reasoning_high.yaml
./run_benchmark.sh configs/reasoning_high.yaml
```

### Multi-GPU Scaling Test

Test how model scales across GPUs:

```bash
# Test 1 GPU
cat > configs/scaling_1gpu.yaml <<EOF
backend:
  type: pytorch
  model: meta-llama/Llama-2-70b-hf
  device_ids: [0]
  device_map: auto
EOF
./run_benchmark.sh configs/scaling_1gpu.yaml

# Test 2 GPUs
cat > configs/scaling_2gpu.yaml <<EOF
backend:
  type: pytorch
  model: meta-llama/Llama-2-70b-hf
  device_ids: [0, 1]
  device_map: auto
EOF
./run_benchmark.sh configs/scaling_2gpu.yaml

# Test 4 GPUs
cat > configs/scaling_4gpu.yaml <<EOF
backend:
  type: pytorch
  model: meta-llama/Llama-2-70b-hf
  device_ids: [0, 1, 2, 3]
  device_map: auto
EOF
./run_benchmark.sh configs/scaling_4gpu.yaml
```

--- -->

## Understanding Results

### Output Files

Benchmarks generate several output files:

```
results/
  benchmark_results.csv       # Main results file
emissions/
  emissions.csv               # CodeCarbon emissions tracking
  emissions_TIMESTAMP.csv     # Per-run emissions
benchmark_output/
  benchmark.log               # Execution logs
  debug_info.json             # Debug information
```



## Project Structure

```
ai_energy_benchmarks/
├── ai_energy_benchmarks/          # Main package
│   ├── backends/                  # Inference backend implementations
│   │   ├── base.py               # Abstract backend base class
│   │   ├── vllm.py               # vLLM backend
│   │   └── pytorch.py            # PyTorch backend
│   ├── formatters/               # Reasoning format handlers
│   │   ├── base.py               # Abstract formatter base
│   │   ├── harmony.py            # Harmony formatter (gpt-oss)
│   │   ├── system_prompt.py      # System prompt formatter
│   │   ├── parameter.py          # Parameter-based formatter
│   │   ├── prefix.py             # Prefix/suffix formatter
│   │   └── registry.py           # Formatter registry
│   ├── config/                   # Configuration files
│   │   ├── parser.py             # Config parsing
│   │   └── reasoning_formats.yaml # Model format registry
│   ├── datasets/                 # Dataset loaders
│   │   └── loader.py             # HuggingFace dataset integration
│   ├── metrics/                  # Metrics collectors
│   │   └── codecarbon.py         # CodeCarbon integration
│   ├── reporters/                # Result reporters
│   │   └── csv_reporter.py       # CSV output
│   ├── utils/                    # Utility functions
│   │   ├── gpu.py                # GPU utilities
│   │   └── logging.py            # Logging setup
│   └── runner.py                 # Main benchmark runner
├── configs/                      # Example configurations
│   ├── gpt_oss_120b.yaml        # vLLM backend example
│   ├── pytorch_test.yaml         # PyTorch single GPU
│   ├── pytorch_multigpu.yaml     # PyTorch multi-GPU
│   └── pytorch_validation.yaml   # Validation config
├── tests/                        # Test suite
│   ├── unit/                     # Unit tests
│   ├── integration/              # Integration tests
│   └── test_formatters.py        # Formatter tests
├── results/                      # Benchmark results output
├── emissions/                    # CodeCarbon emissions data
├── ai_helpers/                   # Development and testing scripts
├── run_benchmark.sh              # Main runner script
├── build_wheel.sh                # Wheel building script
├── docker-compose.yml            # Standard Docker Compose
├── docker-compose.poc.yml        # POC Docker Compose
├── Dockerfile                    # Standard Dockerfile
├── Dockerfile.poc                # POC Dockerfile
├── setup.py                      # Package setup
├── pyproject.toml                # Project metadata
├── requirements.txt              # Dependencies
└── README.md                     # This file
```

**Key Modules:**

- **backends/**: Backend implementations (add new backends here)
- **formatters/**: Reasoning format handlers (extensible via config)
- **config/**: Configuration parsing and reasoning format registry
- **datasets/**: Dataset loading and preprocessing
- **metrics/**: Metrics collection (CodeCarbon, custom metrics)
- **reporters/**: Results output (CSV, JSON, etc.)
- **runner.py**: Main orchestration logic

---

## Development

### Setting Up Development Environment

```bash
# Clone repository
cd ai_energy_benchmarks

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install in development mode with all dependencies
pip install -e ".[all]"

# Install pre-commit hooks (optional but recommended)
pip install pre-commit
pre-commit install
```

**Development Dependencies:**
- pytest: Testing framework
- pytest-cov: Coverage reporting
- ruff: Linting
- mypy: Type checking
- black: Code formatting
- pre-commit: Git hooks

### Running Tests

#### All Tests

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run with coverage
pytest --cov=ai_energy_benchmarks --cov-report=html

# Open coverage report
open htmlcov/index.html  # macOS
xdg-open htmlcov/index.html  # Linux
```

#### Specific Test Categories

```bash
# Unit tests only
pytest tests/unit/

# Integration tests only
pytest tests/integration/

# Specific test file
pytest tests/unit/test_vllm_backend.py

# Specific test function
pytest tests/unit/test_vllm_backend.py::TestVLLMBackend::test_initialization

# Tests matching pattern
pytest -k "test_reasoning"
```

#### Test Markers

```bash
# Run only fast tests (skip slow integration tests)
pytest -m "not integration"

# Run only integration tests
pytest -m integration

# Run with specific markers
pytest -m "pytorch"
pytest -m "vllm"
```

#### Debugging Tests

```bash
# Show print statements
pytest -s

# Show full traceback
pytest --tb=long

# Drop into debugger on failure
pytest --pdb

# Stop on first failure
pytest -x
```

### Code Quality

The project uses multiple tools to ensure code quality:

#### Linting with Ruff

```bash
# Check all code
ruff check ai_energy_benchmarks/

# Check specific files
ruff check ai_energy_benchmarks/backends/

# Auto-fix issues
ruff check --fix ai_energy_benchmarks/

# Show all violations
ruff check --show-fixes ai_energy_benchmarks/
```

#### Type Checking with MyPy

```bash
# Type check all code
mypy ai_energy_benchmarks/

# Type check specific module
mypy ai_energy_benchmarks/backends/

# Strict mode
mypy --strict ai_energy_benchmarks/
```

#### Code Formatting with Ruff

```bash
# Check formatting
ruff format --check ai_energy_benchmarks/

# Format code
ruff format ai_energy_benchmarks/

# Format specific files
ruff format ai_energy_benchmarks/backends/pytorch.py
```

#### Pre-commit Hooks

Run all checks before committing:

```bash
# Install hooks
pre-commit install

# Run manually on all files
pre-commit run --all-files

# Run on specific files
pre-commit run --files ai_energy_benchmarks/backends/pytorch.py
```

**Pre-commit checks:**
- Ruff linting
- Ruff formatting
- MyPy type checking
- Trailing whitespace removal
- End-of-file fixer
- YAML validation

### Building for Distribution

#### Build Wheel

```bash
# Build wheel
./build_wheel.sh

# Output: dist/ai_energy_benchmarks-VERSION-py3-none-any.whl

# Install wheel
pip install dist/ai_energy_benchmarks-*.whl

# Install with optional dependencies
pip install 'dist/ai_energy_benchmarks-*.whl[pytorch]'
pip install 'dist/ai_energy_benchmarks-*.whl[all]'
```

#### Build Docker Images

```bash
# Standard image
docker build -t ai-energy-benchmark:latest .

# POC image
docker build -f Dockerfile.poc -t ai-energy-benchmark:poc .

# Multi-platform build
docker buildx build --platform linux/amd64,linux/arm64 -t ai-energy-benchmark:latest .
```

### Development Workflow

1. **Create feature branch**
   ```bash
   git checkout -b feature/my-feature
   ```

2. **Make changes and test**
   ```bash
   # Make changes
   vim ai_energy_benchmarks/backends/new_backend.py

   # Run tests
   pytest tests/

   # Check code quality
   ruff check ai_energy_benchmarks/
   mypy ai_energy_benchmarks/
   ```

3. **Format and lint**
   ```bash
   ruff format ai_energy_benchmarks/
   ruff check --fix ai_energy_benchmarks/
   ```

4. **Commit changes**
   ```bash
   git add .
   git commit -m "Add new backend"
   # Pre-commit hooks run automatically
   ```

5. **Push and create PR**
   ```bash
   git push origin feature/my-feature
   ```

---

## Docker Deployment

### Building Images

#### Standard Dockerfile

```bash
# Build image
docker build -t ai-energy-benchmark:latest .

# Build with specific tag
docker build -t ai-energy-benchmark:v1.0.0 .

# Build with build args
docker build \
  --build-arg PYTHON_VERSION=3.11 \
  -t ai-energy-benchmark:py311 .
```

#### POC Dockerfile

```bash
# Build POC image (lighter weight)
docker build -f Dockerfile.poc -t ai-energy-benchmark:poc .
```
### Docker Run Commands

#### Basic Docker Run

```bash
docker run --gpus all \
  -v $(pwd)/configs:/app/configs:ro \
  -v $(pwd)/results:/app/results \
  -v $(pwd)/emissions:/app/emissions \
  ai-energy-benchmark:latest \
  ./run_benchmark.sh configs/pytorch_test.yaml
```

#### Docker Run with Network Access

For vLLM backend connecting to host:

```bash
docker run --gpus all \
  --network host \
  -v $(pwd)/configs:/app/configs:ro \
  -v $(pwd)/results:/app/results \
  -e VLLM_ENDPOINT=http://localhost:8000/v1 \
  ai-energy-benchmark:latest \
  ./run_benchmark.sh configs/vllm_config.yaml
```

#### Docker Run with Environment Variables

```bash
docker run --gpus all \
  -e BENCHMARK_BACKEND=pytorch \
  -e BENCHMARK_MODEL=gpt2 \
  -e NUM_SAMPLES=50 \
  -v $(pwd)/results:/app/results \
  ai-energy-benchmark:latest
```

#### Interactive Docker Session

```bash
docker run --gpus all -it \
  -v $(pwd):/workspace \
  ai-energy-benchmark:latest \
  /bin/bash

# Inside container
cd /workspace
python -c "from ai_energy_benchmarks.runner import run_benchmark_from_config; run_benchmark_from_config('configs/test.yaml')"
```

### Docker Volume Mounting

**Read-only configs:**
```bash
-v $(pwd)/configs:/app/configs:ro
```

**Writable results:**
```bash
-v $(pwd)/results:/app/results
-v $(pwd)/emissions:/app/emissions
-v $(pwd)/benchmark_output:/app/benchmark_output
```

**Mount entire directory:**
```bash
-v $(pwd):/workspace
```

### Docker GPU Access

**All GPUs:**
```bash
--gpus all
```

**Specific GPUs:**
```bash
--gpus '"device=0,1"'  # GPUs 0 and 1
--gpus '"device=2"'     # GPU 2 only
```

**GPU memory limits:**
```bash
--gpus 'all,capabilities=compute,utility' \
--memory="32g" \
--memory-swap="32g"
```

---

## Extending the Framework

### Adding New Backends

To add a new backend (e.g., TensorRT-LLM, MLX):

1. **Create backend class** in `ai_energy_benchmarks/backends/`:

```python
# ai_energy_benchmarks/backends/tensorrt.py
from typing import Dict, Any, List
from .base import Backend

class TensorRTBackend(Backend):
    """TensorRT-LLM backend for optimized inference."""

    def __init__(
        self,
        model: str,
        device: str = "cuda",
        device_ids: List[int] = None,
        **kwargs
    ):
        super().__init__()
        self.model = model
        self.device = device
        self.device_ids = device_ids or [0]
        # Initialize TensorRT engine

    def validate_environment(self) -> bool:
        """Validate TensorRT is available."""
        try:
            import tensorrt_llm
            return True
        except ImportError:
            return False

    def load_model(self):
        """Load TensorRT engine."""
        # Implementation here
        pass

    def run_inference(
        self,
        prompt: str,
        reasoning_params: Dict[str, Any] = None,
        **generate_kwargs
    ) -> Dict[str, Any]:
        """Run inference with TensorRT."""
        # Implementation here
        pass

    def cleanup(self):
        """Clean up resources."""
        pass
```

2. **Register backend** in `ai_energy_benchmarks/runner.py`:

```python
from .backends.tensorrt import TensorRTBackend

BACKEND_REGISTRY = {
    'vllm': VLLMBackend,
    'pytorch': PyTorchBackend,
    'tensorrt': TensorRTBackend,  # Add here
}
```

3. **Use new backend** in config:

```yaml
backend:
  type: tensorrt
  model: meta-llama/Llama-2-7b-hf
  device: cuda
```

### Adding New Reasoning Formats

To add support for new reasoning models:

1. **Update** `ai_energy_benchmarks/config/reasoning_formats.yaml`:

```yaml
families:
  new-model-family:
    patterns:
      - "company/new-model"
      - "company/new-model-v2"
    type: system_prompt  # or harmony, parameter, prefix
    enable_flag: "/think"
    disable_flag: "/no_think"
    default_enabled: false
    system_prompt_template: "You are a helpful assistant. Use {flag} to enable reasoning."
    description: "New reasoning model"
```

2. **No code changes needed!** The formatter registry automatically picks up the config.

3. **Test the new format**:

```yaml
backend:
  type: pytorch
  model: company/new-model

scenario:
  reasoning_params:
    enable_thinking: true
```

### Adding New Metrics Collectors

To add custom metrics (e.g., network traffic, disk I/O):

1. **Create metrics class** in `ai_energy_benchmarks/metrics/`:

```python
# ai_energy_benchmarks/metrics/network.py
from typing import Dict, Any

class NetworkMetricsCollector:
    """Collect network traffic metrics."""

    def __init__(self, interface: str = "eth0"):
        self.interface = interface
        self.start_bytes = 0
        self.end_bytes = 0

    def start(self):
        """Start collecting metrics."""
        self.start_bytes = self._get_bytes_transferred()

    def stop(self) -> Dict[str, Any]:
        """Stop and return metrics."""
        self.end_bytes = self._get_bytes_transferred()
        return {
            'network_bytes_transferred': self.end_bytes - self.start_bytes,
            'interface': self.interface
        }

    def _get_bytes_transferred(self) -> int:
        """Get bytes transferred on interface."""
        # Implementation here
        pass
```

2. **Integrate in runner** (modify `runner.py`):

```python
from .metrics.network import NetworkMetricsCollector

# In BenchmarkRunner.run():
network_metrics = NetworkMetricsCollector()
network_metrics.start()
# ... run benchmark ...
metrics.update(network_metrics.stop())
```

### Adding New Reporters

To add output formats (e.g., JSON, database):

1. **Create reporter class** in `ai_energy_benchmarks/reporters/`:

```python
# ai_energy_benchmarks/reporters/json_reporter.py
import json
from typing import Dict, Any
from pathlib import Path

class JSONReporter:
    """Report results in JSON format."""

    def __init__(self, output_file: str):
        self.output_file = Path(output_file)
        self.output_file.parent.mkdir(parents=True, exist_ok=True)

    def report(self, results: Dict[str, Any]):
        """Write results to JSON file."""
        with open(self.output_file, 'w') as f:
            json.dump(results, f, indent=2)
```

2. **Register reporter** in config parser:

```python
REPORTER_REGISTRY = {
    'csv': CSVReporter,
    'json': JSONReporter,  # Add here
}
```

3. **Use in config**:

```yaml
reporter:
  type: json
  output_file: "./results/benchmark_results.json"
```

---

## Troubleshooting

### Backend-Specific Issues

#### PyTorch Backend Issues

**Problem: GPU Out of Memory (OOM)**

```
RuntimeError: CUDA out of memory. Tried to allocate XXX MiB
```

**Solutions:**
1. Reduce batch size:
   ```yaml
   scenario:
     input_shapes:
       batch_size: 1  # Minimum
   ```

2. Use multi-GPU:
   ```yaml
   backend:
     device_ids: [0, 1, 2, 3]
     device_map: auto
   ```

3. Set max memory per GPU:
   ```yaml
   backend:
     max_memory:
       0: "20GB"
       1: "20GB"
   ```

4. Use quantization:
   ```yaml
   backend:
     load_in_8bit: true  # or load_in_4bit: true
   ```

5. Reduce sequence length:
   ```yaml
   scenario:
     generate_kwargs:
       max_new_tokens: 50  # Reduce from 100+
   ```

**Problem: Multi-GPU Not Working**

```
ValueError: Model too large for single GPU
```

**Solutions:**
1. Check device_ids:
   ```bash
   nvidia-smi  # Verify GPU availability
   ```

2. Verify device_map:
   ```yaml
   backend:
     device_ids: [0, 1]
     device_map: auto  # Must be set for multi-GPU
   ```

3. Install accelerate:
   ```bash
   pip install accelerate
   ```

**Problem: Model Loading Errors**

```
OSError: model not found
```

**Solutions:**
1. Check model name:
   ```bash
   # Valid examples
   gpt2
   facebook/opt-1.3b
   meta-llama/Llama-2-7b-hf
   ```

2. Check HuggingFace access:
   ```bash
   huggingface-cli login
   ```

3. Verify model exists:
   ```bash
   python -c "from transformers import AutoModel; AutoModel.from_pretrained('gpt2')"
   ```

**Problem: CUDA Errors**

```
RuntimeError: CUDA error: device-side assert triggered
```

**Solutions:**
1. Check CUDA installation:
   ```bash
   nvidia-smi
   nvcc --version
   python -c "import torch; print(torch.cuda.is_available())"
   ```

2. Update PyTorch:
   ```bash
   pip install --upgrade torch torchvision torchaudio
   ```

3. Clear GPU memory:
   ```bash
   # Kill processes using GPU
   nvidia-smi
   kill -9 <PID>
   ```

#### vLLM Backend Issues

**Problem: vLLM Connection Errors**

```
Backend validation failed: Could not connect to vLLM endpoint
```

**Solutions:**
1. Verify vLLM server is running:
   ```bash
   curl http://localhost:8000/health
   # Expected: {"status": "ok"}
   ```

2. Check endpoint in config:
   ```yaml
   backend:
     endpoint: "http://localhost:8000/v1"  # Must include /v1
   ```

3. Test with curl:
   ```bash
   curl http://localhost:8000/v1/models
   ```

4. Check firewall:
   ```bash
   sudo ufw allow 8000
   ```

**Problem: Server Not Responding**

```
Timeout waiting for vLLM server
```

**Solutions:**
1. Check server logs:
   ```bash
   # In vLLM server terminal
   # Look for errors or warnings
   ```

2. Increase timeout:
   ```bash
   export VLLM_TIMEOUT=300  # 5 minutes
   ```

3. Restart vLLM:
   ```bash
   pkill -f vllm
   vllm serve MODEL --port 8000
   ```

**Problem: Model Mismatch**

```
Model name in config does not match server
```

**Solutions:**
1. Check server model:
   ```bash
   curl http://localhost:8000/v1/models
   ```

2. Update config to match:
   ```yaml
   backend:
     model: openai/gpt-oss-120b  # Must match server
   ```

**Problem: Docker to Host Connection**

```
Cannot connect to host vLLM server from Docker
```

**Solutions:**
1. Use host.docker.internal:
   ```yaml
   backend:
     endpoint: "http://host.docker.internal:8000/v1"
   ```

2. Or use host network:
   ```bash
   docker run --network host ...
   ```

3. Or get host IP:
   ```bash
   # On Linux
   ip addr show docker0 | grep inet

   # Use host IP in config
   endpoint: "http://172.17.0.1:8000/v1"
   ```

### Common Issues

**Problem: Dataset Download Fails**

```
ConnectionError: Could not download dataset
```

**Solutions:**
1. Check internet connection

2. Set HuggingFace cache:
   ```bash
   export HF_HOME=/path/to/cache
   export HF_DATASETS_CACHE=/path/to/cache/datasets
   ```

3. Pre-download dataset:
   ```python
   from datasets import load_dataset
   load_dataset("AIEnergyScore/text_generation")
   ```

4. Use local dataset:
   ```yaml
   scenario:
     dataset_name: /path/to/local/dataset
   ```

**Problem: Import Errors**

```
ModuleNotFoundError: No module named 'ai_energy_benchmarks'
```

**Solutions:**
1. Install package:
   ```bash
   pip install -e .
   ```

2. Verify installation:
   ```bash
   python -c "import ai_energy_benchmarks"
   ```

3. Check Python path:
   ```bash
   python -c "import sys; print(sys.path)"
   ```

**Problem: Permission Errors**

```
PermissionError: [Errno 13] Permission denied: 'results/'
```

**Solutions:**
1. Create directories:
   ```bash
   mkdir -p results emissions benchmark_output
   ```

2. Fix permissions:
   ```bash
   chmod 755 results emissions benchmark_output
   ```

3. Use different output path:
   ```yaml
   output_dir: /tmp/benchmark_output
   ```

**Problem: CodeCarbon Installation**

```
ImportError: codecarbon not installed
```

**Solutions:**
1. Install codecarbon:
   ```bash
   pip install codecarbon
   ```

2. Or disable metrics:
   ```yaml
   metrics:
     enabled: false
   ```

### Debug Mode

Enable verbose logging:

```bash
# Set log level
export LOG_LEVEL=DEBUG

# Run benchmark
./run_benchmark.sh configs/test.yaml
```

Or in Python:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

from ai_energy_benchmarks.runner import run_benchmark_from_config
results = run_benchmark_from_config('config.yaml')
```

Inspect outputs:

```bash
# View benchmark logs
cat benchmark_output/benchmark.log

# View emissions data
cat emissions/emissions.csv

# View results
cat results/benchmark_results.csv
```

### Docker-Specific Issues

**Problem: GPU Not Accessible in Docker**

```
RuntimeError: CUDA not available
```

**Solutions:**
1. Install nvidia-container-toolkit:
   ```bash
   # Ubuntu/Debian
   distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
   curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
   curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
     sudo tee /etc/apt/sources.list.d/nvidia-docker.list
   sudo apt-get update
   sudo apt-get install -y nvidia-container-toolkit
   sudo systemctl restart docker
   ```

2. Test GPU access:
   ```bash
   docker run --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
   ```

3. Check Docker version:
   ```bash
   docker --version  # Should be 19.03+
   ```

**Problem: Volume Permission Errors**

```
Permission denied: '/app/results'
```

**Solutions:**
1. Fix permissions on host:
   ```bash
   sudo chown -R $USER:$USER results/ emissions/
   ```

2. Run with user:
   ```bash
   docker run --user $(id -u):$(id -g) ...
   ```

**Problem: Network Configuration**

```
Cannot resolve host.docker.internal
```

**Solutions:**
1. Use host network (Linux):
   ```bash
   docker run --network host ...
   ```

2. Add host entry (Linux):
   ```bash
   docker run --add-host host.docker.internal:host-gateway ...
   ```

3. Use bridge network with host IP:
   ```bash
   docker run -e VLLM_ENDPOINT=http://172.17.0.1:8000/v1 ...
   ```

---

## Best Practices

### General Best Practices

1. **Start Small for Testing**
   ```yaml
   scenario:
     num_samples: 5  # Test with small dataset first
   ```

2. **Set Accurate Carbon Region**
   ```yaml
   metrics:
     country_iso_code: "USA"
     region: "california"  # More accurate emissions
   ```

3. **Organize Output Directories**
   ```bash
   results/
     2025-10-27/
       model_a/
       model_b/
     2025-10-28/
       ...
   ```

4. **Version Control Configs**
   ```bash
   git add configs/
   git commit -m "Add benchmark config for Model X"

   # But exclude results
   echo "results/" >> .gitignore
   echo "emissions/" >> .gitignore
   ```

5. **Document Configurations**
   ```yaml
   # configs/production.yaml
   name: production_benchmark
   # This config tests production workload with 1000 prompts
   # Expected runtime: 30 minutes
   # Expected energy: ~50 Wh
   scenario:
     num_samples: 1000
   ```

### Backend-Specific Best Practices

#### PyTorch Backend

1. **Use Multi-GPU for Large Models**
   ```yaml
   # Models > 13B parameters
   backend:
     device_ids: [0, 1, 2, 3]
     device_map: auto
   ```

2. **Set max_memory to Prevent OOM**
   ```yaml
   backend:
     max_memory:
       0: "22GB"  # Leave 2GB buffer on 24GB GPU
       1: "22GB"
   ```

3. **Choose Appropriate device_map**
   ```yaml
   # Default: auto (recommended)
   device_map: auto

   # For specific use cases:
   device_map: balanced        # Even distribution
   device_map: balanced_low_0  # Minimize GPU 0
   ```

4. **Monitor Per-GPU Metrics**
   ```bash
   # Check GPU balance after benchmark
   cat results/results.csv | grep gpu_stats

   # Look for:
   # - Similar utilization across GPUs
   # - Similar memory usage
   # - No GPU at 100% while others idle
   ```

5. **Use Quantization for Memory Constraints**
   ```yaml
   # 8-bit quantization (good balance)
   backend:
     load_in_8bit: true

   # 4-bit quantization (max memory savings)
   backend:
     load_in_4bit: true
   ```

#### vLLM Backend

1. **Always Start Server Before Benchmark**
   ```bash
   # Terminal 1
   vllm serve MODEL --port 8000
   # Wait for "Application startup complete"

   # Terminal 2
   ./run_benchmark.sh config.yaml
   ```

2. **Match Model Name to Server**
   ```bash
   # Server
   vllm serve openai/gpt-oss-120b

   # Config
   backend:
     model: openai/gpt-oss-120b  # MUST MATCH
   ```

3. **Use Production-Like vLLM Config**
   ```bash
   vllm serve MODEL \
     --tensor-parallel-size 4 \
     --max-num-seqs 256 \
     --gpu-memory-utilization 0.9 \
     --dtype float16
   ```

4. **Docker to Host Communication**
   ```yaml
   # When benchmark runs in Docker, server on host
   backend:
     endpoint: "http://host.docker.internal:8000/v1"
   ```

5. **Test Server Health First**
   ```bash
   # Before running benchmark
   curl http://localhost:8000/health
   curl http://localhost:8000/v1/models
   ```

### Multi-GPU Best Practices

1. **Check GPU Topology**
   ```bash
   nvidia-smi topo -m
   # Use GPUs with faster interconnect
   ```

2. **Balance Memory Usage**
   ```yaml
   backend:
     max_memory:
       0: "20GB"
       1: "20GB"
       2: "20GB"
       3: "20GB"
   ```

3. **Monitor During Benchmark**
   ```bash
   watch -n 1 nvidia-smi
   # Check for:
   # - Balanced utilization
   # - No thermal throttling
   # - Expected power draw
   ```

4. **Verify Model Fits**
   ```python
   # Estimate model size
   model_params = 70e9  # 70B parameters
   bytes_per_param = 2  # float16
   gb_needed = (model_params * bytes_per_param) / 1e9
   print(f"Need ~{gb_needed}GB across GPUs")
   ```

### Benchmarking Best Practices

1. **Warm-up Runs**
   ```yaml
   # First run may be slower (model loading, compilation)
   # Run twice and use second result
   scenario:
     num_samples: 100
   ```

2. **Control for Variables**
   ```yaml
   # Keep these constant for fair comparison:
   scenario:
     num_samples: 100         # Same across runs
     generate_kwargs:
       max_new_tokens: 100    # Same across runs
       temperature: 0.7       # Same across runs
   ```

3. **Use Same Backend for Comparisons**
   ```bash
   # ✅ Good: Compare PyTorch to PyTorch
   ./run_benchmark.sh configs/pytorch_test.yaml
   ./run_benchmark.sh configs/pytorch_multigpu.yaml

   # ❌ Bad: Compare PyTorch to vLLM
   ./run_benchmark.sh configs/pytorch_test.yaml
   ./run_benchmark.sh configs/gpt_oss_120b.yaml
   ```

4. **Document Environment**
   ```yaml
   # Save environment details with results
   # GPU model, driver version, CUDA version
   # PyTorch/vLLM version
   # System load, temperature
   ```

---

## Reference

### Configuration Schema

Complete YAML schema reference:

```yaml
# Required fields
name: string  # Benchmark name

# Backend configuration (required)
backend:
  type: string  # "pytorch" or "vllm"

  # Common fields
  model: string  # Model name or path
  device: string  # "cuda" or "cpu" (optional, default: "cuda")
  device_ids: list[int]  # GPU IDs (optional, default: [0])

  # PyTorch-specific
  torch_dtype: string  # "auto", "float16", "bfloat16", "float32" (optional)
  device_map: string  # "auto", "balanced", etc. (optional)
  max_memory: dict  # Per-GPU memory limits (optional)
  load_in_8bit: bool  # Enable 8-bit quantization (optional)
  load_in_4bit: bool  # Enable 4-bit quantization (optional)
  trust_remote_code: bool  # Allow custom code (optional)

  # vLLM-specific
  endpoint: string  # vLLM server endpoint (required for vLLM)

# Scenario configuration (required)
scenario:
  dataset_name: string  # HuggingFace dataset or path
  text_column_name: string  # Column with prompts (optional, default: "text")
  num_samples: int  # Number of prompts to process
  truncation: bool  # Truncate long prompts (optional, default: true)

  # Input configuration
  input_shapes:
    batch_size: int  # Batch size (optional, default: 1)

  # Generation parameters
  generate_kwargs:
    max_new_tokens: int  # Max tokens to generate (optional, default: 100)
    min_new_tokens: int  # Min tokens to generate (optional)
    temperature: float  # Sampling temperature (optional, default: 1.0)
    top_p: float  # Nucleus sampling (optional)
    top_k: int  # Top-k sampling (optional)
    do_sample: bool  # Enable sampling (optional, default: false)

  # Reasoning parameters (optional)
  reasoning_params:
    reasoning_effort: string  # "low", "medium", "high" (for Harmony)
    enable_thinking: bool  # Enable reasoning (for other models)
    thinking_budget: int  # Token budget (for DeepSeek)

# Metrics configuration (optional)
metrics:
  type: string  # "codecarbon" (default)
  enabled: bool  # Enable metrics (optional, default: true)
  project_name: string  # Project name (optional)
  output_dir: string  # Output directory (optional, default: "./emissions")
  country_iso_code: string  # Country code (optional, default: "USA")
  region: string  # Specific region (optional)

# Reporter configuration (optional)
reporter:
  type: string  # "csv" (default)
  output_file: string  # Output file path (optional)

# Output directory (optional)
output_dir: string  # Base output directory (default: "./benchmark_output")
```

### API Reference

#### BenchmarkRunner

Main benchmark orchestration class.

```python
from ai_energy_benchmarks.runner import BenchmarkRunner
from ai_energy_benchmarks.config.parser import BenchmarkConfig

# Create config
config = BenchmarkConfig()
config.name = "my_benchmark"
config.backend.type = "pytorch"
config.backend.model = "gpt2"
config.scenario.num_samples = 10

# Create runner
runner = BenchmarkRunner(config)

# Run benchmark
results = runner.run()

# Results structure
results = {
    'summary': {
        'name': str,
        'backend': str,
        'model': str,
        'total_prompts': int,
        'successful_prompts': int,
        'failed_prompts': int,
        'total_energy_wh': float,
        'total_emissions_g_co2eq': float,
        'avg_latency_s': float,
        'throughput_prompts_per_sec': float
    },
    'per_prompt_results': [...],
    'gpu_stats': {...}  # PyTorch only
}
```

#### run_benchmark_from_config

Helper function to run from config file.

```python
from ai_energy_benchmarks.runner import run_benchmark_from_config

# Basic usage
results = run_benchmark_from_config('configs/test.yaml')

# With overrides
overrides = {
    'scenario': {'num_samples': 20},
    'backend': {'model': 'gpt2-medium'}
}
results = run_benchmark_from_config('configs/test.yaml', overrides=overrides)
```

#### ConfigParser

Configuration parsing utilities.

```python
from ai_energy_benchmarks.config.parser import ConfigParser

# Load config
config = ConfigParser.load_config('configs/test.yaml')

# Load with overrides
overrides = {'scenario': {'num_samples': 20}}
config = ConfigParser.load_config_with_overrides('configs/test.yaml', overrides)

# Validate config
is_valid = ConfigParser.validate_config(config)
```

#### Backend Classes

**PyTorchBackend:**
```python
from ai_energy_benchmarks.backends.pytorch import PyTorchBackend

backend = PyTorchBackend(
    model="gpt2",
    device="cuda",
    device_ids=[0],
    torch_dtype="float16"
)
backend.validate_environment()
backend.load_model()
result = backend.run_inference("Hello world")
backend.cleanup()
```

**VLLMBackend:**
```python
from ai_energy_benchmarks.backends.vllm import VLLMBackend

backend = VLLMBackend(
    endpoint="http://localhost:8000/v1",
    model="openai/gpt-oss-120b"
)
backend.validate_environment()
result = backend.run_inference("Hello world")
```

### CLI Reference

#### run_benchmark.sh

```bash
./run_benchmark.sh [CONFIG_FILE]

# Default config
./run_benchmark.sh

# Specific config
./run_benchmark.sh configs/pytorch_test.yaml

# Custom path
./run_benchmark.sh /path/to/config.yaml
```

#### Environment Variables

```bash
# Backend configuration
BENCHMARK_BACKEND=pytorch|vllm
BENCHMARK_MODEL=model_name
VLLM_ENDPOINT=http://localhost:8000/v1

# Scenario configuration
NUM_SAMPLES=100
MAX_NEW_TOKENS=100

# Metrics configuration
COUNTRY_ISO_CODE=USA
REGION=california

# Output configuration
OUTPUT_DIR=/path/to/output
RESULTS_FILE=/path/to/results.csv

# Debugging
LOG_LEVEL=DEBUG|INFO|WARNING|ERROR
```

---

## Contributing

We welcome contributions! Here's how to get involved:

### How to Contribute

1. **Fork the repository**
   ```bash
   git clone https://github.com/yourusername/ai_energy_benchmarks.git
   cd ai_energy_benchmarks
   ```

2. **Create feature branch**
   ```bash
   git checkout -b feature/my-feature
   ```

3. **Set up development environment**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -e ".[all]"
   pre-commit install
   ```

4. **Make changes**
   - Write code
   - Add tests
   - Update documentation

5. **Run tests and checks**
   ```bash
   pytest
   ruff check ai_energy_benchmarks/
   mypy ai_energy_benchmarks/
   ruff format ai_energy_benchmarks/
   ```

6. **Commit changes**
   ```bash
   git add .
   git commit -m "Add feature X"
   # Pre-commit hooks run automatically
   ```

7. **Push and create PR**
   ```bash
   git push origin feature/my-feature
   # Create pull request on GitHub
   ```

### Code Standards

- **Python**: PEP 8 style guide
- **Type hints**: Use type hints for all functions
- **Docstrings**: Google-style docstrings
- **Tests**: Write tests for new features
- **Formatting**: Use ruff for formatting
- **Linting**: Pass ruff checks
- **Type checking**: Pass mypy checks

### Pull Request Process

1. **Update documentation** if adding features
2. **Add tests** for new functionality
3. **Ensure all checks pass** (tests, linting, type checking)
4. **Update CHANGELOG** if applicable
5. **Request review** from maintainers

### Areas for Contribution

- **New backends**: TensorRT-LLM, MLX, GGML, etc.
- **New metrics**: Network, disk I/O, memory bandwidth
- **New reporters**: JSON, database, visualization
- **New reasoning formats**: Support for new models
- **Performance improvements**: Optimization, caching
- **Documentation**: Examples, tutorials, guides
- **Testing**: More test coverage, edge cases

---

## License & Citation

### License

MIT License - see LICENSE file for details.

### Citation

If you use this framework in your research, please cite:

```bibtex
@software{ai_energy_benchmarks,
  title={AI Energy Benchmarks: A Framework for Measuring AI Model Energy Consumption},
  author={NeuralWatt},
  year={2025},
  url={https://github.com/neuralwatt/ai_energy_benchmarks},
  version={1.0.0}
}
```

### Acknowledgments

This framework builds upon:
- **CodeCarbon**: For emissions tracking ([Zenodo DOI: 10.5281/zenodo.17298293](https://zenodo.org/records/17298293))
- **HuggingFace**: For model and dataset ecosystems
- **vLLM**: For high-performance serving
- **PyTorch**: For deep learning infrastructure

---

## Support

### Getting Help

- **Documentation**: You're reading it!
- **GitHub Issues**: [Report bugs and request features](https://github.com/neuralwatt/ai_energy_benchmarks/issues)
- **Email**: info@neuralwatt.com
- **Community**: Join our discussions on GitHub

### Reporting Issues

When reporting issues, please include:

1. **System information**:
   ```bash
   python --version
   nvidia-smi
   pip list | grep -E "torch|vllm|codecarbon"
   ```

2. **Configuration file**: Your YAML config

3. **Error message**: Full error output

4. **Steps to reproduce**: How to trigger the issue

5. **Expected vs actual behavior**

### Feature Requests

We welcome feature requests! Please:
- Check existing issues first
- Describe use case clearly
- Explain why it's beneficial
- Provide examples if possible

---

## Changelog

### Version 1.0.0 (2025-10-27)

**Major Features:**
- ✅ PyTorch backend with multi-GPU support
- ✅ vLLM backend for production deployments
- ✅ Unified reasoning format system (9+ model families)
- ✅ CodeCarbon integration for emissions tracking
- ✅ CSV reporting with per-GPU metrics
- ✅ Docker and Docker Compose support
- ✅ Comprehensive documentation

**Supported Backends:**
- PyTorch (direct inference)
- vLLM (serving infrastructure)

**Supported Reasoning Models:**
- gpt-oss, DeepSeek-R1, SmolLM3, Qwen, Hunyuan, Nemotron, EXAONE, Phi, Gemma

**Known Limitations:**
- Only CSV reporter implemented
- Only CodeCarbon metrics collector
- No streaming support yet
- No batch inference optimization yet

---

Thank you for using AI Energy Benchmarks! We hope this framework helps you build more energy-efficient AI systems. 🌱
