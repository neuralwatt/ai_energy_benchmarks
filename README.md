# AI Energy Benchmarks

A modular benchmarking framework for measuring AI model energy consumption and carbon emissions across different inference backends.

## Overview

AI Energy Benchmarks provides a flexible framework for measuring the energy footprint of AI models during inference. The framework supports multiple backends (vLLM, PyTorch) and integrates with CodeCarbon for accurate emissions tracking.

### Key Features

- **Multiple Backends**: Support for vLLM and PyTorch inference backends
- **Energy Tracking**: Integrated CodeCarbon metrics for energy consumption and CO₂ emissions
- **Flexible Configuration**: YAML-based configuration following Hydra/OmegaConf patterns
- **Dataset Integration**: Built-in support for HuggingFace datasets
- **Reasoning Format Support**: Automatic detection and formatting for reasoning-capable models (gpt-oss, DeepSeek, SmolLM, Qwen, etc.)
- **Modular Design**: Easy to extend with new backends, metrics, or reporters
- **Docker Support**: Containerized deployment for reproducible benchmarks

## Quick Start

### Prerequisites

- Python 3.10 or higher
- NVIDIA GPU with CUDA support (for GPU benchmarks)
- Docker (optional, for containerized deployment)

### Installation

1. **Create and activate a Python environment**:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
```

2. **Install the package**:
```bash
# Basic installation
pip install -e .

# With PyTorch support
pip install -e ".[pytorch]"

# With all development tools
pip install -e ".[all]"
```

### Building for Distribution

To create a wheel for use in other projects (e.g., AIEnergyScore):

```bash
# Build wheel
./build_wheel.sh

# Install from wheel
pip install dist/ai_energy_benchmarks-*.whl

# Install with optional dependencies
pip install 'dist/ai_energy_benchmarks-*.whl[pytorch]'
pip install 'dist/ai_energy_benchmarks-*.whl[all]'
```

The wheel can be copied into Docker images or shared with other projects without requiring the full source tree.

**Benefits of wheel distribution:**
- Smaller Docker build context (only wheel, not full source)
- Faster Docker builds (no copying unnecessary files)
- Cleaner separation between development and deployment
- Foundation for future PyPI distribution

### Running a Benchmark

#### Using the Shell Script

```bash
# Run with default configuration (gpt_oss_120b)
./run_benchmark.sh

# Run with custom configuration
./run_benchmark.sh configs/pytorch_test.yaml
```

#### Using Docker Compose

```bash
# Run benchmark in Docker
docker compose up
```

#### Using Python Directly

```python
from ai_energy_benchmarks.runner import run_benchmark_from_config

results = run_benchmark_from_config('configs/gpt_oss_120b.yaml')
```

## Configuration

Benchmarks are configured using YAML files. See `configs/` directory for examples.

### Configuration Structure

```yaml
name: benchmark_name

backend:
  type: vllm  # or pytorch
  device: cuda
  device_ids: [0]
  model: openai/gpt-oss-120b
  task: text-generation
  endpoint: "http://localhost:8000/v1"  # For vLLM

scenario:
  dataset_name: AIEnergyScore/text_generation
  text_column_name: text
  num_samples: 10
  truncation: true
  reasoning: false
  input_shapes:
    batch_size: 1
  generate_kwargs:
    max_new_tokens: 100
    min_new_tokens: 50

metrics:
  type: codecarbon
  enabled: true
  project_name: "my_benchmark"
  output_dir: "./emissions"
  country_iso_code: "USA"

reporter:
  type: csv
  output_file: "./results/benchmark_results.csv"

output_dir: ./benchmark_output
```

### Available Backends

#### vLLM Backend
For high-performance inference with vLLM serving:
```yaml
backend:
  type: vllm
  endpoint: "http://localhost:8000/v1"
  model: openai/gpt-oss-120b
```

Start vLLM server:
```bash
vllm serve openai/gpt-oss-120b --port 8000
```

#### PyTorch Backend
For direct PyTorch inference:
```yaml
backend:
  type: pytorch
  model: gpt2
  device: cuda
  device_ids: [0]  # Single GPU
```

For multi-GPU deployments with large models:
```yaml
backend:
  type: pytorch
  model: meta-llama/Llama-2-70b-hf
  device: cuda
  device_ids: [0, 1, 2, 3]  # Use 4 GPUs
  device_map: auto  # Automatically distribute model across GPUs
  torch_dtype: auto  # Auto-select optimal dtype
  # Optional: Set max memory per GPU to prevent OOM
  max_memory:
    0: "20GB"
    1: "20GB"
    2: "20GB"
    3: "20GB"
```

**Device Map Strategies:**
- `auto` (recommended): Automatically balance model layers across GPUs
- `balanced`: Evenly distribute layers across all devices
- `balanced_low_0`: Balance across GPUs, minimize GPU 0 usage
- `sequential`: Fill GPUs sequentially (GPU 0 first, then 1, etc.)

## Reasoning Format Support

AI Energy Benchmarks includes a unified reasoning format system that automatically detects and formats prompts for reasoning-capable models. This system eliminates the need for model-specific code changes.

### Supported Models

The framework automatically handles reasoning formats for:

| Model Family | Format Type | Enable Method | Example Usage |
|--------------|-------------|---------------|---------------|
| **gpt-oss** (OpenAI) | Harmony | `reasoning_effort: high/medium/low` | Structured system prompts |
| **SmolLM3** | System Prompt | `/think` flag | Prepended to prompts |
| **DeepSeek-R1** | Prefix | `<think>` tag | Prepended to prompts |
| **Qwen** | Parameter | `enable_thinking: true` | API parameter |
| **Hunyuan** | System Prompt | `/think` flag | Prepended to prompts |
| **Nemotron** | System Prompt | `/no_think` to disable | Default enabled |
| **EXAONE** | Parameter | `enable_thinking: true` | API parameter |
| **Phi** (Microsoft) | Parameter | `reasoning: true` | API parameter |
| **Gemma** (Google) | Parameter | `reasoning: true` | API parameter |

### Using Reasoning Parameters

#### Basic Example (vLLM Backend)

```yaml
backend:
  type: vllm
  endpoint: "http://localhost:8000/v1"
  model: openai/gpt-oss-20b

scenario:
  reasoning_params:
    reasoning_effort: high  # Options: low, medium, high
```

#### PyTorch Backend Example

```yaml
backend:
  type: pytorch
  model: HuggingFaceTB/SmolLM3-3B
  device: cuda

scenario:
  reasoning_params:
    enable_thinking: true
```

#### DeepSeek Example

```yaml
backend:
  type: pytorch
  model: deepseek-ai/DeepSeek-R1

scenario:
  reasoning_params:
    enable_thinking: true
    thinking_budget: 1000  # Token budget for reasoning
```

### Programmatic Usage

```python
from ai_energy_benchmarks.backends.vllm import VLLMBackend

# Backend automatically detects model and applies correct formatting
backend = VLLMBackend(
    endpoint="http://localhost:8000/v1",
    model="openai/gpt-oss-20b"
)

# Run inference with reasoning parameters
result = backend.run_inference(
    prompt="Explain quantum entanglement",
    reasoning_params={"reasoning_effort": "high"}
)
```

### How It Works

1. **Automatic Detection**: The `FormatterRegistry` automatically detects the model type from the model name
2. **Format Selection**: The appropriate formatter is selected from `ai_energy_benchmarks/config/reasoning_formats.yaml`
3. **Prompt Formatting**: The formatter modifies the prompt and/or generation parameters as needed
4. **Backward Compatibility**: Old `use_harmony` parameter still works with deprecation warnings

### Adding New Models

To add support for a new reasoning model, simply update the `reasoning_formats.yaml` file:

```yaml
families:
  new-model-family:
    patterns:
      - "company/new-model"
    type: system_prompt  # or harmony, parameter, prefix
    enable_flag: "/reason"
    disable_flag: "/no_reason"
    default_enabled: false
    description: "New reasoning model using /reason flags"
```

No code changes required! The system automatically picks up the new configuration.

### Migration from Legacy `use_harmony`

If you're using the old `use_harmony` parameter:

```python
# Old approach (deprecated)
backend = VLLMBackend(
    endpoint="http://localhost:8000/v1",
    model="openai/gpt-oss-20b",
    use_harmony=True  # Deprecated
)

# New approach (recommended)
backend = VLLMBackend(
    endpoint="http://localhost:8000/v1",
    model="openai/gpt-oss-20b"
    # Formatting auto-detected from model name
)
```

The old approach still works but will show deprecation warnings. The `use_harmony` parameter will be removed in v2.0.

### Formatter Architecture

The reasoning format system uses a registry-based architecture:

```
FormatterRegistry
├── HarmonyFormatter (gpt-oss models)
├── SystemPromptFormatter (SmolLM, Hunyuan, Nemotron)
├── ParameterFormatter (Qwen, EXAONE, DeepSeek)
└── PrefixFormatter (DeepSeek <think> tag)
```

Each formatter implements:
- `format_prompt()`: Modifies the prompt text
- `get_generation_params()`: Returns additional generation parameters

See `ai_energy_benchmarks/formatters/` for implementation details.

## Project Structure

```
ai_energy_benchmarks/
├── ai_energy_benchmarks/      # Main package
│   ├── backends/              # Inference backend implementations
│   │   ├── vllm.py           # vLLM backend
│   │   └── pytorch.py        # PyTorch backend
│   ├── formatters/           # Reasoning format handlers
│   │   ├── base.py           # Abstract formatter base
│   │   ├── harmony.py        # Harmony formatter (gpt-oss)
│   │   ├── system_prompt.py  # System prompt formatter
│   │   ├── parameter.py      # Parameter-based formatter
│   │   ├── prefix.py         # Prefix/suffix formatter
│   │   └── registry.py       # Formatter registry
│   ├── config/               # Configuration files
│   │   ├── parser.py         # Config parsing
│   │   └── reasoning_formats.yaml  # Model format registry
│   ├── datasets/             # Dataset loaders
│   ├── metrics/              # Metrics collectors (CodeCarbon)
│   ├── reporters/            # Result reporters (CSV)
│   ├── utils/                # Utility functions
│   └── runner.py             # Main benchmark runner
├── configs/                  # Example configurations
│   ├── gpt_oss_120b.yaml
│   ├── pytorch_test.yaml
│   └── pytorch_validation.yaml
├── tests/                    # Test suite
│   └── test_formatters.py    # Formatter tests
├── docs/                     # Documentation
├── examples/                 # Example scripts
├── results/                  # Benchmark results output
├── emissions/                # CodeCarbon emissions data
└── run_benchmark.sh          # Convenience runner script
```

## Development

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=ai_energy_benchmarks --cov-report=html
```

### Code Quality

The project uses:
- **Black** for code formatting
- **Ruff** for linting
- **MyPy** for type checking
- **Pre-commit** hooks for automated checks

```bash
# Install pre-commit hooks
pre-commit install

# Run linting
ruff check ai_energy_benchmarks/

# Run type checking
mypy ai_energy_benchmarks/
```

## Multi-GPU Support

The PyTorch backend includes comprehensive multi-GPU support powered by HuggingFace Accelerate:

### Automatic Model Distribution

Large models are automatically distributed across multiple GPUs using the `device_map` parameter:

```bash
# Run multi-GPU benchmark
./run_benchmark.sh configs/pytorch_multigpu.yaml
```

### GPU Metrics Collection

The framework automatically collects and reports per-GPU metrics:
- **GPU Utilization**: Percentage utilization per GPU
- **Memory Usage**: Used and total memory per GPU
- **Temperature**: GPU temperature (if available)
- **Power Draw**: Power consumption per GPU (if available)

Metrics are included in CSV output with columns like:
- `gpu_stats_gpu_0_utilization_percent`
- `gpu_stats_gpu_1_memory_used_mb`
- `gpu_stats_gpu_2_power_draw_w`

### Monitoring Multiple GPUs

```bash
# Monitor GPU usage during benchmark
watch -n 1 nvidia-smi

# Check specific GPUs
nvidia-smi -i 0,1,2,3
```

### Troubleshooting Multi-GPU

1. **OOM Errors**: Set `max_memory` per GPU in config
2. **GPU Not Found**: Verify GPUs with `nvidia-smi` and update `device_ids`
3. **Unbalanced Load**: Try different `device_map` strategies
4. **CUDA Errors**: Ensure CUDA is properly installed and GPUs are visible

## Output Files

Benchmarks generate several output files:

- **Results CSV**: Detailed benchmark results with per-GPU metrics (`results/`)
- **Emissions Data**: CodeCarbon emissions tracking (`emissions/`)
- **Logs**: Benchmark execution logs (`benchmark_output/`)

## Environment Variables

You can override configuration values using environment variables:

```bash
# Example: Override model and backend
BENCHMARK_BACKEND=pytorch \
BENCHMARK_MODEL=gpt2 \
./run_benchmark.sh
```

## Docker

### Building the Image

```bash
docker build -t ai-energy-benchmark .
```

### Running with Docker

```bash
docker run --gpus all \
  -e BENCHMARK_BACKEND=pytorch \
  -e BENCHMARK_MODEL=gpt2 \
  ai-energy-benchmark
```

### Docker Compose

```bash
# Start benchmark
docker compose up

# Clean up
docker compose down
```

## Troubleshooting

### Common Issues

1. **GPU Memory Issues**: Reduce batch size or number of samples in configuration
2. **vLLM Connection Errors**: Ensure vLLM server is running and endpoint is correct
3. **Dataset Download Fails**: Check internet connection and HuggingFace dataset availability
4. **Import Errors**: Ensure all dependencies are installed with `pip install -e ".[all]"`

### Debug Mode

Enable verbose logging by modifying the configuration or setting environment variables.

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests and linting
5. Submit a pull request

## License

MIT License - see LICENSE file for details.

## Citation

If you use this framework in your research, please cite:

```bibtex
@software{ai_energy_benchmarks,
  title={AI Energy Benchmarks},
  author={NeuralWatt},
  year={2025},
  url={https://github.com/neuralwatt/ai_energy_benchmarks}
}
```

## Support

- **Issues**: [GitHub Issues](https://github.com/neuralwatt/ai_energy_benchmarks/issues)
- **Documentation**: [GitHub Docs](https://github.com/neuralwatt/ai_energy_benchmarks/tree/main/docs)
- **Email**: info@neuralwatt.com
