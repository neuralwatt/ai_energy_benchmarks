# AI Energy Benchmarks

A modular benchmarking framework for measuring AI model energy consumption and carbon emissions across different inference backends.

## Overview

AI Energy Benchmarks provides a flexible framework for measuring the energy footprint of AI models during inference. The framework supports multiple backends (vLLM, PyTorch) and integrates with CodeCarbon for accurate emissions tracking.

### Key Features

- **Multiple Backends**: Support for vLLM and PyTorch inference backends
- **Energy Tracking**: Integrated CodeCarbon metrics for energy consumption and CO₂ emissions
- **Flexible Configuration**: YAML-based configuration following Hydra/OmegaConf patterns
- **Dataset Integration**: Built-in support for HuggingFace datasets
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
  device_ids: [0]
```

## Project Structure

```
ai_energy_benchmarks/
├── ai_energy_benchmarks/      # Main package
│   ├── backends/              # Inference backend implementations
│   │   ├── vllm.py           # vLLM backend
│   │   └── pytorch.py        # PyTorch backend
│   ├── config/               # Configuration parsing
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

## Output Files

Benchmarks generate several output files:

- **Results CSV**: Detailed benchmark results (`results/`)
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
