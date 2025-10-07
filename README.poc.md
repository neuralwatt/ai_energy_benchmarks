# AI Energy Benchmarks - POC Phase

**Version:** 0.0.1 (Proof of Concept)
**Status:** POC - Week 1 Implementation

A modular benchmarking framework for AI energy measurements, designed to consolidate benchmarking capabilities from multiple repositories into a unified, extensible system.

## Overview

This POC validates the approach for the benchmark consolidation plan by implementing:

- **vLLM Backend**: Minimal working implementation for high-performance inference
- **CodeCarbon Integration**: Comprehensive energy and emissions tracking
- **HuggingFace Datasets**: Load prompts from AIEnergyScore/text_generation dataset
- **Hydra-style Configuration**: Compatible with optimum-benchmark format
- **Docker Deployment**: Containerized execution

## Quick Start

### Prerequisites

- Python 3.10+
- vLLM server running (for inference)
- NVIDIA GPU (for energy measurement)
- CUDA-capable environment

### Installation

```bash
cd /home/scott/src/ai_energy_benchmarks

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install package
pip install -e .

# Install with dev dependencies
pip install -e ".[dev]"
```

### Running the POC

#### 1. Start vLLM Server

```bash
# In a separate terminal
vllm serve openai/gpt-oss-120b --port 8000
```

#### 2. Run Benchmark

```bash
# Using shell script
./run_benchmark.sh configs/gpt_oss_120b.yaml

# Or directly with Python
python -c "from ai_energy_benchmarks.runner import run_benchmark_from_config; \
    run_benchmark_from_config('configs/gpt_oss_120b.yaml')"
```

#### 3. View Results

Results are saved to:
- `./results/gpt_oss_120b_results.csv` - Benchmark metrics
- `./emissions/` - CodeCarbon energy reports

### Docker Deployment

```bash
# Build Docker image
docker build -f Dockerfile.poc -t ai_energy_benchmarks:poc .

# Run with Docker Compose
docker compose -f docker-compose.poc.yml up

# Or run directly
docker run --gpus all \
  -v $(pwd)/results:/app/results \
  -v $(pwd)/emissions:/app/emissions \
  --network host \
  ai_energy_benchmarks:poc
```

## Architecture

### Directory Structure

```
ai_energy_benchmarks/
├── ai_energy_benchmarks/          # Main package
│   ├── backends/                  # Backend implementations
│   │   ├── base.py               # Backend interface
│   │   ├── vllm.py               # vLLM backend (working)
│   │   └── pytorch.py            # PyTorch backend (stub)
│   ├── datasets/                  # Dataset loaders
│   │   ├── base.py               # Dataset interface
│   │   └── huggingface.py        # HuggingFace loader (working)
│   ├── metrics/                   # Metrics collectors
│   │   ├── base.py               # MetricsCollector interface
│   │   └── codecarbon.py         # CodeCarbon integration (working)
│   ├── reporters/                 # Results reporters
│   │   ├── base.py               # Reporter interface
│   │   └── csv_reporter.py       # CSV output (working)
│   ├── config/                    # Configuration management
│   │   └── parser.py             # Config parser (Hydra-style)
│   └── runner.py                  # Main benchmark runner
│
├── configs/                       # Configuration files
│   ├── gpt_oss_120b.yaml         # POC configuration
│   ├── backend/                   # Backend configs
│   │   ├── vllm.yaml
│   │   └── pytorch.yaml
│   └── scenario/                  # Scenario configs
│       └── energy_star.yaml
│
├── tests/                         # Test suite
│   ├── unit/                      # Unit tests
│   └── integration/               # Integration tests
│
├── run_benchmark.sh               # Main runner script
├── Dockerfile.poc                 # POC Docker image
├── docker-compose.poc.yml         # Docker Compose config
└── pyproject.toml                 # Package configuration
```

### Core Components

#### 1. Backends (`ai_energy_benchmarks/backends/`)

**vLLM Backend** (Working):
- Connects to vLLM OpenAI-compatible API
- Runs inference via HTTP requests
- Collects token usage and latency metrics

**PyTorch Backend** (Stub):
- Interface defined for Phase 2 implementation
- Will support local model inference

#### 2. Metrics (`ai_energy_benchmarks/metrics/`)

**CodeCarbon Collector** (Working):
- Tracks GPU, CPU, RAM energy consumption
- Calculates carbon emissions (CO₂eq)
- Supports regional carbon intensity
- Outputs to CSV and JSON

#### 3. Datasets (`ai_energy_benchmarks/datasets/`)

**HuggingFace Loader** (Working):
- Loads datasets from HuggingFace Hub
- Primary: `AIEnergyScore/text_generation`
- Supports sample limiting and column selection

#### 4. Reporters (`ai_energy_benchmarks/reporters/`)

**CSV Reporter** (Working):
- Outputs results to CSV files
- Flattens nested dictionaries
- Appends to existing files

## Configuration

### Configuration Format

Follows optimum-benchmark Hydra-style format:

```yaml
# configs/gpt_oss_120b.yaml
name: gpt_oss_120b_poc

backend:
  type: vllm
  device: cuda
  device_ids: [0]
  model: openai/gpt-oss-120b
  endpoint: "http://localhost:8000/v1"

scenario:
  dataset_name: AIEnergyScore/text_generation
  text_column_name: text
  num_samples: 10
  generate_kwargs:
    max_new_tokens: 100
    min_new_tokens: 50

metrics:
  type: codecarbon
  enabled: true
  project_name: "gpt_oss_120b_poc"
  output_dir: "./emissions"
  country_iso_code: "USA"

reporter:
  type: csv
  output_file: "./results/gpt_oss_120b_results.csv"
```

### Configuration Overrides

You can override configuration programmatically:

```python
from ai_energy_benchmarks.runner import run_benchmark_from_config

overrides = {
    'scenario': {'num_samples': 20},
    'backend': {'endpoint': 'http://custom:8000/v1'}
}

results = run_benchmark_from_config(
    'configs/gpt_oss_120b.yaml',
    overrides=overrides
)
```

## Testing

### Run Unit Tests

```bash
pytest tests/unit/ -v
```

### Run Integration Tests

```bash
pytest tests/integration/ -v
```

### Run All Tests with Coverage

```bash
pytest --cov=ai_energy_benchmarks --cov-report=html
```

## POC Success Criteria

- [x] vLLM backend implementation
- [x] CodeCarbon metrics collection
- [x] HuggingFace dataset integration
- [x] Hydra-style configuration
- [x] CSV results output
- [x] Docker deployment
- [x] Unit tests
- [x] Integration tests
- [ ] End-to-end validation with gpt-oss-120b on NVIDIA Pro 6000
- [ ] Integration with AIEnergyScore (pending)
- [ ] Integration with neuralwatt_cloud (pending)

## Known Limitations (POC Phase)

1. **PyTorch Backend**: Stub only - implementation in Phase 2
2. **Load Generators**: Simple sequential inference - genai-perf in Phase 2
3. **Metrics**: Only CodeCarbon - plugin architecture for Phase 3
4. **Scenarios**: Single scenario only - multi-scenario in Phase 4
5. **Reporters**: CSV only - ClickHouse, MLflow in Phase 4

## Next Steps

After POC validation:

1. **Phase 1 (Weeks 2-3)**: Enhance foundation with production-ready tooling
2. **Phase 2 (Weeks 4-6)**: Implement PyTorch backend and genai-perf load generator
3. **Phase 3 (Weeks 7-8)**: Enhance dataset and metrics collection
4. **Phase 4 (Weeks 9-11)**: Migrate neuralwatt_cloud capabilities
5. **Phase 5 (Weeks 12-13)**: AIEnergyScore compatibility
6. **Phase 6 (Weeks 14-15)**: Testing, documentation, release

## Contributing

This is a POC implementation. For contributions, please:

1. Run tests: `pytest`
2. Check linting: `ruff check .`
3. Format code: `black .`
4. Update documentation

## License

MIT License - See LICENSE file

## References

- [Benchmark Consolidation Plan](/home/scott/src/neuralwatt_cloud/design/benchmark_consolidation_plan.md)
- [HuggingFace optimum-benchmark](https://github.com/huggingface/optimum-benchmark)
- [CodeCarbon](https://github.com/mlco2/codecarbon)
- [vLLM Documentation](https://docs.vllm.ai/)
- [AIEnergyScore Dataset](https://huggingface.co/datasets/AIEnergyScore/text_generation)

## Contact

For questions or issues, please open an issue on GitHub or contact the NeuralWatt team.
