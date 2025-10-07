# Configuration Guide

This guide explains how to configure benchmarks in the AI Energy Benchmarks framework.

## Configuration Format

The framework uses a YAML-based configuration format inspired by HuggingFace's optimum-benchmark.

## Configuration Structure

### Complete Example

```yaml
# configs/example.yaml
name: my_benchmark

backend:
  type: vllm
  device: cuda
  device_ids: [0]
  model: openai/gpt-oss-120b
  endpoint: "http://localhost:8000/v1"

scenario:
  dataset_name: AIEnergyScore/text_generation
  text_column_name: text
  num_samples: 100
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
  region: null

reporter:
  type: csv
  output_file: "./results/results.csv"

output_dir: ./benchmark_output
```

## Configuration Sections

### Backend Configuration

Controls the inference backend:

```yaml
backend:
  type: vllm  # Backend type: vllm or pytorch (pytorch is stub in POC)
  device: cuda  # Device: cuda or cpu
  device_ids: [0]  # GPU device IDs to use
  model: openai/gpt-oss-120b  # Model name/path
  endpoint: "http://localhost:8000/v1"  # vLLM endpoint (vLLM only)
```

**Backend Types**:
- `vllm`: vLLM server backend (working in POC)
- `pytorch`: PyTorch local backend (stub in POC, Phase 2)

### Scenario Configuration

Controls the benchmark scenario and dataset:

```yaml
scenario:
  dataset_name: AIEnergyScore/text_generation  # HuggingFace dataset
  text_column_name: text  # Column containing prompts
  num_samples: 100  # Number of prompts to use
  truncation: true  # Enable truncation
  reasoning: false  # Enable reasoning mode (future)
  input_shapes:
    batch_size: 1  # Batch size for inference
  generate_kwargs:
    max_new_tokens: 100  # Max tokens to generate
    min_new_tokens: 50  # Min tokens to generate
```

### Metrics Configuration

Controls energy and performance metrics collection:

```yaml
metrics:
  type: codecarbon  # Metrics collector type
  enabled: true  # Enable/disable metrics
  project_name: "my_benchmark"  # Project name for tracking
  output_dir: "./emissions"  # Output directory for emissions data
  country_iso_code: "USA"  # Country code for carbon intensity
  region: null  # Specific region (e.g., "california")
```

**Supported Carbon Regions**:
- USA: null (uses US average) or "california", "texas", etc.
- EU: "france", "germany", etc.
- See [CodeCarbon docs](https://mlco2.github.io/codecarbon/) for full list

### Reporter Configuration

Controls results output:

```yaml
reporter:
  type: csv  # Reporter type: csv (only CSV in POC)
  output_file: "./results/results.csv"  # Output file path
```

## Configuration Overrides

### Programmatic Overrides

```python
from ai_energy_benchmarks.runner import run_benchmark_from_config

overrides = {
    'scenario': {
        'num_samples': 20
    },
    'backend': {
        'endpoint': 'http://custom-server:8000/v1'
    }
}

results = run_benchmark_from_config(
    'configs/base.yaml',
    overrides=overrides
)
```

### Environment Variable Overrides

You can use environment variables in config files:

```yaml
backend:
  endpoint: "${VLLM_ENDPOINT:-http://localhost:8000/v1}"
  model: "${MODEL_NAME:-openai/gpt-oss-120b}"
```

Then set environment variables:

```bash
export VLLM_ENDPOINT=http://my-server:8000/v1
export MODEL_NAME=meta-llama/Llama-3-70b
./run_benchmark.sh configs/example.yaml
```

## Validation

The framework validates configurations before running:

- Backend type must be 'vllm' or 'pytorch'
- vLLM backend requires endpoint
- num_samples must be >= 1
- Metrics type must be 'codecarbon'
- Reporter type must be 'csv'

Validation errors will stop the benchmark with a clear error message.

## Example Configurations

### Minimal Configuration

```yaml
name: minimal
backend:
  type: vllm
  model: openai/gpt-oss-120b
  endpoint: "http://localhost:8000/v1"
scenario:
  dataset_name: AIEnergyScore/text_generation
  num_samples: 5
```

### Full Production Configuration

```yaml
name: production_benchmark
backend:
  type: vllm
  device: cuda
  device_ids: [0, 1]  # Multi-GPU
  model: openai/gpt-oss-120b
  endpoint: "http://localhost:8000/v1"
scenario:
  dataset_name: AIEnergyScore/text_generation
  text_column_name: text
  num_samples: 1000
  generate_kwargs:
    max_new_tokens: 200
metrics:
  enabled: true
  project_name: "production_run_001"
  output_dir: "/data/emissions"
  country_iso_code: "USA"
  region: "california"
reporter:
  output_file: "/data/results/benchmark_results.csv"
```

## Best Practices

1. **Start Small**: Use small `num_samples` (5-10) for testing
2. **Set Region**: Specify `region` for accurate carbon intensity
3. **Organize Outputs**: Use dated directories for results
4. **Version Control**: Keep configs in git, exclude results
5. **Document Changes**: Add comments explaining non-standard settings

## See Also

- [Getting Started Guide](./getting_started.md)
- [Backend Documentation](./backends.md)
- [Example Configurations](../examples/)
