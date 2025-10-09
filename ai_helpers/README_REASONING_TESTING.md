# Reasoning Model Testing for AIEnergyScore

This directory contains tests and utilities for validating reasoning model support across different benchmark engines.

## Overview

The reasoning model support allows testing AI models with different "thinking" or "reasoning" effort levels, which can impact:
- Inference-time compute requirements
- Energy consumption
- Latency and throughput
- Response quality

## Supported Reasoning Parameters

### Configuration Format

```yaml
scenario:
  reasoning: True  # Enable reasoning mode
  reasoning_params:
    reasoning_effort: high  # Options: low, medium, high
    # Additional model-specific parameters:
    # thinking_budget: 1000  # Token budget for reasoning
    # cot_depth: 3  # Chain-of-thought depth
```

### Backend Support

#### PyTorch Backend
- Passes reasoning parameters directly to `model.generate()`
- Parameters are model-specific (e.g., gpt-oss-20b, DeepSeek-R1, Llama 4)
- Supports: `reasoning_effort`, custom parameters

#### vLLM Backend
- Translates reasoning parameters to OpenAI API `extra_body`
- Compatible with vLLM's model-specific extensions
- Supports: Same parameters as PyTorch

## Test Files

### Configuration Files (AIEnergyScore)

Located in `/home/scott/src/AIEnergyScore/`:

1. **text_generation_gptoss_reasoning_low.yaml**
   - Reasoning effort: low
   - 10 samples (testing)
   - Model: openai/gpt-oss-20b

2. **text_generation_gptoss_reasoning_medium.yaml**
   - Reasoning effort: medium
   - 10 samples (testing)
   - Model: openai/gpt-oss-20b

3. **text_generation_gptoss_reasoning_high.yaml**
   - Reasoning effort: high
   - 10 samples (testing)
   - Model: openai/gpt-oss-20b

### Test Scripts

1. **test_reasoning_config.py**
   - Unit tests for reasoning configuration parsing
   - Validates ScenarioConfig with reasoning params
   - Fast execution (no model loading)

2. **test_reasoning_levels.py**
   - Full integration test across benchmark engines
   - Tests PyTorch, vLLM, and optimum-benchmark
   - Compares results and analyzes variations

## Running Tests

### Quick Config Validation

```bash
cd /home/scott/src/ai_energy_benchmarks
source .venv/bin/activate
python3 ai_helpers/test_reasoning_config.py
```

Expected output: `ALL TESTS PASSED ✓`

### Full Integration Tests

**Prerequisites:**
- GPU available (for PyTorch backend)
- vLLM server running (optional, for vLLM backend tests)
- gpt-oss-20b model downloaded

**Run tests:**

```bash
cd /home/scott/src/ai_energy_benchmarks
source .venv/bin/activate

# Without vLLM (PyTorch + optimum-benchmark only)
python3 ai_helpers/test_reasoning_levels.py

# With vLLM endpoint
export VLLM_ENDPOINT="http://localhost:8000/v1"
python3 ai_helpers/test_reasoning_levels.py
```

**Expected results:**
- Test creates `./test_results/reasoning_levels/` directory
- Runs benchmarks with low/medium/high reasoning efforts
- Generates comparison report
- Saves `test_summary.json` with all results

### Test Output Structure

```
test_results/reasoning_levels/
├── pytorch_low/
│   ├── benchmark_report.json
│   ├── benchmark_results.csv
│   └── emissions/
├── pytorch_medium/
│   └── ...
├── pytorch_high/
│   └── ...
├── vllm_low/ (if vLLM tested)
│   └── ...
├── optimum_low/
│   └── ...
└── test_summary.json  # Aggregated results
```

## Expected Behavior

### Successful Reasoning Support

If reasoning levels are working correctly, you should observe:

1. **Latency Variation**: Higher reasoning effort → longer latency
   - Low: Baseline latency
   - Medium: +20-50% latency increase
   - High: +50-200% latency increase (model-dependent)

2. **Energy Variation**: Higher reasoning effort → more energy
   - Proportional to latency increase
   - Visible in CodeCarbon metrics

3. **Consistency Across Engines**: All three backends should show similar patterns:
   - PyTorch backend
   - vLLM backend
   - optimum-benchmark (AIEnergyScore)

### If Reasoning Is Not Supported

If the model doesn't support reasoning parameters:
- Latency will be similar across all effort levels
- Energy consumption will be consistent
- No errors (parameters silently ignored by model)

## Troubleshooting

### Configuration Errors

If config tests fail:
```bash
# Check imports
cd /home/scott/src/ai_energy_benchmarks
source .venv/bin/activate
python3 -c "from ai_energy_benchmarks.config.parser import ScenarioConfig; print('OK')"
```

### Model Loading Errors

If PyTorch backend fails:
```bash
# Check GPU availability
python3 -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# Check model access
python3 -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('openai/gpt-oss-20b')"
```

### vLLM Connection Errors

If vLLM backend fails:
```bash
# Check vLLM server
curl http://localhost:8000/v1/models

# Check model loaded
curl http://localhost:8000/v1/models | jq '.data[].id'
```

## Integration with neuralwatt_cloud

To use reasoning levels with neuralwatt_cloud Q-learning:

```bash
# Via neuralwatt_cloud scripts with ai_energy_benchmark switch
cd /home/scott/src/neuralwatt_cloud
export USE_AI_ENERGY_BENCHMARK=true

# Benchmark with reasoning
./run-benchmark-genai.sh --llm gpt-oss-20b --profile moderate \
  --reasoning-effort high

# Q-learning with reasoning
./run-qlearning-genai.sh --llm gpt-oss-20b --episodes 100 \
  --reasoning-effort medium
```

## Model-Specific Notes

### gpt-oss-20b
- Supports `reasoning_effort`: low, medium, high
- Parameters passed to model.generate()
- Check model documentation for exact behavior

### DeepSeek-R1
- May use `thinking_budget` instead of `reasoning_effort`
- Explicit chain-of-thought generation
- Requires specific parameter mapping

### Llama 4
- Reasoning support model-dependent
- Check HuggingFace model card for capabilities

## Future Enhancements

1. Add reasoning effort as Q-learning action dimension
2. Implement model-specific parameter mapping registry
3. Add quality metrics (BLEU, ROUGE) to compare reasoning quality
4. Support streaming reasoning tokens visualization
5. Add carbon intensity optimization based on reasoning effort

## References

- [AIEnergyScore Configuration](https://github.com/huggingface/optimum-benchmark)
- [Benchmark Consolidation Plan](/home/scott/src/neuralwatt_cloud/design/benchmark_consolidation_plan.md)
- [ai_energy_benchmarks Documentation](https://github.com/...)
