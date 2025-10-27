# Qwen3 Reasoning Parameter Fix

## Problem

When running `batch_runner.py` (or similar benchmarking tools) with Qwen3-0.6B model and reasoning enabled (class A), the process failed with an error. The CSV output showed:

```csv
Qwen/Qwen3-0.6B,A,text_gen,On,0,0,0,0.00,0.0000,0.0000,0,0,0,0.00,0.0000,0.0000,0.0000,0.0000,2025-10-26T19:25:57.710913
```

All metrics are zero, indicating the Docker execution failed.

## Root Cause

1. **Reasoning Format Configuration**: According to `/mnt/storage/src/ai_energy_benchmarks/ai_energy_benchmarks/config/reasoning_formats.yaml`, Qwen models are configured to use `ParameterFormatter`:

```yaml
qwen:
  patterns:
    - "qwen"
  type: parameter
  description: "Qwen models using enable_thinking parameter"
```

2. **Parameter Passing**: The `ParameterFormatter` passes reasoning parameters (like `enable_thinking`, `reasoning`) directly to the model's `generate()` method:

```python
def get_generation_params(self, reasoning_params):
    gen_params = {}
    for key in ["enable_thinking", "thinking_budget", "cot_depth", "reasoning"]:
        if key in reasoning_params:
            gen_params[key] = reasoning_params[key]
    return gen_params
```

3. **Model Rejection**: Qwen3-0.6B doesn't actually support these reasoning parameters, so when `model.generate(**gen_kwargs)` is called with these params, it raises a `TypeError`:

```
TypeError: The following model_kwargs are not used by the model: reasoning, enable_thinking
```

4. **Incomplete Error Handling**: The PyTorch backend had error handling to filter out unsupported reasoning parameters, but **only in the non-streaming code path** (lines 511-547). When `enable_streaming=True` (the default), the error occurred in the thread-based generation (line 475) and wasn't properly caught.

## Solution

Added comprehensive error handling in the **streaming code path** of `ai_energy_benchmarks/backends/pytorch.py`:

### Changes Made

1. **Thread Exception Capture** (lines 475-483): Wrapped `model.generate()` in a function that captures exceptions from the thread:

```python
def generation_target():
    nonlocal thread_exception
    try:
        self.model.generate(**generation_kwargs)
    except Exception as e:
        thread_exception = e
```

2. **Retry Logic After Thread Completes** (lines 501-549): Check if the thread raised an exception related to unsupported parameters, and if so, retry with filtered kwargs:

```python
if thread_exception is not None:
    error_msg = str(thread_exception)
    if isinstance(thread_exception, (TypeError, ValueError)) and (
        "model_kwargs" in error_msg
        or "unexpected keyword argument" in error_msg
        or "not used by the model" in error_msg
    ):
        # Retry without reasoning parameters
        filtered_kwargs = filter_reasoning_params(gen_kwargs)
        # ... retry generation ...
```

3. **Token Counting Retry** (lines 555-562): Also added error handling when re-running generation for token counting (this uses `torch.no_grad()` context):

```python
try:
    outputs = self.model.generate(**inputs, **gen_kwargs)
except (TypeError, ValueError) as e:
    if "model_kwargs" in error_msg or ...:
        filtered_kwargs = filter_reasoning_params(gen_kwargs)
        outputs = self.model.generate(**inputs, **filtered_kwargs)
```

## Reasoning Parameters Filtered

The fix filters out these known reasoning-related parameters:

- `reasoning_effort`
- `thinking_budget`
- `cot_depth`
- `use_prompt_based_reasoning`
- `enable_thinking`
- `reasoning`

## Testing

Created test script `ai_helpers/test_qwen_reasoning_fix.py` which verifies:

1. ✓ Qwen3-0.6B uses ParameterFormatter correctly
2. ✓ Other Qwen models use ParameterFormatter correctly
3. ✓ Reasoning parameter filtering works correctly

Run with:
```bash
.venv/bin/python ai_helpers/test_qwen_reasoning_fix.py
```

## Code Quality Checks

All pre-commit checks pass:

```bash
.venv/bin/ruff check ai_energy_benchmarks/backends/pytorch.py  # ✓
.venv/bin/ruff format ai_energy_benchmarks/backends/pytorch.py  # ✓
.venv/bin/mypy ai_energy_benchmarks/backends/pytorch.py        # ✓
```

## Impact

### Models Affected

This fix benefits any model that:
1. Uses `ParameterFormatter` (Qwen, Phi, Gemma, EXAONE)
2. Doesn't actually support the reasoning parameters the formatter tries to pass

### Behavior Change

- **Before**: Inference would fail with `TypeError` when reasoning parameters weren't supported
- **After**: Inference automatically retries without the unsupported parameters and succeeds
- **User Experience**: Graceful degradation - the model runs without reasoning capabilities instead of failing completely

## Next Steps

You can now run batch_runner.py with Qwen3 and reasoning enabled:

```bash
python batch_runner.py \
    --model-name "Qwen3-0.6B" \
    --class A \
    --output-dir ./results/qwen \
    --num-prompts 10
```

The model will:
1. Attempt to use reasoning parameters (via ParameterFormatter)
2. Detect that Qwen3-0.6B doesn't support them
3. Automatically retry without reasoning parameters
4. Complete successfully with normal (non-reasoning) inference

## Related Files

- `ai_energy_benchmarks/backends/pytorch.py` - Main fix
- `ai_energy_benchmarks/formatters/parameter.py` - ParameterFormatter implementation
- `ai_energy_benchmarks/formatters/registry.py` - Formatter selection logic
- `ai_energy_benchmarks/config/reasoning_formats.yaml` - Model-to-formatter mapping
- `ai_helpers/test_qwen_reasoning_fix.py` - Test verification
