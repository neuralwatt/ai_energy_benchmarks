# Qwen3 Reasoning Parameter Fix (FINAL)

## Problem Summary

When running benchmarks with Qwen3-0.6B and reasoning enabled, two critical issues occurred:

1. **Wrong Parameter Location**: `enable_thinking` was passed to `model.generate()` causing `ValueError: The following model_kwargs are not used by the model: ['enable_thinking']`
2. **No Early Failure Detection**: Docker would wait the full 60-minute timeout even though the benchmark failed in the first few seconds

## Root Causes

### Issue 1: Parameter Location

**Incorrect Approach** (Original):
```python
# Config: reasoning_formats.yaml
qwen:
  type: parameter  # ❌ Wrong!

# Code: backends/pytorch.py
gen_kwargs["enable_thinking"] = True  # ❌ Passed to generate()
outputs = model.generate(**inputs, **gen_kwargs)  # ❌ Fails!
```

**Why it Failed**:
- Qwen models were configured to use `ParameterFormatter`
- ParameterFormatter passes `enable_thinking` to `model.generate()`
- But Qwen3-0.6B doesn't support this parameter in `generate()` - it needs to be in the **chat template**

**Correct Approach** (Fixed):
According to [Alibaba Cloud documentation](https://www.alibabacloud.com/help/en/model-studio/deep-thinking) and [Hugging Face](https://huggingface.co/Qwen/Qwen3-0.6B):

```python
# Config: reasoning_formats.yaml
qwen:
  type: null  # ✅ Use chat template, not formatter

# Code: backends/pytorch.py
tokenized_inputs = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    return_tensors="pt",
    enable_thinking=True  # ✅ Passed to chat template!
)
```

### Issue 2: No Early Failure Detection

**Problem**:
- Benchmark would process all prompts even if every single one failed
- Docker would wait 60 minutes for timeout
- No feedback to indicate catastrophic failure

**Solution**:
Added fail-fast logic in `runner.py`:
- Track consecutive failures
- After 3 consecutive failures, stop the benchmark immediately
- Raise `RuntimeError` with clear error message
- Docker exits quickly with non-zero code

## Fixes Implemented

### Fix 1: Move `enable_thinking` to Chat Template

**File**: `ai_energy_benchmarks/backends/pytorch.py`

Changed line 314 from:
```python
use_chat_template = has_chat_template and not self.formatter and not self._legacy_use_harmony
```

To:
```python
use_chat_template = has_chat_template and not self._legacy_use_harmony
```

This ensures Qwen models use the chat template path even if they have a formatter assigned.

**Added Qwen-specific handling** (lines 323-342):
```python
# Handle Qwen thinking mode
if "qwen" in self.model_name.lower():
    # For Qwen models, enable_thinking is passed to chat template
    enable_thinking = True  # Default
    if reasoning_params:
        if "enable_thinking" in reasoning_params:
            # Handle string "true"/"false" from config
            val = reasoning_params["enable_thinking"]
            enable_thinking = val if isinstance(val, bool) else str(val).lower() == "true"
        elif "reasoning" in reasoning_params:
            val = reasoning_params["reasoning"]
            enable_thinking = val if isinstance(val, bool) else str(val).lower() == "true"
    chat_kwargs["enable_thinking"] = enable_thinking
    print(f"  Using Qwen chat template (thinking={'enabled' if enable_thinking else 'disabled'})")
```

**Key Features**:
- Handles both `enable_thinking` and `reasoning` parameters
- Converts string values ("true"/"false") to booleans
- Logs which mode is being used
- Defaults to `True` (thinking enabled)

### Fix 2: Update Configuration

**File**: `ai_energy_benchmarks/config/reasoning_formats.yaml`

Changed Qwen entry from:
```yaml
qwen:
  patterns:
    - "qwen"
  type: parameter  # ❌ Wrong approach
```

To:
```yaml
qwen:
  patterns:
    - "qwen"
  type: null  # ✅ No formatter - use chat template
  description: "Qwen models use enable_thinking via tokenizer.apply_chat_template()"
```

This prevents `FormatterRegistry` from assigning a formatter to Qwen models, allowing them to use the native chat template path.

### Fix 3: Early Failure Detection

**File**: `ai_energy_benchmarks/runner.py`

Added fail-fast logic (lines 196-235):

```python
consecutive_failures = 0
max_consecutive_failures = 3  # Fail fast after 3 consecutive failures

for i, prompt in enumerate(prompts):
    result = backend.run_inference(prompt, **gen_kwargs)

    if not result.get("success", False):
        consecutive_failures += 1
        error_msg = result.get("error", "Unknown error")
        print(f"    FAILED: {error_msg}", flush=True)

        if consecutive_failures >= max_consecutive_failures:
            print(f"\nERROR: {consecutive_failures} consecutive failures detected.")
            print(f"Stopping benchmark early to avoid wasting resources.")
            if collector is not None:
                collector.stop()
            raise RuntimeError(
                f"Benchmark failed: {consecutive_failures} consecutive inference failures. "
                f"Last error: {error_msg}"
            )
    else:
        consecutive_failures = 0  # Reset on success
```

**Benefits**:
- Fails after 3 consecutive errors (catches systematic issues)
- Resets counter on success (tolerates occasional failures)
- Cleans up metrics collector before exiting
- Raises `RuntimeError` with descriptive message
- Docker detects failure and exits immediately (no 60-min timeout!)

## How Qwen Reasoning Works

According to the official documentation:

### Qwen3 Mixed Models (Qwen3-0.6B, Qwen3-32B, etc.)

These models support **both** thinking and non-thinking modes:

- **Thinking Mode** (`enable_thinking=True`):
  - Model generates reasoning wrapped in `<think>...</think>` tags
  - Optimal for "complex logical reasoning, math, and coding"
  - Use Temperature=0.6, TopP=0.95, TopK=20, MinP=0
  - DO NOT use greedy decoding (causes repetitions)

- **Non-Thinking Mode** (`enable_thinking=False`):
  - Fast, direct responses
  - Optimal for "efficient, general-purpose dialogue"

### Thinking-Only Models (QwQ-32B, Qwen3-*-Thinking-2507)

These models **always** think:
- `enable_thinking` parameter is ignored
- Always generate `<think>...</think>` content
- Cannot be disabled

### OpenAI API Compatible Usage

For OpenAI-style APIs, pass via `extra_body`:
```python
extra_body={"enable_thinking": True}
```

For open-source models, pass to chat template:
```python
tokenizer.apply_chat_template(..., enable_thinking=True)
```

## Testing

Updated test script verifies:
1. ✅ Qwen3-0.6B does NOT use ParameterFormatter
2. ✅ Other Qwen models also use chat template
3. ✅ Reasoning parameter filtering still works (for other models)

Run with:
```bash
.venv/bin/python ai_helpers/test_qwen_reasoning_fix.py
```

## Code Quality

All checks pass:
```bash
.venv/bin/ruff check ai_energy_benchmarks/  # ✅
.venv/bin/ruff format ai_energy_benchmarks/  # ✅
.venv/bin/mypy ai_energy_benchmarks/backends/pytorch.py  # ✅
.venv/bin/mypy ai_energy_benchmarks/runner.py  # ✅
```

## Expected Behavior

### Before Fix:
```
Running inference...
  Processing prompt 1/10...
Exception in thread Thread-4 (generate):
ValueError: The following `model_kwargs` are not used by the model: ['enable_thinking']
[Process hangs for 60 minutes until timeout]
```

### After Fix:
```
Running inference...
  Processing prompt 1/10...
  Using Qwen chat template (thinking=enabled)
    Completed in 2.3s
  Processing prompt 2/10...
  Using Qwen chat template (thinking=enabled)
    Completed in 2.1s
[... continues successfully ...]
```

### If Something is Really Wrong:
```
Running inference...
  Processing prompt 1/10...
    FAILED in 0.5s: Model not loaded
  Processing prompt 2/10...
    FAILED in 0.5s: Model not loaded
  Processing prompt 3/10...
    FAILED in 0.5s: Model not loaded

======================================================================
ERROR: 3 consecutive failures detected.
Stopping benchmark early to avoid wasting resources.
Last error: Model not loaded
======================================================================

RuntimeError: Benchmark failed: 3 consecutive inference failures.
[Docker exits immediately with code 1]
```

## Impact on Other Models

### Qwen Models
- ✅ Now work correctly with reasoning mode
- ✅ Both `enable_thinking` and `reasoning` parameters supported
- ✅ String "true"/"false" converted to booleans

### Hunyuan Models
- ✅ Still work as before (already used chat template)
- ✅ Same parameter handling

### Other Models (DeepSeek, Phi, Gemma, EXAONE)
- ✅ Still use ParameterFormatter as before
- ✅ Existing error handling still works
- ✅ No breaking changes

## Usage

### Running Qwen3-0.6B with Reasoning:

```bash
python batch_runner.py \
    --model-name "Qwen3-0.6B" \
    --class A \
    --output-dir ./results/qwen \
    --num-prompts 10
```

The benchmark will:
1. Detect Qwen model
2. Use chat template with `enable_thinking=True`
3. Generate responses with thinking tags: `<think>reasoning...</think>final answer`
4. Complete successfully
5. If errors occur, fail fast after 3 consecutive failures

### Disabling Reasoning (Non-Thinking Mode):

```bash
python batch_runner.py \
    --model-name "Qwen3-0.6B" \
    --class N \
    --output-dir ./results/qwen \
    --num-prompts 10
```

Will use `enable_thinking=False` for fast, direct responses.

## Related Files

**Modified**:
- `ai_energy_benchmarks/backends/pytorch.py` - Chat template handling for Qwen
- `ai_energy_benchmarks/config/reasoning_formats.yaml` - Qwen config updated
- `ai_energy_benchmarks/runner.py` - Early failure detection
- `ai_helpers/test_qwen_reasoning_fix.py` - Updated tests

**Documentation**:
- `ai_helpers/QWEN_REASONING_FIX_V2.md` - This file
- `ai_helpers/QWEN_REASONING_FIX.md` - Original (superseded)

## References

1. [Alibaba Cloud: How to use Qwen3 thinking mode](https://www.alibabacloud.com/help/en/model-studio/deep-thinking)
2. [Hugging Face: Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B)
3. [QwQ-32B Blog Post](https://qwenlm.github.io/blog/qwq-32b/)
4. [Qwen3 Technical Report (arXiv:2505.09388)](https://arxiv.org/abs/2505.09388)
