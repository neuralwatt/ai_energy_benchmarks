# Testing Summary for Qwen3 Multi-Format Fix

## What Was Fixed

### Issue 1: Qwen3 Parameter Error ✅
**Problem**: `enable_thinking` parameter passed to `model.generate()` causing:
```
ValueError: The following model_kwargs are not used by the model: ['enable_thinking']
```

**Solution**: Moved `enable_thinking` to `tokenizer.apply_chat_template()` where Qwen expects it.

### Issue 2: 60-Minute Docker Timeout ✅
**Problem**: Benchmark would wait full 60 minutes even when all prompts failed in first few seconds.

**Solution**: Added fail-fast logic that stops after 3 consecutive failures.

### Issue 3: Multi-Format Support ✅
**Problem**: Need to support different reasoning formats for different model families.

**Solution**: Implemented conditional logic:
- **gpt-oss models** → HarmonyFormatter (unchanged)
- **Qwen models** → Chat template with `enable_thinking`
- **Other models** → Their respective formatters

## Files Changed

1. **`ai_energy_benchmarks/backends/pytorch.py`**:
   - Lines 312-327: Multi-format selection logic
   - Lines 335-359: Qwen-specific `enable_thinking` handling
   - Line 278: Streaming disabled by default
   - Lines 365-386: Debug logging

2. **`ai_energy_benchmarks/config/reasoning_formats.yaml`**:
   - Lines 49-53: Qwen set to `type: null` (no formatter)

3. **`ai_energy_benchmarks/runner.py`**:
   - Lines 196-235: Early failure detection

## Expected Test Results

### Test 1: Qwen3-0.6B with Reasoning (Class A)
```bash
python batch_runner.py --model-name "Qwen3-0.6B" --class A --output-dir ./results/qwen --num-prompts 10
```

**Expected Output**:
```
Running inference...
  Processing prompt 1/10...
  Using Qwen chat template (thinking=enabled)
  Applying chat template with kwargs: {'add_generation_prompt': True, 'enable_thinking': True}
  Chat template applied successfully, input shape: torch.Size([1, X])
    Completed in 2.3s
  Processing prompt 2/10...
  [continues successfully]
```

**Success Criteria**:
- ✅ No `ValueError` about `enable_thinking`
- ✅ Log shows "Using Qwen chat template (thinking=enabled)"
- ✅ All 10 prompts complete successfully
- ✅ CSV shows non-zero tokens and metrics

### Test 2: gpt-oss-120b with Reasoning (Class A)
```bash
python batch_runner.py --model-name "gpt-oss-120b" --class A --output-dir ./results/gpt --num-prompts 10
```

**Expected Output**:
```
Running inference...
  Processing prompt 1/10...
  Using Harmony format with high reasoning
    Completed in 3.1s
  Processing prompt 2/10...
  [continues successfully]
```

**Success Criteria**:
- ✅ Log shows "Using Harmony format with high reasoning"
- ✅ All 10 prompts complete successfully
- ✅ No changes to existing behavior (regression test)

### Test 3: Early Failure Detection
To test fail-fast (if a model fails systematically):

**Expected Output**:
```
  Processing prompt 1/10...
    FAILED in 0.5s: [error message]
  Processing prompt 2/10...
    FAILED in 0.5s: [error message]
  Processing prompt 3/10...
    FAILED in 0.5s: [error message]

======================================================================
ERROR: 3 consecutive failures detected.
Stopping benchmark early to avoid wasting resources.
Last error: [error message]
======================================================================

RuntimeError: Benchmark failed: 3 consecutive inference failures.
```

**Success Criteria**:
- ✅ Exits after 3 consecutive failures (not 60 minutes)
- ✅ Docker container exits immediately with error code
- ✅ Clear error message about what failed

## Code Quality ✅

All checks pass:
```bash
.venv/bin/ruff check ai_energy_benchmarks/  # ✅ All checks passed!
.venv/bin/ruff format --check ai_energy_benchmarks/  # ✅ 2 files already formatted
.venv/bin/mypy ai_energy_benchmarks/backends/pytorch.py  # ✅ Success: no issues
.venv/bin/mypy ai_energy_benchmarks/runner.py  # ✅ Success: no issues
```

## How It Works

### Model Type Detection (pytorch.py:312-327)
```python
is_qwen = "qwen" in self.model_name.lower()
is_hunyuan = "hunyuan" in self.model_name.lower()

use_chat_template = (
    has_chat_template
    and not self._legacy_use_harmony
    and (not self.formatter or is_qwen or is_hunyuan)
)
```

**Logic Flow**:
1. **gpt-oss-120b**: Has HarmonyFormatter → `not self.formatter` = False → `use_chat_template` = False → Uses formatter ✅
2. **Qwen3-0.6B**: No formatter (type: null) → `not self.formatter` = True → `use_chat_template` = True → Uses chat template ✅
3. **Hunyuan**: `is_hunyuan` = True → `use_chat_template` = True (override) → Uses chat template ✅

### Qwen-Specific Handling (pytorch.py:337-359)
```python
if "qwen" in self.model_name.lower():
    enable_thinking = True  # Default
    if reasoning_params:
        if "enable_thinking" in reasoning_params:
            val = reasoning_params["enable_thinking"]
            enable_thinking = val if isinstance(val, bool) else str(val).lower() == "true"
        elif "reasoning" in reasoning_params:
            val = reasoning_params["reasoning"]
            enable_thinking = val if isinstance(val, bool) else str(val).lower() == "true"
    chat_kwargs["enable_thinking"] = enable_thinking
    print(f"  Using Qwen chat template (thinking={'enabled' if enable_thinking else 'disabled'})")
```

**Features**:
- Defaults to `enable_thinking=True`
- Handles both `enable_thinking` and `reasoning` parameters
- Converts string "true"/"false" to boolean
- Logs which mode is active

### Early Failure Detection (runner.py:196-235)
```python
consecutive_failures = 0
max_consecutive_failures = 3

for i, prompt in enumerate(prompts):
    result = backend.run_inference(prompt, **gen_kwargs)

    if not result.get("success", False):
        consecutive_failures += 1
        if consecutive_failures >= max_consecutive_failures:
            # Clean up and raise RuntimeError
            raise RuntimeError(f"Benchmark failed: {consecutive_failures} consecutive failures")
    else:
        consecutive_failures = 0  # Reset on success
```

**Features**:
- Tracks consecutive failures
- Fails after 3 in a row (catches systemic issues)
- Resets counter on success (tolerates occasional failures)
- Cleans up metrics collector before exiting

## Next Steps

1. **Test Qwen3-0.6B** with class A (reasoning enabled)
2. **Test gpt-oss-120b** with class A (regression test)
3. **Verify Docker logs** show correct messages
4. **Confirm CSV output** has non-zero metrics

If any test fails, check Docker logs for:
- Which code path was taken (Qwen chat template vs Harmony formatter)
- Error messages
- Where execution stopped (hung vs error)
