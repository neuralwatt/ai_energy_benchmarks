# Final Qwen3 Fix - Supporting Multiple Reasoning Formats

## The Goal

Support different reasoning formats for different model families:
- **gpt-oss (120b, 20b)**: Use HarmonyFormatter (prompt formatting)
- **Qwen**: Use chat template with `enable_thinking` parameter
- **Hunyuan**: Use chat template with `enable_thinking` parameter
- **Other models**: Use their respective formatters (DeepSeek, Phi, etc.)

## The Problem

Original code had:
```python
use_chat_template = has_chat_template and not self.formatter and not self._legacy_use_harmony
```

This meant:
- If a model has a formatter → Don't use chat template
- Qwen was assigned ParameterFormatter → Couldn't use chat template
- Result: `enable_thinking` passed to `generate()` → Error!

## The Solution

### Step 1: Configure Qwen to NOT Use Formatter

**File**: `config/reasoning_formats.yaml`

```yaml
qwen:
  patterns:
    - "qwen"
  type: null  # No formatter - use chat template
  description: "Qwen models use enable_thinking via tokenizer.apply_chat_template()"
```

This ensures `self.formatter` is `None` for Qwen models.

### Step 2: Fix Chat Template Logic

**File**: `backends/pytorch.py` (lines 312-327)

```python
# Special handling for different model types:
# - Qwen/Hunyuan: Use chat template with enable_thinking parameter
# - gpt-oss: Use HarmonyFormatter (self.formatter is set)
# - Others with chat template but no formatter: Use chat template
is_qwen = "qwen" in self.model_name.lower()
is_hunyuan = "hunyuan" in self.model_name.lower()

# Use chat template if:
# 1. Model has chat template AND
# 2. (No formatter assigned OR model is Qwen/Hunyuan which need chat template)
use_chat_template = (
    has_chat_template
    and not self._legacy_use_harmony
    and (not self.formatter or is_qwen or is_hunyuan)
)
```

### How It Works

**For gpt-oss-120b**:
1. Has chat template: ✅
2. Has formatter (HarmonyFormatter): ✅
3. `not self.formatter` = False
4. `is_qwen or is_hunyuan` = False
5. → `use_chat_template` = False
6. → Goes to formatter path (line 388+)
7. → Uses HarmonyFormatter.format_prompt()
8. ✅ Works correctly!

**For Qwen3-0.6B**:
1. Has chat template: ✅
2. Has NO formatter (type: null): ✅
3. `not self.formatter` = True
4. → `use_chat_template` = True
5. → Goes to chat template path (line 329+)
6. → Passes `enable_thinking` to `apply_chat_template()`
7. ✅ Works correctly!

**For Hunyuan** (if it had a formatter):
1. Has chat template: ✅
2. Might have formatter: ❓
3. `is_hunyuan` = True
4. → `use_chat_template` = True (overrides formatter check)
5. → Goes to chat template path
6. ✅ Works correctly!

## Step 3: Add Qwen-Specific Handling

**File**: `backends/pytorch.py` (lines 335-359)

```python
# Handle Qwen thinking mode
if "qwen" in self.model_name.lower():
    enable_thinking = True  # Default
    if reasoning_params:
        if "enable_thinking" in reasoning_params:
            # Handle string "true"/"false" from config
            val = reasoning_params["enable_thinking"]
            enable_thinking = (
                val if isinstance(val, bool) else str(val).lower() == "true"
            )
        elif "reasoning" in reasoning_params:
            val = reasoning_params["reasoning"]
            enable_thinking = (
                val if isinstance(val, bool) else str(val).lower() == "true"
            )
    chat_kwargs["enable_thinking"] = enable_thinking
    print(
        f"  Using Qwen chat template (thinking={'enabled' if enable_thinking else 'disabled'})"
    )
```

## Step 4: Add Early Failure Detection

**File**: `runner.py` (lines 196-235)

```python
consecutive_failures = 0
max_consecutive_failures = 3  # Fail fast after 3 consecutive failures

for i, prompt in enumerate(prompts):
    result = backend.run_inference(prompt, **gen_kwargs)

    if not result.get("success", False):
        consecutive_failures += 1
        error_msg = result.get("error", "Unknown error")
        print(f"    FAILED: {error_msg}")

        if consecutive_failures >= max_consecutive_failures:
            print(f"\nERROR: {consecutive_failures} consecutive failures detected.")
            raise RuntimeError(f"Benchmark failed: {error_msg}")
    else:
        consecutive_failures = 0  # Reset on success
```

## Step 5: Disable Streaming by Default

**File**: `backends/pytorch.py` (line 278)

```python
enable_streaming: bool = False,  # Disabled by default to avoid hangs
```

Changed from `True` to `False` to avoid potential threading issues causing hangs.

## Step 6: Add TTFT Estimation for Non-Streaming

**File**: `backends/pytorch.py` (lines 700-706)

```python
# For non-streaming, estimate TTFT based on generation time and token count
# This is an approximation: total_time / total_tokens ≈ time_per_token
if ttft is None and completion_tokens > 0:
    generation_time = end_time - start_time
    # Estimate TTFT as the time to generate first token
    # Assume roughly linear generation rate
    ttft = generation_time / completion_tokens
```

When streaming is disabled, we estimate TTFT by dividing total generation time by number of tokens. This provides an approximation for the `avg_time_to_first_token` metric in the CSV output.

## Expected Behavior

### Qwen3-0.6B with Reasoning:
```
Running inference...
  Processing prompt 1/10...
  Using Qwen chat template (thinking=enabled)
  Applying chat template with kwargs: {'add_generation_prompt': True, 'enable_thinking': True}
  Chat template applied successfully, input shape: torch.Size([1, 234])
    Completed in 2.3s
```

### gpt-oss-120b with Reasoning:
```
Running inference...
  Processing prompt 1/10...
  Using Harmony format with high reasoning
    Completed in 3.1s
```

### Early Failure Detection:
```
  Processing prompt 1/10...
    FAILED in 0.5s: CUDA out of memory
  Processing prompt 2/10...
    FAILED in 0.5s: CUDA out of memory
  Processing prompt 3/10...
    FAILED in 0.5s: CUDA out of memory

======================================================================
ERROR: 3 consecutive failures detected.
Stopping benchmark early to avoid wasting resources.
======================================================================
```

## Testing

### Test Qwen:
```bash
python batch_runner.py --model-name "Qwen3-0.6B" --class A --num-prompts 10
```

Should see: "Using Qwen chat template (thinking=enabled)"

### Test gpt-oss:
```bash
python batch_runner.py --model-name "gpt-oss-120b" --class A --num-prompts 10
```

Should see: "Using Harmony format with high reasoning"

### Test Other Models:
Should continue working as before with their respective formatters.

## Files Changed

1. `ai_energy_benchmarks/backends/pytorch.py`:
   - Lines 312-327: Chat template selection logic
   - Lines 335-359: Qwen-specific handling
   - Line 278: Streaming disabled by default
   - Lines 365-386: Added debug logging

2. `ai_energy_benchmarks/config/reasoning_formats.yaml`:
   - Qwen config changed to `type: null`

3. `ai_energy_benchmarks/runner.py`:
   - Lines 196-235: Early failure detection

4. Tests & Documentation:
   - `ai_helpers/test_qwen_reasoning_fix.py`
   - `ai_helpers/FINAL_FIX_SUMMARY.md` (this file)

## Summary

The fix ensures that:
✅ Qwen uses `enable_thinking` via chat template (correct approach)
✅ gpt-oss continues using HarmonyFormatter (unchanged)
✅ Other models use their formatters (unchanged)
✅ Benchmarks fail fast on systemic errors (no 60-min timeout)
✅ All pre-commit checks pass (ruff, mypy)
