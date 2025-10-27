# Testing Instructions for Qwen3 Fix

## What Was Changed

1. **Qwen3 now uses chat template with `enable_thinking`** instead of passing it to `generate()`
2. **Early failure detection** - benchmark fails fast after 3 consecutive errors
3. **Streaming disabled by default** to avoid potential hangs

## Test with Docker

```bash
# Your Docker command here
docker logs <container_id>
```

## What You Should See

### Successful Run:
```
Running inference...
  Processing prompt 1/10...
  Using Qwen chat template (thinking=enabled)
  Applying chat template with kwargs: {'add_generation_prompt': True, 'enable_thinking': True}
  Chat template applied successfully, input shape: torch.Size([1, X])
    Completed in 2.3s
  Processing prompt 2/10...
  [... continues ...]
```

### If Parameter Not Supported (Fallback):
```
  Chat template warning: TypeError: ...
  Fallback chat template applied, input shape: torch.Size([1, X])
```

### If Systematic Failure (Fast Fail):
```
  Processing prompt 1/10...
    FAILED in 0.5s: Error message
  Processing prompt 2/10...
    FAILED in 0.5s: Error message
  Processing prompt 3/10...
    FAILED in 0.5s: Error message

======================================================================
ERROR: 3 consecutive failures detected.
Stopping benchmark early to avoid wasting resources.
======================================================================
```

## If It Still Hangs

The issue is likely:
1. **Model loading** - Check if model is actually loaded
2. **CUDA/GPU issue** - Check GPU availability
3. **Chat template issue** - The template might not support `enable_thinking`

### Quick Debug

Add this line before running to get more output:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Alternative: Revert to Original Behavior

If you need the original working version, you can:
1. Comment out the Qwen-specific chat template code (lines 323-342)
2. Change line 314 back to:
```python
use_chat_template = has_chat_template and not self.formatter and not self._legacy_use_harmony
```

This will make it use the formatter approach again, which will trigger the error but at least you'll see what's happening.

## Key Files Changed

- `ai_energy_benchmarks/backends/pytorch.py` - Main logic
- `ai_energy_benchmarks/config/reasoning_formats.yaml` - Config
- `ai_energy_benchmarks/runner.py` - Fast fail logic
