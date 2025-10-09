# Reasoning Support - Quick Start

## Run Tests

```bash
cd /home/scott/src/ai_energy_benchmarks

# Run full validation (RECOMMENDED)
./run_reasoning_test.sh

# OR run individual tests
source .venv/bin/activate
python3 ai_helpers/test_reasoning_config.py   # Config parsing
python3 ai_helpers/test_reasoning_mock.py      # Mock parameter flow
python3 ai_helpers/test_reasoning_levels.py    # Full integration
```

## Expected Results

**✅ All tests should pass:**
- Config tests: `ALL TESTS PASSED ✓`
- Mock tests: `ALL MOCK TESTS PASSED ✓`
- Integration: PyTorch `10/10 successful`, vLLM `SKIPPED` (no server), AIEnergyScore `SKIPPED` (needs setup)

## Configuration Examples

**Test configs created:**
- `/home/scott/src/AIEnergyScore/text_generation_gptoss_reasoning_low.yaml`
- `/home/scott/src/AIEnergyScore/text_generation_gptoss_reasoning_medium.yaml`
- `/home/scott/src/AIEnergyScore/text_generation_gptoss_reasoning_high.yaml`

## Documentation

**Main docs:**
- `VALIDATION_COMPLETE.md` - Test results and status
- `FINAL_STATUS.md` - Implementation summary
- `ai_helpers/README_REASONING_TESTING.md` - Detailed testing guide
- `ai_helpers/REASONING_IMPLEMENTATION_SUMMARY.md` - Technical details

## Status

✅ **COMPLETE AND WORKING**

PyTorch backend validated with gpt-oss-20b model:
- 10/10 successful inferences
- Energy measurement: 4.03 Wh
- Graceful fallback for unsupported models

**Use `./run_reasoning_test.sh` to validate anytime!**
