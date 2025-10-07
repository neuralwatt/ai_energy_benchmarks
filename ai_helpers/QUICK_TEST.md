# Quick PyTorch Backend Test

Since AIEnergyScore has compatibility issues with Blackwell GPU, test ai_energy_benchmarks directly.

## Run Test

```bash
cd /home/scott/src/ai_energy_benchmarks
source .venv/bin/activate
./run_validation_test.sh
```

## Configuration

- **Model:** `meta-llama/Llama-3.2-3B-Instruct` (smaller, faster)
- **Samples:** 100 (quick test)
- **Tokens:** 10 per completion
- **Backend:** PyTorch (local inference)

## Results

Results will be in:
- `./results/pytorch_validation_results.csv`
- `./emissions/pytorch_validation/`

## View Results

```bash
python3 << 'EOF'
import pandas as pd
df = pd.read_csv('results/pytorch_validation_results.csv')

print("PyTorch Backend Test Results")
print("=" * 50)
print(f"Samples completed: {len(df)}")
print(f"Throughput: {df['throughput_tokens_per_sec'].mean():.2f} tokens/s")
print(f"Mean Latency: {df['latency_seconds'].mean():.4f} s")
print(f"Total Tokens: {df['total_tokens'].sum()}")
if 'gpu_energy_wh' in df.columns:
    print(f"GPU Energy: {df['gpu_energy_wh'].sum():.2f} Wh")
EOF
```

## Next Steps

Once this works, you can:
1. Increase num_samples back to 1000
2. Test with larger models if needed
3. Compare with vLLM backend if desired
