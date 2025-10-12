#!/usr/bin/env python3
"""Test single inference to debug issues."""

from ai_energy_benchmarks.backends.pytorch import PyTorchBackend
from ai_energy_benchmarks.datasets.huggingface import HuggingFaceDataset

# Load model
print("Loading model...")
backend = PyTorchBackend(
    model="openai/gpt-oss-20b", device="cuda", device_ids=[0], torch_dtype="auto", device_map="auto"
)

# Load a sample prompt
print("\nLoading dataset...")
dataset = HuggingFaceDataset()
prompts = dataset.load(
    {"name": "EnergyStarAI/text_generation", "text_column": "text", "num_samples": 1}
)

print(f"\nSample prompt: {prompts[0][:100]}...")

# Try inference
print("\nRunning inference...")
result = backend.run_inference(prompts[0], max_tokens=10, temperature=0.7)

print("\n=== Result ===")
print(f"Success: {result.get('success')}")
print(f"Error: {result.get('error')}")
print(f"Generated text: {result.get('text', '')[:200]}")
print(f"Tokens: {result.get('total_tokens')}")
print(f"Latency: {result.get('latency_seconds')} seconds")
