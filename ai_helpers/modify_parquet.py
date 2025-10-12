import pandas as pd

# List of new prompts
new_prompts = [
    'Summarize the core flaw in this argument: "If everyone recycled, climate change would end."',
    "What’s the most likely unintended consequence of universal basic income?",
    "Explain why correlation ≠ causation in one sentence.",
    "Which is more fragile: democracy or capitalism? Justify briefly.",
    "What’s the simplest reason AI alignment is hard?",
    'Describe the paradox in "the pursuit of happiness."',
    "Why might optimizing for energy efficiency reduce overall system resilience?",
    "What’s the most contrarian but plausible take on inflation?",
    "Give one sentence that explains why benchmarks can mislead operational decisions.",
    "What’s the most elegant way to describe entropy to a 10-year-old?",
]

# Path to the input parquet file
input_path = r"train-00000-of-00001.parquet"

# Column name to replace
prompt_column = "prompt"

# Load the parquet file
df = pd.read_parquet(input_path)

# Truncate to 10 rows (take the first 10)
df_truncated = df.head(10).copy()

# Replace the specified column with the new prompts
df_truncated[prompt_column] = new_prompts

# Save as a new parquet file
output_path = "modified_train.parquet"
df_truncated.to_parquet(output_path, index=False)

print(f"Modified dataset with 10 rows saved as {output_path}")
