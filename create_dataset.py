import numpy as np
import pandas as pd


def create_energy_dataset(num_samples=1000):
    """
    Creates a synthetic dataset for AI energy benchmarks.
    """
    np.random.seed(42)  # For reproducibility
    models = ["GPT-3", "BERT", "ResNet", "Transformer"]
    data = {
        "model_name": np.random.choice(models, num_samples),
        "energy_consumed": np.random.uniform(10, 1000, num_samples),  # in kWh
        "runtime": np.random.uniform(1, 60, num_samples),  # in minutes
        "accuracy": np.random.uniform(0.5, 1.0, num_samples),
    }
    df = pd.DataFrame(data)
    df.to_csv("ai_energy_dataset.csv", index=False)
    print("Dataset created and saved as ai_energy_dataset.csv")


if __name__ == "__main__":
    create_energy_dataset()
