"""HuggingFace datasets loader for benchmark prompts."""

from typing import Any, Dict, List

from ai_energy_benchmarks.datasets.base import Dataset


class HuggingFaceDataset(Dataset):
    """Load datasets from HuggingFace datasets library."""

    def __init__(self):
        """Initialize HuggingFace dataset loader."""
        self.dataset: Any = None

    def load(self, config: Dict[str, Any]) -> List[str]:
        """Load prompts from HuggingFace dataset.

        Args:
            config: Dataset configuration with keys:
                - name: Dataset name (e.g., "AIEnergyScore/text_generation")
                - split: Dataset split (default: "train")
                - text_column: Column containing prompts (default: "text")
                - num_samples: Number of samples to load (default: all)
                - cache_dir: Cache directory (optional)

        Returns:
            List of prompt strings
        """
        try:
            from datasets import load_dataset  # type: ignore[import-untyped]

            dataset_name = config.get("name", config.get("dataset_name"))
            if not dataset_name:
                raise ValueError("Dataset name must be specified in config")

            split = config.get("split", "train")
            cache_dir = config.get("cache_dir")

            print(f"Loading dataset: {dataset_name} (split: {split})")

            # Load dataset
            self.dataset = load_dataset(dataset_name, split=split, cache_dir=cache_dir)

            # Extract prompts from specified column
            text_column = config.get("text_column", config.get("text_column_name", "text"))
            if text_column not in self.dataset.column_names:
                raise ValueError(
                    f"Column '{text_column}' not found in dataset. "
                    f"Available columns: {self.dataset.column_names}"
                )

            prompts_data = self.dataset[text_column]
            prompts_list = [str(item) for item in prompts_data]

            # Limit number of samples if specified
            num_samples = config.get("num_samples")
            if num_samples and num_samples > 0:
                prompts_list = prompts_list[:num_samples]
                print(f"Loaded {num_samples} samples from {dataset_name}")
            else:
                print(f"Loaded {len(prompts_list)} samples from {dataset_name}")

            return prompts_list

        except ImportError as err:
            raise ImportError(
                "HuggingFace datasets library not installed. Install with: pip install datasets"
            ) from err
        except ValueError:
            raise
        except Exception as err:
            raise RuntimeError(f"Error loading dataset: {err}") from err

    def validate(self) -> bool:
        """Validate dataset availability.

        Returns:
            bool: True if dataset is available
        """
        try:
            import importlib.util

            return importlib.util.find_spec("datasets") is not None
        except Exception:
            return False
