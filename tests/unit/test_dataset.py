"""Unit tests for HuggingFace dataset loader."""

from unittest.mock import Mock, patch

import pytest

from ai_energy_benchmarks.datasets.huggingface import HuggingFaceDataset


class TestHuggingFaceDataset:
    """Test HuggingFace dataset loader."""

    def test_initialization(self):
        """Test dataset loader initialization."""
        dataset = HuggingFaceDataset()
        assert dataset.dataset is None

    @patch("datasets.load_dataset")
    def test_load_dataset_success(self, mock_load):
        """Test successful dataset loading."""
        mock_dataset = Mock()
        mock_dataset.column_names = ["text", "label"]
        mock_dataset.__getitem__ = Mock(return_value=["prompt1", "prompt2", "prompt3"])
        mock_load.return_value = mock_dataset

        dataset = HuggingFaceDataset()
        prompts = dataset.load({"name": "test-dataset", "text_column": "text", "num_samples": 2})

        assert len(prompts) == 2

    @patch("datasets.load_dataset")
    def test_load_dataset_missing_name(self, mock_load):
        """Test loading dataset without name."""
        dataset = HuggingFaceDataset()

        with pytest.raises(ValueError, match="Dataset name must be specified"):
            dataset.load({})

    @patch("datasets.load_dataset")
    def test_load_dataset_invalid_column(self, mock_load):
        """Test loading dataset with invalid column name."""
        mock_dataset = Mock()
        mock_dataset.column_names = ["text", "label"]
        mock_load.return_value = mock_dataset

        dataset = HuggingFaceDataset()

        with pytest.raises(ValueError, match="Column .* not found"):
            dataset.load({"name": "test-dataset", "text_column": "invalid_column"})

    def test_validate_with_datasets_installed(self):
        """Test validation when datasets is installed."""
        dataset = HuggingFaceDataset()
        # This will return True if datasets is actually installed
        result = dataset.validate()
        assert isinstance(result, bool)
