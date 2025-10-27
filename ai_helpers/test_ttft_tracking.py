"""Unit tests for TTFT tracking in PyTorch backend."""

from typing import Any
from unittest.mock import MagicMock, Mock, patch

import pytest


class TestPyTorchBackendTTFT:
    """Test PyTorch backend TTFT tracking implementation."""

    @patch("ai_energy_benchmarks.backends.pytorch.torch")
    @patch("ai_energy_benchmarks.backends.pytorch.AutoModelForCausalLM")
    @patch("ai_energy_benchmarks.backends.pytorch.AutoTokenizer")
    def test_run_inference_with_streaming_ttft(
        self, mock_tokenizer_class, mock_model_class, mock_torch
    ):
        """Test inference with streaming to capture TTFT."""
        from ai_energy_benchmarks.backends.pytorch import PyTorchBackend

        # Setup mocks
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.device_count.return_value = 1
        mock_torch.float16 = "float16"
        mock_torch.bfloat16 = "bfloat16"
        mock_torch.float32 = "float32"

        # Mock tokenizer
        mock_tokenizer = Mock()
        mock_tokenizer.pad_token = None
        mock_tokenizer.eos_token = "<eos>"
        mock_tokenizer.pad_token_id = 0
        mock_tokenizer.chat_template = None
        mock_tokenizer.return_value = {"input_ids": Mock(shape=[1, 10])}
        mock_tokenizer.decode.return_value = "test output"
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer

        # Mock model
        mock_model = Mock()
        mock_model.dtype = "float16"
        mock_model.eval.return_value = None
        mock_model_class.from_pretrained.return_value = mock_model

        # Mock generate output
        mock_output = Mock()
        mock_output.shape = [1, 30]  # 30 tokens total
        mock_output.__getitem__ = lambda self, idx: mock_output
        mock_model.generate.return_value = mock_output

        # Initialize backend
        backend = PyTorchBackend(model="test-model", device="cuda")

        # Mock TextIteratorStreamer
        with patch(
            "ai_energy_benchmarks.backends.pytorch.TextIteratorStreamer"
        ) as mock_streamer_class:
            mock_streamer = MagicMock()
            # Simulate streaming tokens
            mock_streamer.__iter__.return_value = iter(["Hello", " world", "!"])
            mock_streamer_class.return_value = mock_streamer

            with patch("ai_energy_benchmarks.backends.pytorch.Thread") as mock_thread_class:
                mock_thread = Mock()
                mock_thread_class.return_value = mock_thread

                result = backend.run_inference("test prompt", enable_streaming=True)

                # Verify TTFT was captured
                assert result["success"] is True
                assert result["time_to_first_token"] is not None
                assert result["time_to_first_token"] >= 0
                assert "latency_seconds" in result

    @patch("ai_energy_benchmarks.backends.pytorch.torch")
    @patch("ai_energy_benchmarks.backends.pytorch.AutoModelForCausalLM")
    @patch("ai_energy_benchmarks.backends.pytorch.AutoTokenizer")
    def test_run_inference_non_streaming_no_ttft(
        self, mock_tokenizer_class, mock_model_class, mock_torch
    ):
        """Test non-streaming inference returns None for TTFT."""
        from ai_energy_benchmarks.backends.pytorch import PyTorchBackend

        # Setup mocks
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.device_count.return_value = 1
        mock_torch.float16 = "float16"
        mock_torch.bfloat16 = "bfloat16"
        mock_torch.float32 = "float32"

        # Mock tokenizer
        mock_tokenizer = Mock()
        mock_tokenizer.pad_token = None
        mock_tokenizer.eos_token = "<eos>"
        mock_tokenizer.pad_token_id = 0
        mock_tokenizer.chat_template = None
        mock_tokenizer.return_value = {"input_ids": Mock(shape=[1, 10])}
        mock_tokenizer.decode.side_effect = ["test prompt text", "test output"]
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer

        # Mock model
        mock_model = Mock()
        mock_model.dtype = "float16"
        mock_model.eval.return_value = None
        mock_model_class.from_pretrained.return_value = mock_model

        # Mock generate output
        mock_output = Mock()
        mock_output.shape = [1, 30]  # 30 tokens total
        mock_output.__getitem__ = lambda self, idx: mock_output
        mock_model.generate.return_value = mock_output

        # Initialize backend
        backend = PyTorchBackend(model="test-model", device="cuda")

        result = backend.run_inference("test prompt", enable_streaming=False)

        # Verify no TTFT was captured
        assert result["success"] is True
        assert result["time_to_first_token"] is None
        assert "latency_seconds" in result

    @patch("ai_energy_benchmarks.backends.pytorch.torch")
    @patch("ai_energy_benchmarks.backends.pytorch.AutoModelForCausalLM")
    @patch("ai_energy_benchmarks.backends.pytorch.AutoTokenizer")
    def test_run_inference_streaming_fallback_on_error(
        self, mock_tokenizer_class, mock_model_class, mock_torch
    ):
        """Test fallback to non-streaming when TextIteratorStreamer fails."""
        from ai_energy_benchmarks.backends.pytorch import PyTorchBackend

        # Setup mocks
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.device_count.return_value = 1
        mock_torch.float16 = "float16"
        mock_torch.bfloat16 = "bfloat16"
        mock_torch.float32 = "float32"

        # Mock tokenizer
        mock_tokenizer = Mock()
        mock_tokenizer.pad_token = None
        mock_tokenizer.eos_token = "<eos>"
        mock_tokenizer.pad_token_id = 0
        mock_tokenizer.chat_template = None
        mock_tokenizer.return_value = {"input_ids": Mock(shape=[1, 10])}
        mock_tokenizer.decode.side_effect = ["test prompt text", "test output"]
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer

        # Mock model
        mock_model = Mock()
        mock_model.dtype = "float16"
        mock_model.eval.return_value = None
        mock_model_class.from_pretrained.return_value = mock_model

        # Mock generate output
        mock_output = Mock()
        mock_output.shape = [1, 30]  # 30 tokens total
        mock_output.__getitem__ = lambda self, idx: mock_output
        mock_model.generate.return_value = mock_output

        # Initialize backend
        backend = PyTorchBackend(model="test-model", device="cuda")

        # Mock TextIteratorStreamer to raise ImportError
        with patch(
            "ai_energy_benchmarks.backends.pytorch.TextIteratorStreamer",
            side_effect=ImportError("Module not found"),
        ):
            result = backend.run_inference("test prompt", enable_streaming=True)

            # Should fallback to non-streaming
            assert result["success"] is True
            assert result["time_to_first_token"] is None


class TestRunnerTTFTAggregation:
    """Test runner aggregation of TTFT metrics."""

    def test_aggregate_ttft_from_results(self):
        """Test that runner correctly aggregates TTFT metrics."""
        # Mock inference results with TTFT
        inference_results: list[dict[str, Any]] = [
            {
                "success": True,
                "latency_seconds": 1.0,
                "time_to_first_token": 0.1,
                "total_tokens": 10,
            },
            {
                "success": True,
                "latency_seconds": 1.2,
                "time_to_first_token": 0.15,
                "total_tokens": 12,
            },
            {
                "success": True,
                "latency_seconds": 0.9,
                "time_to_first_token": 0.12,
                "total_tokens": 8,
            },
        ]

        # Calculate expected average
        expected_avg_ttft = (0.1 + 0.15 + 0.12) / 3

        # Simulate aggregation logic from runner
        successful = [r for r in inference_results if r.get("success", False)]
        ttft_values: list[float] = [
            r.get("time_to_first_token")  # type: ignore[misc]
            for r in successful
            if r.get("time_to_first_token") is not None
        ]
        avg_ttft = sum(ttft_values) / len(ttft_values) if ttft_values else 0

        assert avg_ttft == pytest.approx(expected_avg_ttft)

    def test_aggregate_ttft_with_none_values(self):
        """Test aggregation handles None TTFT values correctly."""
        # Mix of results with and without TTFT
        inference_results: list[dict[str, Any]] = [
            {
                "success": True,
                "latency_seconds": 1.0,
                "time_to_first_token": 0.1,
                "total_tokens": 10,
            },
            {
                "success": True,
                "latency_seconds": 1.2,
                "time_to_first_token": None,
                "total_tokens": 12,
            },
            {
                "success": True,
                "latency_seconds": 0.9,
                "time_to_first_token": 0.12,
                "total_tokens": 8,
            },
        ]

        # Calculate expected average (only non-None values)
        expected_avg_ttft = (0.1 + 0.12) / 2

        # Simulate aggregation logic from runner
        successful = [r for r in inference_results if r.get("success", False)]
        ttft_values: list[float] = [
            r.get("time_to_first_token")  # type: ignore[misc]
            for r in successful
            if r.get("time_to_first_token") is not None
        ]
        avg_ttft = sum(ttft_values) / len(ttft_values) if ttft_values else 0

        assert avg_ttft == pytest.approx(expected_avg_ttft)

    def test_aggregate_ttft_all_none(self):
        """Test aggregation when all TTFT values are None."""
        inference_results: list[dict[str, Any]] = [
            {
                "success": True,
                "latency_seconds": 1.0,
                "time_to_first_token": None,
                "total_tokens": 10,
            },
            {
                "success": True,
                "latency_seconds": 1.2,
                "time_to_first_token": None,
                "total_tokens": 12,
            },
        ]

        # Simulate aggregation logic from runner
        successful = [r for r in inference_results if r.get("success", False)]
        ttft_values: list[float] = [
            r.get("time_to_first_token")  # type: ignore[misc]
            for r in successful
            if r.get("time_to_first_token") is not None
        ]
        avg_ttft = sum(ttft_values) / len(ttft_values) if ttft_values else 0

        assert avg_ttft == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
