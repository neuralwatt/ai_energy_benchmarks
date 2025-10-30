"""Unit tests for vLLM backend."""

from unittest.mock import Mock, patch

from ai_energy_benchmarks.backends.vllm import VLLMBackend


class TestVLLMBackend:
    """Test vLLM backend implementation."""

    def test_initialization(self):
        """Test backend initialization."""
        backend = VLLMBackend(endpoint="http://localhost:8000/v1", model="test-model")
        assert backend.endpoint == "http://localhost:8000"
        assert backend.model == "test-model"

    def test_endpoint_normalization(self):
        """Test endpoint URL normalization."""
        backend = VLLMBackend(endpoint="http://localhost:8000/v1/", model="test-model")
        assert backend.endpoint == "http://localhost:8000"

    @patch("ai_energy_benchmarks.backends.vllm.requests.get")
    def test_validate_environment_success(self, mock_get):
        """Test successful environment validation."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"data": [{"id": "test-model"}]}
        mock_get.return_value = mock_response

        backend = VLLMBackend(endpoint="http://localhost:8000", model="test-model")
        assert backend.validate_environment() is True

    @patch("ai_energy_benchmarks.backends.vllm.requests.get")
    def test_validate_environment_failure(self, mock_get):
        """Test failed environment validation."""
        mock_get.side_effect = Exception("Connection error")

        backend = VLLMBackend(endpoint="http://localhost:8000", model="test-model")
        assert backend.validate_environment() is False

    @patch("ai_energy_benchmarks.backends.vllm.requests.get")
    def test_health_check_success(self, mock_get):
        """Test successful health check."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_get.return_value = mock_response

        backend = VLLMBackend(endpoint="http://localhost:8000", model="test-model")
        assert backend.health_check() is True

    @patch("ai_energy_benchmarks.backends.vllm.requests.post")
    def test_run_inference_success(self, mock_post):
        """Test successful inference."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "test response"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
        }
        mock_post.return_value = mock_response

        backend = VLLMBackend(endpoint="http://localhost:8000", model="test-model")
        result = backend.run_inference("test prompt", enable_streaming=False)

        assert result["success"] is True
        assert result["text"] == "test response"
        assert result["total_tokens"] == 30

    @patch("ai_energy_benchmarks.backends.vllm.requests.post")
    def test_run_inference_timeout(self, mock_post):
        """Test inference timeout."""
        import requests

        mock_post.side_effect = requests.exceptions.Timeout()

        backend = VLLMBackend(endpoint="http://localhost:8000", model="test-model")
        result = backend.run_inference("test prompt")

        assert result["success"] is False
        assert "timeout" in result["error"].lower()

    def test_get_endpoint_info(self):
        """Test getting endpoint info."""
        backend = VLLMBackend(endpoint="http://localhost:8000", model="test-model")
        info = backend.get_endpoint_info()

        assert info["backend"] == "vllm"
        assert info["model"] == "test-model"
        assert info["endpoint"] == "http://localhost:8000"

    @patch("ai_energy_benchmarks.backends.vllm.requests.post")
    def test_run_inference_with_streaming_ttft(self, mock_post):
        """Test inference with streaming to capture TTFT."""
        # Mock SSE streaming response
        mock_response = Mock()
        mock_response.status_code = 200

        # Simulate SSE stream with multiple chunks
        sse_lines = [
            b'data: {"choices":[{"delta":{"content":"Hello"}}],"usage":{}}',
            b'data: {"choices":[{"delta":{"content":" world"}}],"usage":{}}',
            b'data: {"choices":[{"delta":{"content":"!"}}],"usage":{"prompt_tokens":10,"completion_tokens":3,"total_tokens":13}}',
            b"data: [DONE]",
        ]
        mock_response.iter_lines.return_value = iter(sse_lines)
        mock_post.return_value = mock_response

        backend = VLLMBackend(endpoint="http://localhost:8000", model="test-model")
        result = backend.run_inference("test prompt", enable_streaming=True)

        assert result["success"] is True
        assert result["text"] == "Hello world!"
        assert result["time_to_first_token"] is not None
        assert result["time_to_first_token"] > 0
        assert result["total_tokens"] == 13

    @patch("ai_energy_benchmarks.backends.vllm.requests.post")
    def test_run_inference_non_streaming_no_ttft(self, mock_post):
        """Test non-streaming inference returns None for TTFT."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "test response"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
        }
        mock_post.return_value = mock_response

        backend = VLLMBackend(endpoint="http://localhost:8000", model="test-model")
        result = backend.run_inference("test prompt", enable_streaming=False)

        assert result["success"] is True
        assert result["text"] == "test response"
        assert result["time_to_first_token"] is None

    @patch("ai_energy_benchmarks.backends.vllm.requests.post")
    def test_run_inference_streaming_error_handling(self, mock_post):
        """Test error handling during streaming."""
        mock_response = Mock()
        mock_response.status_code = 200
        # Simulate malformed SSE stream
        mock_response.iter_lines.return_value = iter([b"invalid json", b"data: [DONE]"])
        mock_post.return_value = mock_response

        backend = VLLMBackend(endpoint="http://localhost:8000", model="test-model")
        result = backend.run_inference("test prompt", enable_streaming=True)

        # Should still succeed but with empty content
        assert result["success"] is True
        assert result["text"] == ""
