"""vLLM backend implementation for high-performance inference."""

import requests
import time
from typing import Dict, Any, Optional
from ai_energy_benchmarks.backends.base import Backend


class VLLMBackend(Backend):
    """vLLM backend for high-performance LLM serving."""

    def __init__(self, endpoint: str, model: str, timeout: int = 300):
        """Initialize vLLM backend.

        Args:
            endpoint: vLLM server endpoint (e.g., "http://localhost:8000/v1")
            model: Model name (e.g., "openai/gpt-oss-120b")
            timeout: Request timeout in seconds
        """
        self.endpoint = endpoint.rstrip('/v1').rstrip('/')
        self.model = model
        self.timeout = timeout

    def validate_environment(self) -> bool:
        """Check if vLLM is running and model is loaded.

        Returns:
            bool: True if vLLM is available with the correct model
        """
        try:
            response = requests.get(
                f"{self.endpoint}/v1/models",
                timeout=10
            )
            if response.status_code != 200:
                return False

            models = response.json()
            model_ids = [m.get('id', '') for m in models.get('data', [])]
            return self.model in model_ids

        except Exception as e:
            print(f"vLLM validation error: {e}")
            return False

    def health_check(self) -> bool:
        """Check if vLLM server is healthy.

        Returns:
            bool: True if server is healthy
        """
        try:
            response = requests.get(
                f"{self.endpoint}/health",
                timeout=5
            )
            return response.status_code == 200
        except Exception as e:
            print(f"vLLM health check error: {e}")
            return False

    def get_endpoint_info(self) -> Dict[str, Any]:
        """Get backend endpoint information.

        Returns:
            Dict with backend configuration and status
        """
        return {
            "backend": "vllm",
            "endpoint": self.endpoint,
            "model": self.model,
            "healthy": self.health_check(),
            "validated": self.validate_environment()
        }

    def run_inference(
        self,
        prompt: str,
        max_tokens: int = 100,
        temperature: float = 0.7,
        **kwargs
    ) -> Dict[str, Any]:
        """Run inference on a single prompt via vLLM OpenAI-compatible API.

        Args:
            prompt: Input text prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            **kwargs: Additional generation parameters

        Returns:
            Dict with response text and metadata
        """
        start_time = time.time()

        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
            **kwargs
        }

        try:
            response = requests.post(
                f"{self.endpoint}/v1/chat/completions",
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()

            result = response.json()
            end_time = time.time()

            # Extract response data
            choice = result.get('choices', [{}])[0]
            message = choice.get('message', {})
            completion_text = message.get('content', '')

            # Extract usage stats
            usage = result.get('usage', {})

            return {
                "text": completion_text,
                "prompt_tokens": usage.get('prompt_tokens', 0),
                "completion_tokens": usage.get('completion_tokens', 0),
                "total_tokens": usage.get('total_tokens', 0),
                "latency_seconds": end_time - start_time,
                "model": self.model,
                "success": True,
                "error": None
            }

        except requests.exceptions.Timeout:
            return {
                "text": "",
                "success": False,
                "error": "Request timeout",
                "latency_seconds": time.time() - start_time
            }
        except Exception as e:
            return {
                "text": "",
                "success": False,
                "error": str(e),
                "latency_seconds": time.time() - start_time
            }
