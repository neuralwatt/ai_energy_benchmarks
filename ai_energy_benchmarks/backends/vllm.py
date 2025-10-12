"""vLLM backend implementation for high-performance inference."""

import time
from typing import Any, Dict, Optional, cast

import requests  # type: ignore[import-untyped]

from ai_energy_benchmarks.backends.base import Backend


class VLLMBackend(Backend):
    """vLLM backend for high-performance LLM serving."""

    def __init__(
        self,
        endpoint: str,
        model: str,
        timeout: int = 300,
        use_harmony: Optional[bool] = None,
    ):
        """Initialize vLLM backend.

        Args:
            endpoint: vLLM server endpoint (e.g., "http://localhost:8000/v1")
            model: Model name (e.g., "openai/gpt-oss-120b")
            timeout: Request timeout in seconds
            use_harmony: Enable Harmony formatting for gpt-oss models (auto-detects if None)
        """
        self.endpoint = endpoint.rstrip("/v1").rstrip("/")
        self.model = model
        self.timeout = timeout

        # Auto-detect Harmony formatting for gpt-oss models
        detected_use_harmony = (
            use_harmony if use_harmony is not None else "gpt-oss" in model.lower()
        )
        self.use_harmony: bool = bool(detected_use_harmony)

    def format_harmony_prompt(self, text: str, reasoning_effort: str = "high") -> str:
        """Format a prompt using OpenAI Harmony formatting for gpt-oss models.

        Args:
            text: The user's question/prompt text
            reasoning_effort: Reasoning level (low, medium, high)

        Returns:
            Harmony-formatted prompt with system message and user message
        """
        # Harmony format structure with system and user messages
        harmony_prompt = (
            "<|start|>system<|message|>"
            "You are a helpful AI assistant.\n"
            f"Reasoning: {reasoning_effort}\n"
            "# Valid channels: analysis, commentary, final"
            "<|end|>\n"
            f"<|start|>user<|message|>{text}<|end|>"
        )
        return harmony_prompt

    def validate_environment(self) -> bool:
        """Check if vLLM is running and model is loaded.

        Returns:
            bool: True if vLLM is available with the correct model
        """
        try:
            response = requests.get(f"{self.endpoint}/v1/models", timeout=10)
            status_code = cast(int, response.status_code)
            if status_code != 200:
                return False

            models = response.json()
            model_ids = [m.get("id", "") for m in models.get("data", [])]
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
            response = requests.get(f"{self.endpoint}/health", timeout=5)
            status_code = cast(int, response.status_code)
            return status_code == 200
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
            "validated": self.validate_environment(),
        }

    def run_inference(
        self,
        prompt: str,
        max_tokens: int = 100,
        temperature: float = 0.7,
        reasoning_params: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Run inference on a single prompt via vLLM OpenAI-compatible API.

        Args:
            prompt: Input text prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            reasoning_params: Optional reasoning parameters for thinking models
            **kwargs: Additional generation parameters

        Returns:
            Dict with response text and metadata
        """
        start_time = time.time()

        # Apply Harmony formatting if enabled for gpt-oss models
        formatted_prompt = prompt
        if self.use_harmony:
            reasoning_effort = "high"  # Default
            if reasoning_params and "reasoning_effort" in reasoning_params:
                reasoning_effort = reasoning_params["reasoning_effort"]
            formatted_prompt = self.format_harmony_prompt(prompt, reasoning_effort)
            print(f"  Using Harmony format with {reasoning_effort} reasoning")

        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": formatted_prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        # Add reasoning parameters via extra_body if provided
        if reasoning_params:
            # vLLM/OpenAI API supports extra_body for custom parameters
            extra_body: Dict[str, Any] = {}

            # Map reasoning effort to model-specific parameters
            if "reasoning_effort" in reasoning_params:
                effort = reasoning_params["reasoning_effort"]
                extra_body["reasoning_effort"] = effort
                print(f"Using reasoning effort: {effort}")

            # Pass through other reasoning parameters
            for key, value in reasoning_params.items():
                if key not in extra_body:
                    extra_body[key] = value

            if extra_body:
                payload["extra_body"] = extra_body

        # Add any additional kwargs
        payload.update(kwargs)

        try:
            response = requests.post(
                f"{self.endpoint}/v1/chat/completions", json=payload, timeout=self.timeout
            )
            response.raise_for_status()

            result = response.json()
            end_time = time.time()

            # Extract response data
            choice = result.get("choices", [{}])[0]
            message = choice.get("message", {})
            completion_text = message.get("content", "")

            # Extract usage stats
            usage = result.get("usage", {})

            return {
                "text": completion_text,
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0),
                "latency_seconds": end_time - start_time,
                "model": self.model,
                "success": True,
                "error": None,
            }

        except requests.exceptions.Timeout:
            return {
                "text": "",
                "success": False,
                "error": "Request timeout",
                "latency_seconds": time.time() - start_time,
            }
        except Exception as e:
            return {
                "text": "",
                "success": False,
                "error": str(e),
                "latency_seconds": time.time() - start_time,
            }
