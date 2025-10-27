"""vLLM backend implementation for high-performance inference."""

import time
import warnings
from typing import Any, Dict, Optional, cast

import requests  # type: ignore[import-untyped]

from ai_energy_benchmarks.backends.base import Backend
from ai_energy_benchmarks.formatters.registry import FormatterRegistry


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
            use_harmony: DEPRECATED - Enable Harmony formatting for gpt-oss models (auto-detects if None)
        """
        self.endpoint = endpoint.rstrip("/v1").rstrip("/")
        self.model = model
        self.timeout = timeout

        # DEPRECATED: Backward compatibility
        self._legacy_use_harmony: Optional[bool]
        if use_harmony is not None:
            warnings.warn(
                "use_harmony parameter is deprecated. "
                "Reasoning format is now auto-detected via FormatterRegistry. "
                "This parameter will be removed in v2.0.",
                DeprecationWarning,
                stacklevel=2,
            )
            self._legacy_use_harmony = use_harmony
        else:
            self._legacy_use_harmony = None

        # Auto-detect Harmony formatting for gpt-oss models (legacy)
        detected_use_harmony = (
            use_harmony if use_harmony is not None else "gpt-oss" in model.lower()
        )
        self.use_harmony: bool = bool(detected_use_harmony)

        # Initialize formatter registry
        self.formatter_registry = FormatterRegistry()
        self.formatter = self.formatter_registry.get_formatter(model)

    def format_harmony_prompt(self, text: str, reasoning_effort: str = "high") -> str:
        """DEPRECATED: Format a prompt using OpenAI Harmony formatting for gpt-oss models.

        Args:
            text: The user's question/prompt text
            reasoning_effort: Reasoning level (low, medium, high)

        Returns:
            Harmony-formatted prompt with system message and user message
        """
        warnings.warn(
            "format_harmony_prompt() is deprecated. Use FormatterRegistry instead.",
            DeprecationWarning,
            stacklevel=2,
        )
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
        enable_streaming: bool = True,
        **kwargs,
    ) -> Dict[str, Any]:
        """Run inference on a single prompt via vLLM OpenAI-compatible API.

        Args:
            prompt: Input text prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            reasoning_params: Optional reasoning parameters for thinking models
            enable_streaming: Enable streaming for TTFT tracking (default: True)
            **kwargs: Additional generation parameters

        Returns:
            Dict with response text and metadata (includes time_to_first_token if streaming enabled)
        """
        start_time = time.time()
        ttft: Optional[float] = None

        # Format prompt using registry formatter (new approach)
        formatted_prompt = prompt
        extra_gen_params: Dict[str, Any] = {}

        if self.formatter:
            # Use new formatter from registry
            formatted_prompt = self.formatter.format_prompt(prompt, reasoning_params)
            extra_gen_params = self.formatter.get_generation_params(reasoning_params)
        elif self._legacy_use_harmony:
            # DEPRECATED: Legacy Harmony formatting
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
            "stream": enable_streaming,  # Enable streaming for TTFT
        }

        # Add formatter-provided parameters
        if extra_gen_params:
            payload["extra_body"] = extra_gen_params

        # Add reasoning parameters via extra_body if provided (legacy path)
        if reasoning_params and not self.formatter:
            # vLLM/OpenAI API supports extra_body for custom parameters
            extra_body = cast(Dict[str, Any], payload.get("extra_body", {}))

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
            if enable_streaming:
                # Streaming path for TTFT tracking
                import json

                response = requests.post(
                    f"{self.endpoint}/v1/chat/completions",
                    json=payload,
                    timeout=self.timeout,
                    stream=True,
                )
                response.raise_for_status()

                completion_text = ""
                first_chunk = True
                usage_data: Dict[str, Any] = {}

                # Process SSE stream
                for line in response.iter_lines():
                    if not line:
                        continue

                    # Parse SSE format: "data: {...}"
                    if line.startswith(b"data: "):
                        if line == b"data: [DONE]":
                            break

                        try:
                            data = json.loads(line[6:])  # Skip "data: " prefix

                            # Capture TTFT on first chunk with content
                            if first_chunk:
                                delta = data.get("choices", [{}])[0].get("delta", {})
                                content = delta.get("content", "")
                                if content:
                                    ttft = time.time() - start_time
                                    first_chunk = False

                            # Extract delta content
                            delta = data.get("choices", [{}])[0].get("delta", {})
                            content = delta.get("content", "")
                            completion_text += content

                            # Capture usage stats if available
                            if "usage" in data:
                                usage_data = data["usage"]

                        except json.JSONDecodeError:
                            continue

                end_time = time.time()

                return {
                    "text": completion_text,
                    "prompt_tokens": usage_data.get("prompt_tokens", 0),
                    "completion_tokens": usage_data.get("completion_tokens", 0),
                    "total_tokens": usage_data.get("total_tokens", 0),
                    "latency_seconds": end_time - start_time,
                    "time_to_first_token": ttft,
                    "model": self.model,
                    "success": True,
                    "error": None,
                }
            else:
                # Non-streaming path (legacy)
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
                    "time_to_first_token": None,
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
                "time_to_first_token": None,
            }
        except Exception as e:
            return {
                "text": "",
                "success": False,
                "error": str(e),
                "latency_seconds": time.time() - start_time,
                "time_to_first_token": None,
            }
