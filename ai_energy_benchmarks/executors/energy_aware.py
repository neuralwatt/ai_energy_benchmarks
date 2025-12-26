"""Energy-aware executor for running benchmarks with energy metric collection.

This module provides an executor that sends requests directly to OpenAI-compatible
APIs and captures energy metrics from the response if available.
"""

import asyncio
import hashlib
import random
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

try:
    import aiohttp

    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False

from ..profiles import LoadProfileConfig


@dataclass
class RequestResult:
    """Result from a single API request."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    request_duration_seconds: float  # Wall clock time for the request
    # Energy fields - None if not present in response
    energy_joules: Optional[float] = None
    energy_kwh: Optional[float] = None
    avg_power_watts: Optional[float] = None
    inference_duration_seconds: Optional[float] = None  # From energy.duration_seconds
    attribution_method: Optional[str] = None
    attribution_ratio: Optional[float] = None
    # Error tracking
    error: Optional[str] = None
    status_code: Optional[int] = None


@dataclass
class ProfileResult:
    """Aggregated result from running a profile."""

    profile_name: str
    model: str
    endpoint: str
    timestamp: datetime
    request_count: int
    successful_requests: int
    failed_requests: int
    concurrency: int
    # Token metrics
    total_tokens: int
    total_prompt_tokens: int
    total_completion_tokens: int
    # Timing
    total_wall_clock_seconds: float  # Total time to run all requests
    total_inference_seconds: float  # Sum of inference durations
    tokens_per_second: float
    # Energy metrics - None if API doesn't provide energy data
    total_energy_joules: Optional[float] = None
    total_energy_kwh: Optional[float] = None
    wh_per_request: Optional[float] = None
    tokens_per_joule: Optional[float] = None
    avg_power_watts: Optional[float] = None
    energy_available: bool = False
    # Individual results for detailed analysis
    individual_results: List[RequestResult] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for CSV/JSON output."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "endpoint": self.endpoint,
            "model": self.model,
            "profile": self.profile_name,
            "requests": self.request_count,
            "successful": self.successful_requests,
            "failed": self.failed_requests,
            "concurrency": self.concurrency,
            "total_tokens": self.total_tokens,
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "wall_clock_seconds": round(self.total_wall_clock_seconds, 3),
            "inference_seconds": round(self.total_inference_seconds, 3),
            "tokens_per_second": round(self.tokens_per_second, 2),
            "total_energy_j": (
                round(self.total_energy_joules, 2) if self.total_energy_joules else None
            ),
            "total_energy_kwh": self.total_energy_kwh,
            "wh_per_request": round(self.wh_per_request, 6) if self.wh_per_request else None,
            "tokens_per_joule": round(self.tokens_per_joule, 4) if self.tokens_per_joule else None,
            "avg_power_watts": round(self.avg_power_watts, 1) if self.avg_power_watts else None,
            "energy_available": self.energy_available,
        }


class EnergyAwareExecutor:
    """Executor that sends requests directly and captures energy metrics.

    This executor sends HTTP requests to OpenAI-compatible APIs and parses
    the 'energy' field from responses if present. It supports concurrent
    requests based on the profile configuration.

    Example:
        executor = EnergyAwareExecutor(seed=42)
        result = await executor.run(
            profile=get_profile("light"),
            endpoint="https://api.neuralwatt.com/v1",
            model="Qwen/Qwen3-Coder-480B-A35B-Instruct",
            api_key="sk-xxxxx"
        )
        print(f"Energy per request: {result.wh_per_request} Wh")
    """

    # Base prompts for generating varied requests
    BASE_PROMPTS = [
        "Explain the concept of machine learning and its applications in modern technology.",
        "Describe the process of photosynthesis and its importance in the ecosystem.",
        "What are the key principles of object-oriented programming?",
        "Discuss the history and significance of the Renaissance period.",
        "Explain the fundamentals of quantum physics and quantum computing.",
        "Describe the structure and function of DNA in living organisms.",
        "What are the main causes and effects of climate change?",
        "Explain the principles of supply and demand in economics.",
        "Describe the process of cellular respiration in biological systems.",
        "What are the key components of a computer operating system?",
        "Explain the concept of neural networks and deep learning.",
        "Describe the major events of World War II and their global impact.",
        "What are the fundamental principles of thermodynamics?",
        "Explain how blockchain technology works and its applications.",
        "Describe the structure and function of the human immune system.",
    ]

    def __init__(self, seed: Optional[int] = None):
        """Initialize the executor.

        Args:
            seed: Random seed for reproducible prompt generation
        """
        if not AIOHTTP_AVAILABLE:
            raise ImportError(
                "aiohttp is required for EnergyAwareExecutor. Install it with: pip install aiohttp"
            )
        self.seed = seed
        self._random = random.Random(seed) if seed is not None else random.Random()

    def _generate_prompts(self, count: int, input_token_range: tuple) -> List[str]:
        """Generate deterministic prompts for the benchmark.

        Args:
            count: Number of prompts to generate
            input_token_range: (min_tokens, max_tokens) for prompt length

        Returns:
            List of prompt strings
        """
        prompts = []
        min_tokens, max_tokens = input_token_range

        # Extended content blocks for building longer prompts (~50 tokens each)
        content_blocks = [
            "Please provide detailed examples and explanations with specific use cases.",
            "Include practical applications and real-world scenarios where this applies.",
            "Discuss the historical context, development, and evolution over time.",
            "Explain the technical details, mechanisms, and underlying principles involved.",
            "Compare and contrast with related concepts, alternatives, and similar approaches.",
            "Analyze the advantages, disadvantages, trade-offs, and considerations.",
            "Provide step-by-step instructions, processes, and implementation guidelines.",
            "Include relevant statistics, data, research findings, and empirical evidence.",
            "Discuss future trends, developments, predictions, and potential innovations.",
            "Explain the impact on society, industry, economics, and various stakeholders.",
            "Describe the key components, architecture, structure, and organization.",
            "Outline the methodology, approach, framework, and best practices.",
            "Address common challenges, problems, limitations, and how to overcome them.",
            "Discuss security implications, risks, vulnerabilities, and mitigation strategies.",
            "Explain performance characteristics, optimization techniques, and efficiency.",
            "Cover testing strategies, validation methods, and quality assurance approaches.",
            "Describe integration patterns, compatibility considerations, and interoperability.",
            "Discuss scalability aspects, growth considerations, and capacity planning.",
            "Explain maintenance requirements, operational procedures, and lifecycle management.",
            "Address regulatory compliance, standards, certifications, and legal considerations.",
        ]

        for i in range(count):
            base_prompt = self.BASE_PROMPTS[i % len(self.BASE_PROMPTS)]

            # Add unique prefix to prevent caching
            if self.seed is not None:
                unique_id = hashlib.md5(f"{self.seed}_{i}".encode()).hexdigest()[:8]
            else:
                unique_id = hashlib.md5(f"{time.time()}_{i}".encode()).hexdigest()[:8]

            unique_prefix = f"[Request {i + 1}/{count}, ID:{unique_id}] "
            prompt = unique_prefix + base_prompt

            # Extend prompt to reach target token count
            target_tokens = self._random.randint(min_tokens, max_tokens)
            estimated_tokens = len(prompt) // 4  # Rough estimate: ~4 chars per token

            # Add content blocks to reach target
            block_index = 0
            while estimated_tokens < target_tokens:
                block = content_blocks[block_index % len(content_blocks)]
                # Add variation to repeated blocks
                if block_index >= len(content_blocks):
                    variation = f" (aspect {block_index // len(content_blocks) + 1})"
                    block = block.rstrip(".") + variation + "."
                prompt += " " + block
                estimated_tokens = len(prompt) // 4
                block_index += 1
                # Safety limit to prevent infinite loops
                if block_index > 200:
                    break

            prompts.append(prompt)

        return prompts

    async def run(
        self,
        profile: LoadProfileConfig,
        endpoint: str,
        model: str,
        api_key: str,
        timeout_seconds: int = 120,
    ) -> ProfileResult:
        """Execute a profile and return aggregated results.

        Args:
            profile: Load profile configuration
            endpoint: API endpoint URL (e.g., "https://api.neuralwatt.com/v1")
            model: Model name to use
            api_key: API key for authentication
            timeout_seconds: Timeout for individual requests

        Returns:
            ProfileResult with aggregated metrics
        """
        # Generate prompts
        prompts = self._generate_prompts(profile.request_count, profile.input_token_range)

        # Get max_tokens from profile's output_token_range (use upper bound)
        max_tokens = profile.output_token_range[1]

        # Normalize endpoint URL
        endpoint = endpoint.rstrip("/")
        if not endpoint.endswith("/v1"):
            if "/v1" not in endpoint:
                endpoint = f"{endpoint}/v1"

        # Create semaphore for concurrency control
        semaphore = asyncio.Semaphore(profile.concurrency)

        # Track wall clock time
        start_time = time.perf_counter()

        # Create timeout for requests
        timeout = aiohttp.ClientTimeout(total=timeout_seconds)

        async with aiohttp.ClientSession(timeout=timeout) as session:
            tasks = [
                self._send_request(
                    session, semaphore, endpoint, model, api_key, prompt, i, max_tokens
                )
                for i, prompt in enumerate(prompts)
            ]
            results: List[RequestResult] = await asyncio.gather(*tasks)

        end_time = time.perf_counter()
        wall_clock_seconds = end_time - start_time

        return self._aggregate_results(
            profile_name=profile.name,
            model=model,
            endpoint=endpoint,
            concurrency=profile.concurrency,
            results=results,
            wall_clock_seconds=wall_clock_seconds,
        )

    async def _send_request(
        self,
        session: "aiohttp.ClientSession",
        semaphore: asyncio.Semaphore,
        endpoint: str,
        model: str,
        api_key: str,
        prompt: str,
        index: int,
        max_tokens: int = 500,
    ) -> RequestResult:
        """Send a single request and parse response.

        Args:
            session: aiohttp session
            semaphore: Concurrency limiter
            endpoint: API endpoint
            model: Model name
            api_key: API key
            prompt: Prompt text
            index: Request index for logging
            max_tokens: Maximum tokens to generate

        Returns:
            RequestResult with parsed metrics
        """
        async with semaphore:
            request_start = time.perf_counter()
            try:
                async with session.post(
                    f"{endpoint}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": model,
                        "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": max_tokens,
                    },
                ) as response:
                    request_duration = time.perf_counter() - request_start
                    status_code = response.status

                    if status_code != 200:
                        error_text = await response.text()
                        return RequestResult(
                            prompt_tokens=0,
                            completion_tokens=0,
                            total_tokens=0,
                            request_duration_seconds=request_duration,
                            error=f"HTTP {status_code}: {error_text[:200]}",
                            status_code=status_code,
                        )

                    data = await response.json()

                    # Parse usage
                    usage = data.get("usage", {})
                    prompt_tokens = usage.get("prompt_tokens", 0)
                    completion_tokens = usage.get("completion_tokens", 0)
                    total_tokens = usage.get("total_tokens", prompt_tokens + completion_tokens)

                    # Parse energy (may be None)
                    energy = data.get("energy")

                    if energy:
                        return RequestResult(
                            prompt_tokens=prompt_tokens,
                            completion_tokens=completion_tokens,
                            total_tokens=total_tokens,
                            request_duration_seconds=request_duration,
                            energy_joules=energy.get("energy_joules"),
                            energy_kwh=energy.get("energy_kwh"),
                            avg_power_watts=energy.get("avg_power_watts"),
                            inference_duration_seconds=energy.get("duration_seconds"),
                            attribution_method=energy.get("attribution_method"),
                            attribution_ratio=energy.get("attribution_ratio"),
                            status_code=status_code,
                        )
                    else:
                        return RequestResult(
                            prompt_tokens=prompt_tokens,
                            completion_tokens=completion_tokens,
                            total_tokens=total_tokens,
                            request_duration_seconds=request_duration,
                            status_code=status_code,
                        )

            except asyncio.TimeoutError:
                request_duration = time.perf_counter() - request_start
                return RequestResult(
                    prompt_tokens=0,
                    completion_tokens=0,
                    total_tokens=0,
                    request_duration_seconds=request_duration,
                    error="Request timeout",
                )
            except Exception as e:
                request_duration = time.perf_counter() - request_start
                return RequestResult(
                    prompt_tokens=0,
                    completion_tokens=0,
                    total_tokens=0,
                    request_duration_seconds=request_duration,
                    error=str(e),
                )

    def _aggregate_results(
        self,
        profile_name: str,
        model: str,
        endpoint: str,
        concurrency: int,
        results: List[RequestResult],
        wall_clock_seconds: float,
    ) -> ProfileResult:
        """Aggregate individual results into profile summary.

        Args:
            profile_name: Name of the profile
            model: Model name
            endpoint: API endpoint
            concurrency: Concurrency level used
            results: List of individual request results
            wall_clock_seconds: Total wall clock time

        Returns:
            ProfileResult with aggregated metrics
        """
        successful = [r for r in results if r.error is None]
        failed = [r for r in results if r.error is not None]

        total_tokens = sum(r.total_tokens for r in successful)
        total_prompt_tokens = sum(r.prompt_tokens for r in successful)
        total_completion_tokens = sum(r.completion_tokens for r in successful)

        # Calculate inference duration from energy data if available, else use request duration
        total_inference_seconds = sum(
            (
                r.inference_duration_seconds
                if r.inference_duration_seconds
                else r.request_duration_seconds
            )
            for r in successful
        )

        # Throughput based on wall clock time (real-world throughput with concurrency)
        tokens_per_second = (
            total_completion_tokens / wall_clock_seconds if wall_clock_seconds > 0 else 0
        )

        # Check if energy data is available
        has_energy = any(r.energy_joules is not None for r in successful)

        if has_energy and successful:
            energy_results = [r for r in successful if r.energy_joules is not None]
            total_energy_j = sum(
                r.energy_joules for r in energy_results if r.energy_joules is not None
            )
            total_energy_kwh = sum(r.energy_kwh for r in energy_results if r.energy_kwh)
            power_readings = [r.avg_power_watts for r in energy_results if r.avg_power_watts]
            avg_power = sum(power_readings) / len(power_readings) if power_readings else None

            # Calculate derived metrics
            wh_per_request = (
                (total_energy_kwh * 1000) / len(energy_results) if total_energy_kwh else None
            )
            # Use total_tokens (prompt + completion) since energy is consumed for both prefill and decode
            tokens_per_joule = total_tokens / total_energy_j if total_energy_j > 0 else None
        else:
            total_energy_j = None
            total_energy_kwh = None
            avg_power = None
            wh_per_request = None
            tokens_per_joule = None

        return ProfileResult(
            profile_name=profile_name,
            model=model,
            endpoint=endpoint,
            timestamp=datetime.now(timezone.utc),
            request_count=len(results),
            successful_requests=len(successful),
            failed_requests=len(failed),
            concurrency=concurrency,
            total_tokens=total_tokens,
            total_prompt_tokens=total_prompt_tokens,
            total_completion_tokens=total_completion_tokens,
            total_wall_clock_seconds=wall_clock_seconds,
            total_inference_seconds=total_inference_seconds,
            tokens_per_second=tokens_per_second,
            total_energy_joules=total_energy_j,
            total_energy_kwh=total_energy_kwh,
            wh_per_request=wh_per_request,
            tokens_per_joule=tokens_per_joule,
            avg_power_watts=avg_power,
            energy_available=has_energy,
            individual_results=results,
        )


def run_sync(
    profile: LoadProfileConfig,
    endpoint: str,
    model: str,
    api_key: str,
    seed: Optional[int] = None,
    timeout_seconds: int = 120,
) -> ProfileResult:
    """Synchronous wrapper for running the executor.

    Args:
        profile: Load profile configuration
        endpoint: API endpoint URL
        model: Model name
        api_key: API key
        seed: Random seed for reproducibility
        timeout_seconds: Request timeout

    Returns:
        ProfileResult with aggregated metrics
    """
    executor = EnergyAwareExecutor(seed=seed)
    return asyncio.run(executor.run(profile, endpoint, model, api_key, timeout_seconds))
