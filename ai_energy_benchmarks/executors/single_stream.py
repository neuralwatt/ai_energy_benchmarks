"""Single-stream executor for on-box per-request energy measurement.

This executor is the on-box equivalent of `EnergyAwareExecutor`: instead of
reading per-request energy from an inference gateway's response (the
`energy` field exposed by the NW gateway), it samples GPU power locally
via `nvidia-smi` while each request is in flight and computes
`energy_joules = avg_power_watts * duration_seconds` per request.

This attribution is only valid at concurrency=1 — when requests don't
overlap on the GPU, the sampled power belongs entirely to the request
in flight. The executor enforces that by ignoring the profile's
`concurrency` setting and running prompts serially.

Workflow per request:
    1. Start a background `GpuPowerSampler` thread.
    2. POST one chat-completion to the vLLM endpoint synchronously.
    3. Stop the sampler. Compute energy = avg_power * wall_clock_duration.
    4. Append a `RequestResult` populated from the response usage and the
       local power measurement.

Use this for benchmarking raw vLLM (or any OpenAI-compatible endpoint
that does not expose per-request energy in the response). For
deployments where the endpoint exposes `energy` in the response, use
`EnergyAwareExecutor` instead.
"""

import json
import subprocess
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from typing import List, Optional

from ..profiles import LoadProfileConfig
from .energy_aware import (
    EnergyAwareExecutor,
    ProfileResult,
    RequestResult,
)

DEFAULT_POWER_SAMPLE_INTERVAL_S = 0.1
"""Default GPU power sampling interval in seconds.

100 ms gives enough samples for short requests (1-2 s) without measurable
overhead. Lower intervals don't improve accuracy because nvidia-smi
itself caches its underlying NVML readings on a similar cadence.
"""

NVIDIA_SMI_TIMEOUT_S = 5
"""Timeout for a single nvidia-smi power query.

If a query stalls beyond this, we drop the sample and continue. nvidia-smi
occasionally blocks on contended GPUs; dropping a sample is preferable to
hanging the entire benchmark.
"""


class GpuPowerSampler:
    """Sample total GPU power draw in a background thread.

    Calls `nvidia-smi --query-gpu=power.draw` at a fixed interval and
    averages all readings. Multi-GPU systems return the power of the
    first GPU (index 0); a future enhancement could sum across GPUs for
    multi-GPU vLLM tensor-parallel runs.

    Not thread-safe across multiple concurrent users — the intended
    usage is one sampler per request, in a single-stream loop.
    """

    def __init__(self, interval_s: float = DEFAULT_POWER_SAMPLE_INTERVAL_S):
        self.interval_s = interval_s
        self.samples: List[float] = []
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        self.samples = []
        self._stop.clear()
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)

    def _sample_loop(self) -> None:
        while not self._stop.is_set():
            try:
                result = subprocess.run(
                    ["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits"],
                    capture_output=True,
                    text=True,
                    timeout=NVIDIA_SMI_TIMEOUT_S,
                )
                if result.returncode == 0:
                    first_line = result.stdout.strip().split("\n")[0]
                    self.samples.append(float(first_line))
            except (ValueError, subprocess.TimeoutExpired, FileNotFoundError):
                # ValueError: parse failure
                # TimeoutExpired: nvidia-smi stalled
                # FileNotFoundError: nvidia-smi not on PATH (e.g. mock test env)
                pass
            self._stop.wait(self.interval_s)

    @property
    def avg_power_watts(self) -> float:
        if not self.samples:
            return 0.0
        return sum(self.samples) / len(self.samples)


class SingleStreamExecutor:
    """Run prompts one at a time and measure per-request GPU energy locally.

    Forces concurrency=1 regardless of the profile's `concurrency` field,
    because per-request energy attribution via power sampling is only
    valid when requests don't overlap on the GPU.

    Example:
        from ai_energy_benchmarks.profiles.definitions import get_profile
        executor = SingleStreamExecutor(seed=42)
        result = executor.run(
            profile=get_profile("single_stream_light"),
            endpoint="http://localhost:8000/v1",
            model="Qwen/Qwen3-32B",
        )
        for r in result.individual_results:
            print(f"{r.total_tokens} tok, {r.energy_joules:.1f} J")
    """

    def __init__(
        self,
        seed: Optional[int] = None,
        power_sample_interval_s: float = DEFAULT_POWER_SAMPLE_INTERVAL_S,
    ):
        """Initialize the executor.

        Args:
            seed: Random seed for reproducible prompt generation.
            power_sample_interval_s: How often to sample GPU power during
                a request, in seconds. Default 0.1 s.
        """
        self.seed = seed
        self.power_sample_interval_s = power_sample_interval_s
        # Reuse EnergyAwareExecutor's prompt generation so single-stream and
        # gateway runs draw from the same prompt pool.
        self._prompt_helper = EnergyAwareExecutor(seed=seed)

    def run(
        self,
        profile: LoadProfileConfig,
        endpoint: str,
        model: str,
        api_key: Optional[str] = None,
        timeout_seconds: int = 120,
    ) -> ProfileResult:
        """Execute the profile single-stream and return aggregated results.

        Args:
            profile: Load profile config. `concurrency` is ignored (forced to 1).
            endpoint: vLLM endpoint URL (e.g. "http://localhost:8000/v1").
            model: Model name to use in the request.
            api_key: Optional bearer token. Most local vLLM deployments
                don't require one; pass None to skip the header.
            timeout_seconds: Per-request HTTP timeout.

        Returns:
            ProfileResult with `individual_results` populated and energy
            fields computed from local power sampling.
        """
        prompts = self._prompt_helper._generate_prompts(
            profile.request_count, profile.input_token_range
        )
        max_tokens = profile.output_token_range[1]

        endpoint = endpoint.rstrip("/")
        if not endpoint.endswith("/v1") and "/v1" not in endpoint:
            endpoint = f"{endpoint}/v1"

        results: List[RequestResult] = []
        sampler = GpuPowerSampler(interval_s=self.power_sample_interval_s)

        wall_start = time.perf_counter()
        for prompt in prompts:
            results.append(
                self._send_one_request(
                    sampler, endpoint, model, api_key, prompt, max_tokens, timeout_seconds
                )
            )
        wall_clock_seconds = time.perf_counter() - wall_start

        return self._aggregate_results(
            profile_name=profile.name,
            model=model,
            endpoint=endpoint,
            results=results,
            wall_clock_seconds=wall_clock_seconds,
        )

    def _send_one_request(
        self,
        sampler: GpuPowerSampler,
        endpoint: str,
        model: str,
        api_key: Optional[str],
        prompt: str,
        max_tokens: int,
        timeout_seconds: int,
    ) -> RequestResult:
        """Send one request with GPU power sampling and return a RequestResult."""
        payload = json.dumps(
            {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
            }
        ).encode()
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        req = urllib.request.Request(f"{endpoint}/chat/completions", data=payload, headers=headers)

        sampler.start()
        t0 = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=timeout_seconds) as resp:
                body = resp.read()
                status = resp.status
        except urllib.error.HTTPError as e:
            duration = time.perf_counter() - t0
            sampler.stop()
            return RequestResult(
                prompt_tokens=0,
                completion_tokens=0,
                total_tokens=0,
                request_duration_seconds=duration,
                error=f"HTTP {e.code}: {e.reason}",
                status_code=e.code,
            )
        except Exception as e:
            duration = time.perf_counter() - t0
            sampler.stop()
            return RequestResult(
                prompt_tokens=0,
                completion_tokens=0,
                total_tokens=0,
                request_duration_seconds=duration,
                error=str(e),
            )

        duration = time.perf_counter() - t0
        sampler.stop()

        avg_power = sampler.avg_power_watts
        energy_j = avg_power * duration

        try:
            data = json.loads(body)
        except json.JSONDecodeError as e:
            return RequestResult(
                prompt_tokens=0,
                completion_tokens=0,
                total_tokens=0,
                request_duration_seconds=duration,
                error=f"JSON decode: {e}",
                status_code=status,
            )

        usage = data.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        total_tokens = usage.get("total_tokens", prompt_tokens + completion_tokens)

        return RequestResult(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            request_duration_seconds=duration,
            energy_joules=energy_j,
            energy_kwh=energy_j / 3_600_000.0,
            avg_power_watts=avg_power,
            inference_duration_seconds=duration,
            attribution_method="local-power-sampling",
            attribution_ratio=1.0,
            status_code=status,
        )

    def _aggregate_results(
        self,
        profile_name: str,
        model: str,
        endpoint: str,
        results: List[RequestResult],
        wall_clock_seconds: float,
    ) -> ProfileResult:
        successful = [r for r in results if r.error is None]
        failed = [r for r in results if r.error is not None]

        total_tokens = sum(r.total_tokens for r in successful)
        total_prompt_tokens = sum(r.prompt_tokens for r in successful)
        total_completion_tokens = sum(r.completion_tokens for r in successful)
        total_inference_seconds = sum(r.request_duration_seconds for r in successful)

        tokens_per_second = (
            total_completion_tokens / wall_clock_seconds if wall_clock_seconds > 0 else 0
        )

        energy_results = [r for r in successful if r.energy_joules is not None]
        has_energy = len(energy_results) > 0

        total_energy_j: Optional[float] = None
        total_energy_kwh: Optional[float] = None
        wh_per_request: Optional[float] = None
        tokens_per_joule: Optional[float] = None
        avg_power: Optional[float] = None

        if has_energy:
            total_energy_j = sum(r.energy_joules or 0.0 for r in energy_results)
            total_energy_kwh = total_energy_j / 3_600_000.0
            wh_per_request = (total_energy_kwh * 1000) / len(energy_results)
            tokens_per_joule = total_tokens / total_energy_j if total_energy_j > 0 else None
            power_readings = [r.avg_power_watts for r in energy_results if r.avg_power_watts]
            avg_power = sum(power_readings) / len(power_readings) if power_readings else None

        return ProfileResult(
            profile_name=profile_name,
            model=model,
            endpoint=endpoint,
            timestamp=datetime.now(timezone.utc),
            request_count=len(results),
            successful_requests=len(successful),
            failed_requests=len(failed),
            concurrency=1,
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
