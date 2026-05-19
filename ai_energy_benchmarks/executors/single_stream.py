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
    1. Start a fresh background `GpuPowerSampler` thread.
    2. POST one chat-completion to the vLLM endpoint synchronously.
    3. Stop the sampler. Compute energy = avg_power * wall_clock_duration.
    4. Append a `RequestResult` populated from the response usage and the
       local power measurement.

Use this for benchmarking raw vLLM (or any OpenAI-compatible endpoint
that does not expose per-request energy in the response). For
deployments where the endpoint exposes `energy` in the response, use
`EnergyAwareExecutor` instead.
"""

import hashlib
import json
import logging
import random
import subprocess
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from typing import List, Optional

from ..profiles import LoadProfileConfig
from .energy_aware import (
    ProfileResult,
    RequestResult,
)

logger = logging.getLogger(__name__)

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

SAMPLE_ERROR_WARN_RATIO = 0.25
"""Warn the operator when more than this fraction of nvidia-smi calls for a
single request failed. At >25% error rate the avg-power figure has lost
enough samples that the per-request energy number is no longer trustworthy,
even though we still report it (since it's better than silently dropping
the request)."""

# Base prompts for generating varied requests. Duplicated from
# EnergyAwareExecutor.BASE_PROMPTS so this executor doesn't pull in the
# aiohttp dependency just to share a list of strings.
_BASE_PROMPTS = [
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

_CONTENT_BLOCKS = [
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


def _generate_prompts(
    count: int,
    input_token_range: tuple,
    seed: Optional[int],
    rng: random.Random,
) -> List[str]:
    """Generate deterministic prompts matching EnergyAwareExecutor's format.

    Standalone helper (no aiohttp dependency) so SingleStreamExecutor can
    run in minimal installs. The output is character-for-character
    identical to EnergyAwareExecutor._generate_prompts for the same seed
    and input_token_range — both executors draw from the same prompt
    pool, so cross-executor comparisons remain apples-to-apples.
    """
    prompts: List[str] = []
    min_tokens, max_tokens = input_token_range
    for i in range(count):
        base_prompt = _BASE_PROMPTS[i % len(_BASE_PROMPTS)]
        if seed is not None:
            unique_id = hashlib.md5(f"{seed}_{i}".encode()).hexdigest()[:8]
        else:
            unique_id = hashlib.md5(f"{time.time()}_{i}".encode()).hexdigest()[:8]
        prompt = f"[Request {i + 1}/{count}, ID:{unique_id}] " + base_prompt

        target_tokens = rng.randint(min_tokens, max_tokens)
        estimated_tokens = len(prompt) // 4  # ~4 chars/token
        block_index = 0
        while estimated_tokens < target_tokens:
            block = _CONTENT_BLOCKS[block_index % len(_CONTENT_BLOCKS)]
            if block_index >= len(_CONTENT_BLOCKS):
                variation = f" (aspect {block_index // len(_CONTENT_BLOCKS) + 1})"
                block = block.rstrip(".") + variation + "."
            prompt += " " + block
            estimated_tokens = len(prompt) // 4
            block_index += 1
            if block_index > 200:
                break
        prompts.append(prompt)
    return prompts


class GpuPowerSampler:
    """Sample total GPU power draw in a background thread.

    Calls `nvidia-smi --query-gpu=power.draw` at a fixed interval and
    averages the per-sample readings. On multi-GPU systems (e.g. vLLM
    `--tensor-parallel N`) the readings from every visible GPU are
    summed at each sample so the per-request `avg_power_watts` reflects
    total board power, not just GPU 0. A partial sample (one GPU's line
    unparseable) is dropped entirely and counted as a sample error.

    Not thread-safe across multiple concurrent users — the intended
    usage is one sampler per request, in a single-stream loop.
    """

    def __init__(self, interval_s: float = DEFAULT_POWER_SAMPLE_INTERVAL_S):
        self.interval_s = interval_s
        self.samples: List[float] = []
        # Track sample-loop failures separately so callers can audit data
        # quality. Silent timeouts/parse errors would otherwise just shrink
        # the average without any signal that some readings were lost.
        self.sample_errors: int = 0
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        self.samples = []
        self.sample_errors = 0
        self._stop.clear()
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Signal the sample loop to exit and wait for the thread to terminate.

        The join timeout must exceed `NVIDIA_SMI_TIMEOUT_S` because the loop
        body can block in `subprocess.run(nvidia-smi)` for up to that long.
        A timeout shorter than NVIDIA_SMI_TIMEOUT_S can leave the thread
        alive after stop() returns, which (together with sampler reuse)
        causes cross-request sample contamination. Add a small interval
        buffer for the post-call `_stop.wait(interval_s)` sleep.
        """
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=NVIDIA_SMI_TIMEOUT_S + self.interval_s + 1.0)

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
                    # Sum power across every visible GPU. nvidia-smi emits one
                    # line per GPU; for `--tensor-parallel N` the model spans
                    # N GPUs and reporting only line 0 would undercount energy
                    # by ~N×. A blank line (rare on multi-GPU boxes) is
                    # skipped; a non-numeric line raises ValueError, which we
                    # catch below and count as a sample error so a malformed
                    # reading doesn't silently truncate the per-GPU sum.
                    lines = [line for line in result.stdout.strip().split("\n") if line.strip()]
                    if not lines:
                        raise ValueError("nvidia-smi returned no GPU lines")
                    self.samples.append(sum(float(line) for line in lines))
                else:
                    self.sample_errors += 1
            except (ValueError, subprocess.TimeoutExpired, OSError):
                # ValueError: parse failure
                # TimeoutExpired: nvidia-smi stalled
                # OSError: covers FileNotFoundError (binary missing on PATH,
                # e.g. mock test envs) AND PermissionError (binary present
                # but not executable, e.g. restrictive container seccomp /
                # bind-mount without +x). Catching only FileNotFoundError
                # would let PermissionError kill the background sampler
                # thread silently, leaving the request with no energy
                # measurement and no `sample_errors` audit trail.
                self.sample_errors += 1
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
        self._random = random.Random(seed) if seed is not None else random.Random()

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
        prompts = _generate_prompts(
            profile.request_count,
            profile.input_token_range,
            self.seed,
            self._random,
        )
        max_tokens = profile.output_token_range[1]

        endpoint = endpoint.rstrip("/")
        if not endpoint.endswith("/v1") and "/v1" not in endpoint:
            endpoint = f"{endpoint}/v1"

        results: List[RequestResult] = []
        wall_start = time.perf_counter()
        for prompt in prompts:
            # Fresh sampler per request prevents cross-request sample
            # contamination if a prior nvidia-smi call is still in flight
            # when stop() returns. Constructing a sampler is cheap (no
            # threads spawned until start()).
            sampler = GpuPowerSampler(interval_s=self.power_sample_interval_s)
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
                sample_errors=sampler.sample_errors,
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
                sample_errors=sampler.sample_errors,
                error=str(e),
            )

        duration = time.perf_counter() - t0
        sampler.stop()

        sample_errors = sampler.sample_errors
        sample_count = len(sampler.samples)
        total_attempts = sample_count + sample_errors
        if total_attempts > 0 and sample_errors / total_attempts > SAMPLE_ERROR_WARN_RATIO:
            logger.warning(
                "nvidia-smi sample-error rate %.0f%% (%d errors / %d attempts) "
                "for this request — energy figure may be unreliable",
                100.0 * sample_errors / total_attempts,
                sample_errors,
                total_attempts,
            )

        # If nvidia-smi was missing, timed out, or returned unparseable
        # output for every sample, we have no power data for this request.
        # Reporting energy_joules=0 in that case would silently produce
        # invalid benchmark output (downstream would treat 0 J as real,
        # not as missing). Leave energy fields None — downstream
        # aggregation already filters by `energy_joules is not None`.
        if not sampler.samples:
            energy_j: Optional[float] = None
            energy_kwh: Optional[float] = None
            avg_power: Optional[float] = None
            attribution: Optional[str] = None
        else:
            avg_power = sampler.avg_power_watts
            energy_j = avg_power * duration
            energy_kwh = energy_j / 3_600_000.0
            attribution = "local-power-sampling"

        try:
            data = json.loads(body)
        except json.JSONDecodeError as e:
            # The HTTP request completed and the sampler ran cleanly — the
            # energy measurement is still valid even though we can't parse
            # token counts. Carry energy + sample_errors through so the
            # downstream auditing stays consistent with the HTTP-error and
            # generic-exception paths above.
            return RequestResult(
                prompt_tokens=0,
                completion_tokens=0,
                total_tokens=0,
                request_duration_seconds=duration,
                energy_joules=energy_j,
                energy_kwh=energy_kwh,
                avg_power_watts=avg_power,
                inference_duration_seconds=duration if energy_j is not None else None,
                attribution_method=attribution,
                attribution_ratio=1.0 if energy_j is not None else None,
                sample_errors=sample_errors,
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
            energy_kwh=energy_kwh,
            avg_power_watts=avg_power,
            inference_duration_seconds=duration if energy_j is not None else None,
            attribution_method=attribution,
            attribution_ratio=1.0 if energy_j is not None else None,
            sample_errors=sample_errors,
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
            # tokens_per_joule must divide tokens from the SAME requests whose
            # energy we summed. Using `total_tokens` (all successful) here
            # would inflate the metric when some requests lost power data —
            # numerator and denominator must come from the same set.
            energy_token_total = sum(r.total_tokens for r in energy_results)
            tokens_per_joule = energy_token_total / total_energy_j if total_energy_j > 0 else None
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
