#!/usr/bin/env python3
"""CLI entry point for single-stream on-box energy benchmarks.

Runs prompts one at a time against a local vLLM (or other OpenAI-compatible)
endpoint, samples GPU power locally during each request, and emits one
JSONL row per request with energy / token / timing metrics.

This is the on-box equivalent of `ai-energy-profile`, but for energy
measurement instead of load testing. Use this against raw vLLM where
the response doesn't include an `energy` field. Use `EnergyAwareExecutor`
(via library API, not CLI yet) against the NW gateway where it does.

Output format: a JSONL file with a `_meta` header row followed by one
row per request. The schema matches what downstream consumers in the
`energy-aware-inference` repo expect, so the file can be loaded directly
by their `frontier dataset build` pipeline.
"""

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

from ..executors.single_stream import (
    DEFAULT_POWER_SAMPLE_INTERVAL_S,
    SingleStreamExecutor,
)
from ..profiles import LoadProfileConfig
from ..profiles.definitions import get_profile, is_multi_phase

SINGLE_STREAM_PROFILES = ["single_stream_light", "single_stream_moderate", "single_stream_heavy"]


def _positive_float(raw: str) -> float:
    """argparse type validator: accept only strictly positive floats.

    A zero or negative sample interval would either busy-loop nvidia-smi
    (interval=0) or feed a negative duration to threading.Event.wait()
    where behavior is implementation-defined. Reject up front instead
    of producing nonsense measurements.
    """
    value = float(raw)
    if value <= 0:
        raise argparse.ArgumentTypeError(f"must be > 0 (got {raw!r})")
    return value


def detect_gpu_type() -> str:
    """Return normalized GPU model (e.g. 'A100', 'H200') from nvidia-smi."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            raw = result.stdout.strip().split("\n")[0]
            return raw.replace("NVIDIA ", "").replace("-", " ")
    except (subprocess.TimeoutExpired, OSError):
        # OSError covers FileNotFoundError (binary missing) and
        # PermissionError (binary not executable in restricted envs).
        # In both cases we want a graceful "unknown" fallback rather
        # than aborting meta-header construction.
        pass
    return "unknown"


def build_meta_header(
    model: str,
    profile_name: str,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    """Build the `_meta` JSONL header row.

    Shape matches what downstream `energy-aware-inference` expects so the
    file can be consumed by their loader without translation. The
    `benchmark_source` field is set to `"codecarbon_sweep"` for backwards
    compatibility with their tagging, even though we no longer use the
    CodeCarbon library here — `codecarbon_sweep` historically meant
    "on-box single-stream power-sampling," which is exactly what this CLI
    does.
    """
    gpu_count = args.tensor_parallel
    meta: Dict[str, Any] = {
        "_meta": True,
        "model": model,
        "gpu_type": args.gpu_type or detect_gpu_type(),
        "profile_name": profile_name.replace("single_stream_", ""),
        "gpu_count": gpu_count,
        "tensor_parallel": args.tensor_parallel,
        "concurrency": 1,
        "serving_engine": args.serving_engine,
        "benchmark_source": "codecarbon_sweep",
        "measurement_mode": "local-power-sampling",
        "energy_collector": "nvidia-smi",
        "power_sample_interval_s": args.power_sample_interval,
        "serving_host": args.endpoint,
        "benchmark_host": os.environ.get("HOSTNAME", platform.node() or "unknown"),
    }
    if args.quantization:
        meta["quantization"] = args.quantization
    if args.max_model_len:
        meta["max_model_len"] = args.max_model_len
    if args.gpu_memory_utilization is not None:
        meta["gpu_memory_utilization"] = args.gpu_memory_utilization
    return meta


def request_result_to_jsonl_row(req: Any) -> Dict[str, Any]:
    """Convert a RequestResult to the JSONL row shape downstream expects."""
    if req.error is not None:
        row_err: Dict[str, Any] = {
            "error": req.error,
            "duration_seconds": round(req.request_duration_seconds, 4),
            "estimated": False,
        }
        if req.sample_errors is not None:
            row_err["sample_errors"] = req.sample_errors
        return row_err
    output_tokens = req.completion_tokens or 0
    total_tokens = req.total_tokens or 0
    row: Dict[str, Any] = {
        "duration_seconds": round(req.request_duration_seconds, 4),
        "input_tokens": req.prompt_tokens,
        "output_tokens": output_tokens,
        "thinking_tokens": 0,
        "estimated": False,
    }
    # Only emit energy fields when sampling actually produced data — emitting
    # zeros for "no samples captured" silently looks like a successful 0 J run.
    if req.energy_joules is not None:
        energy_j = req.energy_joules
        row["energy_joules"] = round(energy_j, 4)
        row["avg_power_watts"] = round(req.avg_power_watts or 0.0, 1)
        row["tokens_per_joule"] = round(total_tokens / energy_j, 4) if energy_j > 0 else 0
        row["energy_per_useful_token"] = (
            round(energy_j / output_tokens, 4) if output_tokens > 0 else 0
        )
    else:
        row["energy_joules"] = None
        row["avg_power_watts"] = None
        row["tokens_per_joule"] = None
        row["energy_per_useful_token"] = None
        row["energy_attribution"] = "no-samples"
    # Sample-error count lets downstream audit measurement quality even when
    # the energy figure looks plausible.
    if req.sample_errors is not None:
        row["sample_errors"] = req.sample_errors
    return row


def write_jsonl_output(
    output_path: Path,
    meta: Dict[str, Any],
    results: List[Any],
) -> None:
    """Write the meta header + per-request rows to JSONL."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        f.write(json.dumps(meta) + "\n")
        for r in results:
            f.write(json.dumps(request_result_to_jsonl_row(r)) + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Run a single-stream on-box energy benchmark against a vLLM endpoint",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default single_stream_light profile against local vLLM
  ai-energy-single-stream --model Qwen/Qwen3-32B

  # Heavy profile against a remote vLLM, save output to a specific dir
  ai-energy-single-stream --model Qwen/Qwen3-32B \\
    --endpoint http://10.0.0.5:8000/v1 \\
    --profile single_stream_heavy \\
    --output-dir ./results

  # Reproducible run with explicit seed and TP=2
  ai-energy-single-stream --model meta-llama/Llama-3.3-70B \\
    --profile single_stream_moderate \\
    --tensor-parallel 2 --seed 42

Profiles available (single-stream only — concurrency is always 1 here):
  single_stream_light     - 30 requests, 100-200 output tokens
  single_stream_moderate  - 30 requests, 200-500 output tokens
  single_stream_heavy     - 20 requests, 500-1000 output tokens
        """,
    )

    parser.add_argument(
        "--profile",
        choices=SINGLE_STREAM_PROFILES,
        default="single_stream_light",
        help="Single-stream profile to use (default: single_stream_light)",
    )
    parser.add_argument(
        "--endpoint",
        default="http://localhost:8000/v1",
        help="OpenAI-compatible endpoint URL (default: http://localhost:8000/v1)",
    )
    parser.add_argument("--model", required=True, help="Model name (required)")
    parser.add_argument(
        "--api-key",
        default=None,
        help="Optional bearer token (omit for unauthenticated local vLLM)",
    )
    parser.add_argument(
        "--output-dir",
        default="./single_stream_results",
        help="Directory for JSONL output (default: ./single_stream_results)",
    )
    parser.add_argument(
        "--output-file",
        default=None,
        help="Override output filename; default is <model>_<profile>_<timestamp>.jsonl",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=120,
        help="Per-request HTTP timeout in seconds (default: 120)",
    )
    parser.add_argument(
        "--power-sample-interval",
        type=_positive_float,
        default=DEFAULT_POWER_SAMPLE_INTERVAL_S,
        help=f"GPU power sample interval in seconds (default: {DEFAULT_POWER_SAMPLE_INTERVAL_S})",
    )
    parser.add_argument(
        "--tensor-parallel",
        type=int,
        default=1,
        help="Tensor parallelism — only metadata, doesn't change measurement (default: 1)",
    )
    parser.add_argument(
        "--serving-engine",
        default="vllm",
        help="Serving engine label for metadata (default: vllm)",
    )
    parser.add_argument("--gpu-type", default=None, help="Override detected GPU type label")
    parser.add_argument("--quantization", default=None, help="Quantization label for metadata")
    parser.add_argument("--max-model-len", type=int, default=None, help="Metadata only")
    parser.add_argument("--gpu-memory-utilization", type=float, default=None, help="Metadata only")

    args = parser.parse_args()

    profile = get_profile(args.profile)
    if is_multi_phase(profile):
        print(f"Profile '{args.profile}' is multi-phase; single-stream requires single-phase.")
        sys.exit(2)
    assert isinstance(profile, LoadProfileConfig)

    print("Single-stream benchmark")
    print(f"  profile:  {args.profile}")
    print(f"  endpoint: {args.endpoint}")
    print(f"  model:    {args.model}")
    print(f"  requests: {profile.request_count}")
    print(f"  out_tok:  {profile.output_token_range}")
    print(f"  seed:     {args.seed}")
    print("=" * 60)

    executor = SingleStreamExecutor(
        seed=args.seed, power_sample_interval_s=args.power_sample_interval
    )
    result = executor.run(
        profile=profile,
        endpoint=args.endpoint,
        model=args.model,
        api_key=args.api_key,
        timeout_seconds=args.timeout_seconds,
    )

    if args.output_file:
        output_path = Path(args.output_dir) / args.output_file
    else:
        safe_model = args.model.replace("/", "_").replace("\\", "_")
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        short_profile = args.profile.replace("single_stream_", "")
        output_path = Path(args.output_dir) / f"{safe_model}_{short_profile}_{timestamp}.jsonl"

    meta = build_meta_header(args.model, args.profile, args)
    write_jsonl_output(output_path, meta, result.individual_results)

    print(f"\nSaved {result.successful_requests} measurements → {output_path}")
    print(f"  failed: {result.failed_requests}")
    if result.energy_available:
        print(f"  total energy: {result.total_energy_joules:.1f} J")
        print(f"  avg power:    {result.avg_power_watts:.1f} W")
        print(f"  tok/J:        {result.tokens_per_joule:.3f}")

    # Exit non-zero when ANY requests failed, not just when none succeeded.
    # The old `successful_requests > 0` check treated 19/20 failures as a
    # successful run, which made CI/orchestrators silently merge bad data.
    # Callers who tolerate partial failure can compare counts themselves.
    if result.failed_requests > 0 or result.successful_requests == 0:
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
