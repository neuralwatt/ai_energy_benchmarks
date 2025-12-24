#!/usr/bin/env python3
"""CLI entry point for genai-perf profile execution.

This module provides a command-line interface for running load profiles using
the genai-perf tool.
"""

import argparse
import os
import sys
from datetime import datetime, timezone

from ..executors.genai_perf import GenAIPerfExecutor
from ..profiles import LoadProfileConfig
from ..profiles.definitions import get_profile, is_multi_phase, list_profiles


def main():
    """Main CLI entry point for profile execution."""
    parser = argparse.ArgumentParser(
        description="Generate load for vLLM using genai-perf profiles",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run a light load test (20 requests)
  ai-energy-profile --profile light --model my-model

  # Run a stress test with a custom endpoint
  ai-energy-profile --profile stress --model my-model --endpoint http://my-server:8000/v1

  # Run against an authenticated API endpoint
  ai-energy-profile --profile light --model meta-llama/Llama-3.3-70B-Instruct \\
    --endpoint https://api.neuralwatt.com/v1 --api-key nw_live_xxxxx

  # Run the multiphase workload
  ai-energy-profile --profile multiphase --model my-model

  # Run with reproducible inputs using a seed
  ai-energy-profile --profile moderate --model my-model --seed 42

Available profiles:
  light      - Light load (20 requests, concurrency 2)
  moderate   - Moderate load (40 requests, concurrency 4)
  heavy      - Heavy load (80 requests, concurrency 8)
  stress     - Stress test (240 requests, concurrency 24)
  multiphase - Multi-phase workload with variability
  pattern    - Multi-phase pattern test
  power_test - Extended phases for power measurement
        """,
    )

    parser.add_argument(
        "--profile",
        choices=list_profiles(),
        default="moderate",
        help="Load profile to use (default: moderate)",
    )
    parser.add_argument(
        "--endpoint",
        default="http://localhost:8000/v1",
        help="vLLM endpoint URL (default: http://localhost:8000/v1)",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Model name (required)",
    )
    parser.add_argument(
        "--output-dir",
        default="./benchmark_output",
        help="Output directory for results (default: ./benchmark_output)",
    )
    parser.add_argument(
        "--config",
        help="Path to custom configuration file (YAML or JSON)",
    )
    parser.add_argument(
        "--run-id-suffix",
        help='Suffix to append to episode RunId for differentiation (e.g., "monitor", "agent")',
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Random seed for reproducible inputs and outputs",
    )
    parser.add_argument(
        "--prompts-file",
        help="Path to custom prompts file (overrides seed-based generation)",
    )
    parser.add_argument(
        "--api-key",
        help="API key for authenticated endpoints (passed as Bearer token)",
    )
    parser.add_argument(
        "--endpoint-type",
        choices=["chat", "completions"],
        default="chat",
        help="Endpoint type: 'chat' for /v1/chat/completions, 'completions' for /v1/completions (default: chat)",
    )

    args = parser.parse_args()

    # Create executor with seed
    executor = GenAIPerfExecutor(seed=args.seed)

    # Get profile config
    profile = get_profile(args.profile)

    # Display configuration
    print("Starting genai-perf load generation")
    print(f"Profile: {args.profile}")
    print(f"Endpoint: {args.endpoint}")
    print(f"Model: {args.model}")
    if is_multi_phase(profile):
        total_requests = sum(p.request_count for p in profile.phases)
        print(f"Total requests: {total_requests} (across {len(profile.phases)} phases)")
    else:
        assert isinstance(profile, LoadProfileConfig)
        print(f"Request count: {profile.request_count}")
        print(f"Concurrency: {profile.concurrency}")
    if args.seed is not None:
        print(f"Random Seed: {args.seed} (reproducible inputs enabled)")
    if args.api_key:
        print("API Key: *** (authentication enabled)")
    print("=" * 60)

    # Create timestamped output directory
    # Sanitize model name for directory (replace / with _)
    safe_model = args.model.replace("/", "_").replace("\\", "_")
    if args.seed is not None:
        timestamp = f"seed{args.seed}_{safe_model}"
    else:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f"{args.profile}_{timestamp}")

    # Execute based on profile type
    if is_multi_phase(profile):
        # Run multi-phase profile
        success = executor.run_multi_phase(
            profile,
            args.endpoint,
            args.model,
            output_dir,
            run_id_suffix=args.run_id_suffix,
            api_key=args.api_key,
            endpoint_type=args.endpoint_type,
        )

        if success:
            print(f"Multi-phase profile completed. Results in: {output_dir}")
        else:
            print("Multi-phase profile failed")
            sys.exit(1)
    else:
        # Run single-phase test
        assert isinstance(profile, LoadProfileConfig)
        success = executor.run(
            profile,
            args.endpoint,
            args.model,
            output_dir,
            run_id_suffix=args.run_id_suffix,
            prompts_file=args.prompts_file,
            api_key=args.api_key,
            endpoint_type=args.endpoint_type,
        )

        if success:
            print(f"Load test completed. Results in: {output_dir}")
        else:
            print("Load test failed")
            sys.exit(1)


if __name__ == "__main__":
    main()
