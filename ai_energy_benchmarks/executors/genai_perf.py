"""GenAI-Perf executor for running load generation benchmarks.

This module wraps the NVIDIA genai-perf CLI tool for load generation.
"""

import hashlib
import json
import os
import random
import signal
import subprocess
import threading
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union

from ..profiles import LoadProfileConfig, MultiPhaseProfile


class GenAIPerfExecutor:
    """Executor that wraps the genai-perf CLI tool for load generation."""

    def __init__(self, seed: Optional[int] = None):
        """Initialize the GenAI-Perf executor.

        Args:
            seed: Random seed for reproducible inputs
        """
        self.seed = seed
        self.running = True
        self.current_process: Optional[subprocess.Popen] = None
        self._process_lock = threading.Lock()

        # Set seed for reproducible random number generation
        if self.seed is not None:
            random.seed(self.seed)

        # Set up signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        print(f"\nReceived signal {signum}. Shutting down gracefully...")
        self.running = False
        with self._process_lock:
            if self.current_process:
                print("Terminating current genai-perf process...")
                self.current_process.terminate()
                try:
                    self.current_process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    print("Force killing genai-perf process...")
                    self.current_process.kill()

    def build_command(
        self,
        profile: Union[LoadProfileConfig, Dict[str, Any]],
        endpoint: str,
        model: str,
        output_dir: str,
        prompts_file: Optional[str] = None,
        api_key: Optional[str] = None,
        endpoint_type: str = "chat",
    ) -> List[str]:
        """Build genai-perf command with parameters.

        Args:
            profile: Load profile configuration
            endpoint: API endpoint URL
            model: Model name
            output_dir: Directory for output files
            prompts_file: Optional path to pre-generated prompts file
            api_key: Optional API key for authenticated endpoints
            endpoint_type: API endpoint type ('chat' or 'completions')

        Returns:
            Command as list of strings
        """
        # Convert LoadProfileConfig to dict if needed
        params = profile.to_dict() if isinstance(profile, LoadProfileConfig) else profile

        # Get token ranges
        input_tokens = params.get("input_token_range", [100, 500])
        output_tokens = params.get("output_token_range", [100, 500])

        # Calculate averages for genai-perf parameters
        avg_input = sum(input_tokens) // 2
        avg_output = sum(output_tokens) // 2

        # Get concurrency and request count
        concurrency = params.get("concurrency", 1)
        request_count = max(params.get("request_count", 100), concurrency)

        # Base command - genai-perf expects base URL without /v1 suffix
        base_url = endpoint.rstrip("/v1").rstrip("/")
        cmd = [
            "genai-perf",
            "profile",
            "--model",
            model,
            "--endpoint-type",
            endpoint_type,
            "--url",
            base_url,
            "--concurrency",
            str(concurrency),
            "--request-count",
            str(request_count),
            "--synthetic-input-tokens-mean",
            str(avg_input),
            "--output-tokens-mean",
            str(avg_output),
            "--artifact-dir",
            output_dir,
        ]

        # Add API key header if provided
        if api_key:
            cmd.extend(["--header", f"Authorization: Bearer {api_key}"])

        # Use pre-generated prompts file if provided, otherwise use synthetic generation
        if prompts_file and os.path.exists(prompts_file):
            cmd.extend(["--input-file", prompts_file])
            print(f"  Using pre-generated prompts: {prompts_file}")
        else:
            # Add token variance for more realistic workloads
            if self.seed is not None:
                hash_obj = hashlib.md5(str(self.seed).encode())
                hash_int = int(hash_obj.hexdigest()[:8], 16)
                input_stddev = 1 + (hash_int % max(1, (input_tokens[1] - input_tokens[0]) // 4))
                output_stddev = 1 + (
                    (hash_int >> 8) % max(1, (output_tokens[1] - output_tokens[0]) // 4)
                )
            else:
                input_stddev = max(1, (input_tokens[1] - input_tokens[0]) // 4)
                output_stddev = max(1, (output_tokens[1] - output_tokens[0]) // 4)

            cmd.extend(["--synthetic-input-tokens-stddev", str(input_stddev)])
            cmd.extend(["--output-tokens-stddev", str(output_stddev)])

            # Always pass a random seed to genai-perf for varied prompts
            random_seed = self.seed if self.seed is not None else random.randint(0, 2**31)
            cmd.extend(["--random-seed", str(random_seed)])

        return cmd

    def generate_deterministic_prompts(
        self,
        profile: Union[LoadProfileConfig, Dict[str, Any]],
        output_dir: str,
        cache_strategy: str = "minimal",
    ) -> Optional[str]:
        """Generate deterministic prompts file for reproducible inputs.

        Args:
            profile: Load profile configuration
            output_dir: Directory for output files
            cache_strategy: Cache strategy for prompt generation

        Returns:
            Path to generated prompts file, or None if generation failed
        """
        try:
            # Convert LoadProfileConfig to dict if needed
            params = profile.to_dict() if isinstance(profile, LoadProfileConfig) else profile

            # Create prompts directory
            prompts_dir = os.path.join(output_dir, "prompts")
            os.makedirs(prompts_dir, exist_ok=True)

            # Generate filename based on seed and parameters for caching
            params_str = json.dumps(params, sort_keys=True)
            cache_key = hashlib.md5(
                f"{self.seed}_{params_str}_{cache_strategy}".encode()
            ).hexdigest()[:8]
            prompts_file = os.path.join(prompts_dir, f"prompts_{cache_key}.jsonl")

            # Return existing file if it exists (caching)
            if os.path.exists(prompts_file):
                print(f"  Using cached prompts file: {prompts_file}")
                return prompts_file

            # Get parameters
            input_tokens = params.get("input_token_range", [100, 500])
            request_count = params.get("request_count", 100)

            # Generate deterministic prompts
            prompts = []
            base_prompts = [
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
            ]

            # Use seed combined with cache_key for deterministic but unique prompts per phase
            phase_seed = int(hashlib.md5(f"{self.seed}_{cache_key}".encode()).hexdigest()[:8], 16)
            prompt_random = random.Random(phase_seed)

            for i in range(int(request_count)):
                base_prompt = base_prompts[i % len(base_prompts)]

                # Add unique prefix to prevent prefix caching in vLLM
                unique_prefix = (
                    f"[Request {i + 1} of {request_count}, ID:{phase_seed:08x}-{i:04d}] "
                )
                base_prompt = unique_prefix + base_prompt

                # Add deterministic variation to reach target token count
                target_tokens = prompt_random.randint(input_tokens[0], input_tokens[1])
                estimated_tokens = len(base_prompt) // 4

                if estimated_tokens < target_tokens:
                    additional_content = [
                        "Please provide detailed examples and explanations.",
                        "Include practical applications and use cases.",
                        "Discuss the historical context and development.",
                        "Explain the technical details and mechanisms involved.",
                        "Compare and contrast with related concepts.",
                        "Analyze the advantages and disadvantages.",
                        "Provide step-by-step instructions or processes.",
                        "Include relevant statistics and data.",
                        "Discuss future trends and developments.",
                        "Explain the impact on society and industry.",
                    ]

                    additions: List[str] = []
                    tokens_needed = target_tokens - estimated_tokens
                    while len(" ".join(additions)) // 4 < tokens_needed:
                        addition = additional_content[len(additions) % len(additional_content)]
                        additions.append(addition)

                    final_prompt = f"{base_prompt} {' '.join(additions)}"
                else:
                    final_prompt = base_prompt[: target_tokens * 4]

                prompt_entry = {"text_input": final_prompt, "stream": False}
                prompts.append(prompt_entry)

            # Write prompts to file
            with open(prompts_file, "w") as f:
                for prompt in prompts:
                    f.write(json.dumps(prompt) + "\n")

            print(f"  Generated {len(prompts)} deterministic prompts: {prompts_file}")
            return prompts_file

        except Exception as e:
            print(f"  Error generating prompts file: {e}")
            return None

    def run(
        self,
        profile: Union[LoadProfileConfig, Dict[str, Any]],
        endpoint: str,
        model: str,
        output_dir: str,
        run_id_suffix: Optional[str] = None,
        prompts_file: Optional[str] = None,
        api_key: Optional[str] = None,
        endpoint_type: str = "chat",
    ) -> bool:
        """Execute genai-perf with the given profile.

        Args:
            profile: Load profile configuration
            endpoint: API endpoint URL
            model: Model name
            output_dir: Directory for output files
            run_id_suffix: Suffix for RunId differentiation
            prompts_file: Custom prompts file (overrides seed-based generation)
            api_key: Optional API key for authenticated endpoints
            endpoint_type: API endpoint type ('chat' or 'completions')

        Returns:
            True if successful, False otherwise
        """
        if not self.running:
            return False

        # Convert LoadProfileConfig to dict if needed
        params = profile.to_dict() if isinstance(profile, LoadProfileConfig) else profile
        cache_strategy = params.get("cache_strategy", "minimal")

        # Use custom prompts file if provided, otherwise generate from seed
        actual_prompts_file = None
        if prompts_file and os.path.exists(prompts_file):
            actual_prompts_file = prompts_file
            print(f"  Using custom prompts file: {prompts_file}")
        elif self.seed is not None:
            actual_prompts_file = self.generate_deterministic_prompts(
                params, output_dir, cache_strategy
            )

        # Build command
        cmd = self.build_command(
            params, endpoint, model, output_dir, actual_prompts_file, api_key, endpoint_type
        )

        print("Running genai-perf command:")
        print(f"  {' '.join(cmd)}")

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        try:
            # Run genai-perf
            with self._process_lock:
                self.current_process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    env=os.environ.copy(),
                    universal_newlines=True,
                    bufsize=1,
                )

            # Stream output in real-time
            stdout = self.current_process.stdout
            while stdout is not None:
                output = stdout.readline()
                if output == "" and self.current_process.poll() is not None:
                    break
                if output:
                    print(output.strip())

            # Wait for completion
            self.current_process.wait()

            if self.current_process.returncode == 0:
                print("genai-perf completed successfully")

                # Fix output files for database compatibility
                from ..formatters.episode import EpisodeFormatter

                formatter = EpisodeFormatter()
                formatter.fix_output(output_dir, run_id_suffix)

                return True
            else:
                print(f"genai-perf failed with return code {self.current_process.returncode}")
                return False

        except Exception as e:
            print(f"Error running genai-perf: {e}")
            return False
        finally:
            with self._process_lock:
                self.current_process = None

    def run_multi_phase(
        self,
        profile: MultiPhaseProfile,
        endpoint: str,
        model: str,
        output_dir: str,
        run_id_suffix: Optional[str] = None,
        api_key: Optional[str] = None,
        endpoint_type: str = "chat",
    ) -> bool:
        """Execute multi-phase profile with cycling through phases.

        Args:
            profile: Multi-phase profile configuration
            endpoint: API endpoint URL
            model: Model name
            output_dir: Directory for output files
            run_id_suffix: Suffix for RunId differentiation
            api_key: Optional API key for authenticated endpoints
            endpoint_type: API endpoint type ('chat' or 'completions')

        Returns:
            True if successful, False otherwise
        """
        phases = profile.phases
        if not phases:
            print("No phases defined for multi-phase profile")
            return False

        total_requests = sum(p.request_count for p in phases)
        print(f"Starting multi-phase profile with {len(phases)} phases:")
        for i, phase in enumerate(phases):
            print(
                f"  Phase {i + 1}: {phase.name} - {phase.request_count} requests (concurrency {phase.concurrency})"
            )
        print(f"Total requests: {total_requests}")
        print("=" * 60)

        # Track actual episode timing
        episode_actual_start_time = None
        episode_actual_end_time = None

        # Execute each phase sequentially
        phase_results = []
        for i, phase in enumerate(phases):
            if not self.running:
                break

            print(f"\nStarting Phase {i + 1}: {phase.name} ({phase.request_count} requests)")

            # Capture episode start time when first phase begins
            if episode_actual_start_time is None:
                episode_actual_start_time = datetime.now(timezone.utc)
                print(f"  Episode started at: {episode_actual_start_time} UTC")

            # Skip phases with no requests
            if phase.request_count == 0:
                print("  Skipping phase with 0 requests")
                continue

            # Create phase-specific output directory
            phase_output_dir = os.path.join(output_dir, f"phase_{i + 1}_{phase.name}")

            # Run this phase
            success = self.run(
                phase,
                endpoint,
                model,
                phase_output_dir,
                api_key=api_key,
                endpoint_type=endpoint_type,
            )

            # Update episode end time after each phase completes
            episode_actual_end_time = datetime.now(timezone.utc)

            if success:
                print(f"Phase {phase.name} completed successfully")
                phase_results.append(
                    {"phase": phase.name, "request_count": phase.request_count, "success": True}
                )
            else:
                print(f"Phase {phase.name} failed")
                phase_results.append(
                    {"phase": phase.name, "request_count": phase.request_count, "success": False}
                )

            # Brief pause between phases
            time.sleep(1)

        # Summary
        successful_phases = sum(1 for r in phase_results if r["success"])
        print("\n" + "=" * 60)
        print(
            f"Multi-phase profile completed: {successful_phases}/{len(phase_results)} phases successful"
        )

        for result in phase_results:
            status = "PASS" if result["success"] else "FAIL"
            print(f"  {status} {result['phase']}: {result['request_count']} requests")

        # Aggregate phase results into single episode summary
        if successful_phases > 0:
            # Extract episode ID from output directory path
            path_parts = output_dir.split(os.sep)
            episode_id = None
            for part in path_parts:
                if part.startswith("episode_"):
                    episode_id = part.replace("episode_", "")
                    break

            if episode_id is None:
                episode_id = os.path.basename(output_dir).replace("multiphase_", "")
                if len(episode_id) > 10:
                    episode_id = "0"

            # Calculate actual episode elapsed time
            if episode_actual_start_time and episode_actual_end_time:
                actual_elapsed_seconds = (
                    episode_actual_end_time - episode_actual_start_time
                ).total_seconds()
                print(f"  Actual episode duration: {actual_elapsed_seconds:.1f} seconds")

            from ..formatters.episode import EpisodeFormatter

            formatter = EpisodeFormatter()
            aggregation_success = formatter.aggregate_phases(
                output_dir,
                episode_id,
                episode_actual_start_time,
                episode_actual_end_time,
                run_id_suffix,
            )
            if aggregation_success:
                print("Episode summary ready for database upload")
            else:
                print("Episode aggregation failed - individual phase records will be uploaded")

        return successful_phases > 0
