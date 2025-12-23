"""Episode summary formatter for genai-perf results.

This module formats genai-perf output into episode summaries suitable for
database ingestion (CSV format compatible with any SQL database).
Ported from neuralwatt_cloud/genai_perf_load_proper.py.
"""

import glob
import json
import os
import shutil
from datetime import datetime, timedelta, timezone
from typing import List, Optional, Tuple

import pandas as pd


class EpisodeFormatter:
    """Format genai-perf output into episode summaries for database ingestion."""

    def fix_output(self, output_dir: str, run_id_suffix: Optional[str] = None) -> bool:
        """Convert genai-perf output to episode summary.

        Args:
            output_dir: Directory containing genai-perf output
            run_id_suffix: Optional suffix to append to RunId

        Returns:
            True if conversion was successful, False otherwise
        """
        print("  Converting genai-perf output to episode summary...")

        # Extract episode ID from output directory path
        path_parts = output_dir.split(os.sep)
        episode_id = None
        for part in path_parts:
            if part.startswith("episode_"):
                episode_id = part.replace("episode_", "")
                break

        if episode_id is None:
            basename = os.path.basename(output_dir)
            episode_id = (
                basename.replace("light_", "")
                .replace("moderate_", "")
                .replace("heavy_", "")
                .replace("stress_", "")
                .replace("train_", "")
            )
            if len(episode_id) > 10:
                episode_id = "0"

        # For single-phase profiles, create episode summary using the same aggregation logic
        success = self.aggregate_phases(
            output_dir,
            episode_id,
            actual_start_time=None,
            actual_end_time=None,
            run_id_suffix=run_id_suffix,
        )

        if success:
            print("  Successfully created episode summary")
            return True
        else:
            print("  Failed to create episode summary, falling back to original approach...")
            return self._fallback_rename(output_dir)

    def _fallback_rename(self, output_dir: str) -> bool:
        """Fallback approach: rename CSV files for database compatibility.

        Args:
            output_dir: Directory containing genai-perf output

        Returns:
            True if any files were renamed, False otherwise
        """
        csv_files = glob.glob(f"{output_dir}/**/profile_export_genai_perf.csv", recursive=True)

        if not csv_files:
            print(f"  No genai-perf CSV files found in {output_dir}")
            return False

        renamed_count = 0
        for csv_file in csv_files:
            try:
                timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
                parent_dir = os.path.dirname(csv_file)
                new_filename = f"inference.load.genai_perf_phase_{timestamp}.csv"
                new_path = os.path.join(parent_dir, new_filename)

                shutil.move(csv_file, new_path)
                print(f"  Renamed: {os.path.basename(csv_file)} -> {new_filename}")
                renamed_count += 1

            except (OSError, shutil.Error) as e:
                # Log expected file-related errors but continue processing other files.
                print(f"  Error renaming {csv_file}: {e}")
            except Exception:
                # Re-raise unexpected exceptions so they are not silently ignored.
                raise

        print(f"  Renamed {renamed_count} files for database compatibility")
        return renamed_count > 0

    def aggregate_phases(
        self,
        output_dir: str,
        episode_id: str,
        actual_start_time: Optional[datetime] = None,
        actual_end_time: Optional[datetime] = None,
        run_id_suffix: Optional[str] = None,
    ) -> bool:
        """Aggregate multi-phase results into single episode summary for database upload.

        Args:
            output_dir: Directory containing phase subdirectories
            episode_id: Unique episode identifier
            actual_start_time: Actual episode start time (when first phase began)
            actual_end_time: Actual episode end time (when last phase completed)
            run_id_suffix: Optional suffix to append to RunId

        Returns:
            True if aggregation successful, False otherwise
        """
        print("Aggregating phase results into episode summary...")

        # Find all phase JSON files with individual request data
        phase_json_files = glob.glob(
            f"{output_dir}/**/profile_export_genai_perf.json", recursive=True
        )

        if not phase_json_files:
            print("  No phase JSON files found to aggregate")
            return False

        print(f"  Found {len(phase_json_files)} phase JSON files to aggregate")

        # Aggregate actual performance data across all phases
        episode_total_tokens = 0.0
        episode_total_duration_seconds = 0.0
        episode_total_requests = 0
        phase_count = 0

        # Read and sum actual metrics from each phase JSON file
        for json_file in phase_json_files:
            try:
                print(f"    Processing {os.path.basename(os.path.dirname(json_file))}...")

                with open(json_file, "r") as f:
                    phase_data = json.load(f)

                # Extract actual request count
                if "request_count" in phase_data and "avg" in phase_data["request_count"]:
                    phase_request_count = int(phase_data["request_count"]["avg"])
                else:
                    phase_request_count = 0

                # Extract actual token counts
                phase_total_output_tokens = 0
                phase_total_input_tokens = 0

                if (
                    "output_sequence_length" in phase_data
                    and "avg" in phase_data["output_sequence_length"]
                ):
                    phase_avg_output_tokens = phase_data["output_sequence_length"]["avg"]
                    phase_total_output_tokens = phase_avg_output_tokens * phase_request_count

                if (
                    "input_sequence_length" in phase_data
                    and "avg" in phase_data["input_sequence_length"]
                ):
                    phase_avg_input_tokens = phase_data["input_sequence_length"]["avg"]
                    phase_total_input_tokens = phase_avg_input_tokens * phase_request_count

                phase_total_tokens = phase_total_output_tokens + phase_total_input_tokens

                # Extract actual timing data
                if "request_latency" in phase_data and "avg" in phase_data["request_latency"]:
                    phase_avg_latency_ms = phase_data["request_latency"]["avg"]
                    phase_total_duration = (phase_avg_latency_ms / 1000.0) * phase_request_count
                else:
                    phase_total_duration = 0

                # Skip phases with no actual work
                if phase_request_count == 0 or phase_total_tokens == 0:
                    print("      Skipping phase with no requests or tokens")
                    continue

                # Sum across all phases
                episode_total_tokens += phase_total_tokens
                episode_total_duration_seconds += phase_total_duration
                episode_total_requests += phase_request_count

                print(
                    f"      Phase totals: {phase_request_count} requests, {phase_total_tokens:.0f} tokens, {phase_total_duration:.1f}s processing"
                )
                phase_count += 1

            except Exception as e:
                print(f"      Error reading {json_file}: {e}")
                continue

        if phase_count == 0:
            print("  No valid phase data found")
            return False

        # Create episode summary record with actual aggregated totals
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

        # Extract actual start/end times from benchmark data
        episode_start_time, episode_end_time = self._extract_actual_timing(phase_json_files)

        if episode_start_time and episode_end_time:
            actual_elapsed_seconds = (episode_end_time - episode_start_time).total_seconds()
            print(f"  Using actual episode timing: {actual_elapsed_seconds:.1f}s elapsed")
        else:
            # Fallback to current time estimation
            episode_end_time = datetime.now(timezone.utc)
            estimated_duration = max(31, episode_total_duration_seconds)
            episode_start_time = episode_end_time - timedelta(seconds=estimated_duration)
            print(f"  Using estimated episode timing: {estimated_duration:.1f}s elapsed")

        # Extract actual token data from benchmark results
        episode_prompt_tokens, episode_response_tokens = self._extract_actual_tokens(
            phase_json_files
        )

        if episode_prompt_tokens > 0 and episode_response_tokens > 0:
            print(
                f"  Using actual token data: {episode_prompt_tokens} prompt + {episode_response_tokens} response = {episode_prompt_tokens + episode_response_tokens} total"
            )
        else:
            # Fallback to estimates based on total tokens
            episode_avg_input_tokens = 150
            episode_prompt_tokens = episode_total_requests * episode_avg_input_tokens
            episode_response_tokens = int(max(0, episode_total_tokens - episode_prompt_tokens))
            print(
                f"  Using estimated token data: {episode_prompt_tokens} prompt + {episode_response_tokens} response"
            )

        episode_prompt_eval_duration = episode_total_duration_seconds * 0.1
        episode_response_eval_duration = episode_total_duration_seconds * 0.9

        # Calculate throughput metrics
        episode_tokens_per_sec = (
            episode_total_tokens / episode_total_duration_seconds
            if episode_total_duration_seconds > 0
            else 0
        )
        episode_prompt_tokens_per_sec = (
            episode_prompt_tokens / episode_prompt_eval_duration
            if episode_prompt_eval_duration > 0
            else 0
        )
        episode_response_tokens_per_sec = (
            episode_response_tokens / episode_response_eval_duration
            if episode_response_eval_duration > 0
            else 0
        )

        # Build RunId with optional suffix
        run_id = f"episode_{episode_id}"
        if run_id_suffix:
            run_id += f"_{run_id_suffix}"

        episode_summary = {
            "RunId": run_id,
            "StartTime": episode_start_time,
            "EndTime": episode_end_time,
            "TotalTokens": int(episode_total_tokens),
            "TotalDurationSeconds": episode_total_duration_seconds,
            "TokensPerSecond": episode_tokens_per_sec,
            "PromptTokens": int(episode_prompt_tokens),
            "PromptEvalDurationSeconds": episode_prompt_eval_duration,
            "PromptTokensPerSecond": episode_prompt_tokens_per_sec,
            "ResponseTokens": int(episode_response_tokens),
            "ResponseEvalDurationSeconds": episode_response_eval_duration,
            "ResponseTokensPerSecond": episode_response_tokens_per_sec,
            "LongOrShortPrompt": "medium",
            "Model": f"multi-phase-episode-{phase_count}phases",
        }

        # Create episode summary DataFrame and save as CSV
        episode_df = pd.DataFrame([episode_summary])
        episode_csv_path = os.path.join(
            output_dir, f"inference.load.genai_perf_episode_{timestamp}.csv"
        )

        # Save as CSV (compatible with any SQL database)
        episode_df.to_csv(episode_csv_path, index=False)

        print(f"  Created episode summary: {os.path.basename(episode_csv_path)}")
        print(
            f"    Episode totals: {episode_total_requests} requests, {episode_total_tokens:.0f} tokens"
        )
        print(f"    Processing time: {episode_total_duration_seconds:.1f}s")
        print(
            f"    Episode throughput: {episode_tokens_per_sec:.1f} tokens/sec across {phase_count} phases"
        )

        # Clean up individual phase CSV files
        self._cleanup_phase_files(output_dir)

        print("  Only episode summary will be uploaded to database")
        return True

    def _extract_actual_timing(
        self, json_files: List[str]
    ) -> Tuple[Optional[datetime], Optional[datetime]]:
        """Extract actual start and end times from profile_export.json files.

        Args:
            json_files: List of genai-perf JSON output files

        Returns:
            Tuple of (earliest_timestamp, latest_timestamp)
        """
        earliest_timestamp = None
        latest_timestamp = None

        for json_file in json_files:
            profile_export_path = json_file.replace(
                "profile_export_genai_perf.json", "profile_export.json"
            )
            if not os.path.exists(profile_export_path):
                continue

            try:
                with open(profile_export_path, "r") as f:
                    profile_data = json.load(f)

                if "experiments" in profile_data and len(profile_data["experiments"]) > 0:
                    requests = profile_data["experiments"][0].get("requests", [])

                    for request in requests:
                        if "timestamp" in request:
                            timestamp_ns = request["timestamp"]
                            timestamp_s = timestamp_ns / 1_000_000_000.0
                            request_time = datetime.fromtimestamp(timestamp_s, tz=timezone.utc)

                            if earliest_timestamp is None or request_time < earliest_timestamp:
                                earliest_timestamp = request_time
                            if latest_timestamp is None or request_time > latest_timestamp:
                                latest_timestamp = request_time

            except Exception as e:
                print(f"      Could not extract timing from {profile_export_path}: {e}")
                continue

        return earliest_timestamp, latest_timestamp

    def _extract_actual_tokens(self, json_files: List[str]) -> Tuple[int, int]:
        """Extract actual token counts from profile_export.json files.

        Args:
            json_files: List of genai-perf JSON output files

        Returns:
            Tuple of (total_prompt_tokens, total_response_tokens)
        """
        total_prompt_tokens = 0
        total_response_tokens = 0

        for json_file in json_files:
            profile_export_path = json_file.replace(
                "profile_export_genai_perf.json", "profile_export.json"
            )
            if not os.path.exists(profile_export_path):
                continue

            try:
                with open(profile_export_path, "r") as f:
                    profile_data = json.load(f)

                if "experiments" in profile_data and len(profile_data["experiments"]) > 0:
                    requests = profile_data["experiments"][0].get("requests", [])

                    for request in requests:
                        if "response_outputs" in request and len(request["response_outputs"]) > 0:
                            response_text = request["response_outputs"][0].get("response", "{}")
                            try:
                                response_json = json.loads(response_text)
                                if "usage" in response_json:
                                    usage = response_json["usage"]
                                    total_prompt_tokens += usage.get("prompt_tokens", 0)
                                    total_response_tokens += usage.get("completion_tokens", 0)
                            except json.JSONDecodeError:
                                continue

            except Exception as e:
                print(f"      Could not extract tokens from {profile_export_path}: {e}")
                continue

        return total_prompt_tokens, total_response_tokens

    def _cleanup_phase_files(self, output_dir: str) -> None:
        """Clean up individual phase CSV files after aggregation.

        Args:
            output_dir: Directory containing phase files
        """
        phase_csv_files = glob.glob(f"{output_dir}/**/*.csv", recursive=True)
        phase_csv_files = [f for f in phase_csv_files if "genai_perf_phase_" in os.path.basename(f)]

        if phase_csv_files:
            print(
                f"  Removing {len(phase_csv_files)} individual phase CSV files (keeping episode summary and JSON files)..."
            )
            cleaned_count = 0
            for csv_file in phase_csv_files:
                try:
                    os.remove(csv_file)
                    cleaned_count += 1
                except Exception as e:
                    print(f"    Could not remove {csv_file}: {e}")

            print(f"  Removed {cleaned_count} individual phase CSV files")
        else:
            print("  No individual phase CSV files found to clean up")
