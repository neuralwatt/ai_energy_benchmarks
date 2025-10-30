"""Main benchmark runner for POC."""

import time
from typing import Any, Dict, List, Optional

from ai_energy_benchmarks.backends.base import Backend
from ai_energy_benchmarks.backends.pytorch import PyTorchBackend
from ai_energy_benchmarks.backends.vllm import VLLMBackend
from ai_energy_benchmarks.config.parser import BenchmarkConfig, ConfigParser
from ai_energy_benchmarks.datasets.base import Dataset
from ai_energy_benchmarks.datasets.huggingface import HuggingFaceDataset
from ai_energy_benchmarks.metrics.codecarbon import CodeCarbonCollector
from ai_energy_benchmarks.reporters.csv_reporter import CSVReporter
from ai_energy_benchmarks.utils import GPUMonitor


class BenchmarkRunner:
    """Main benchmark runner for POC phase."""

    def __init__(self, config: BenchmarkConfig):
        """Initialize benchmark runner.

        Args:
            config: Benchmark configuration
        """
        self.config = config
        self.backend: Optional[Backend] = None
        self.dataset: Optional[Dataset] = None
        self.metrics_collector: Optional[CodeCarbonCollector] = None
        self.reporter: Optional[CSVReporter] = None

        # Initialize components
        self._initialize_backend()
        self._initialize_dataset()
        self._initialize_metrics()
        self._initialize_reporter()

    def _initialize_backend(self):
        """Initialize inference backend."""
        backend_type = self.config.backend.type

        if backend_type == "vllm":
            endpoint = self.config.backend.endpoint
            if endpoint is None:
                raise ValueError("vLLM backend requires endpoint configuration")
            self.backend = VLLMBackend(
                endpoint=endpoint,
                model=self.config.backend.model,
            )
        elif backend_type == "pytorch":
            self.backend = PyTorchBackend(
                model=self.config.backend.model,
                device=self.config.backend.device,
                device_ids=self.config.backend.device_ids,
            )
        else:
            raise ValueError(f"Unknown backend type: {backend_type}")

        print(f"Initialized {backend_type} backend")

    def _initialize_dataset(self):
        """Initialize dataset loader."""
        self.dataset = HuggingFaceDataset()
        print("Initialized HuggingFace dataset loader")

    def _initialize_metrics(self):
        """Initialize metrics collector."""
        if not self.config.metrics.enabled:
            print("Metrics collection disabled")
            return

        if self.config.metrics.type == "codecarbon":
            self.metrics_collector = CodeCarbonCollector(
                project_name=self.config.metrics.project_name,
                output_dir=self.config.metrics.output_dir,
                country_iso_code=self.config.metrics.country_iso_code,
                region=self.config.metrics.region,
                gpu_ids=self.config.backend.device_ids,
            )
            print("Initialized CodeCarbon metrics collector")
        else:
            raise ValueError(f"Unknown metrics type: {self.config.metrics.type}")

    def _initialize_reporter(self):
        """Initialize results reporter."""
        if self.config.reporter.type == "csv":
            self.reporter = CSVReporter(output_file=self.config.reporter.output_file)
            print("Initialized CSV reporter")
        else:
            raise ValueError(f"Unknown reporter type: {self.config.reporter.type}")

    def validate(self) -> bool:
        """Validate benchmark configuration and environment.

        Returns:
            bool: True if validation passes
        """
        print("Validating benchmark environment...")

        # Validate config
        try:
            ConfigParser.validate_config(self.config)
        except ValueError as e:
            print(f"Config validation failed: {e}")
            return False

        # Validate backend
        backend = self.backend
        if backend is None or not backend.validate_environment():
            print("Backend validation failed")
            return False

        if not backend.health_check():
            print("Backend health check failed")
            return False

        # Validate dataset
        dataset = self.dataset
        if dataset is None or not dataset.validate():
            print("Dataset validation failed")
            return False

        # Validate reporter
        reporter = self.reporter
        if reporter is None or not reporter.validate():
            print("Reporter validation failed")
            return False

        print("Validation passed!")
        return True

    def run(self) -> Dict[str, Any]:
        """Execute the benchmark.

        Returns:
            Dict with benchmark results
        """
        print(f"\nStarting benchmark: {self.config.name}")
        print(f"Backend: {self.config.backend.type}")
        print(f"Model: {self.config.backend.model}")
        print(f"Dataset: {self.config.scenario.dataset_name}")
        print(f"Samples: {self.config.scenario.num_samples}\n")

        # Validate environment
        if not self.validate():
            raise RuntimeError("Validation failed. Cannot run benchmark.")

        # Load dataset
        print("Loading dataset...")
        dataset = self.dataset
        if dataset is None:
            raise RuntimeError("Dataset not initialized")

        prompts = dataset.load(
            {
                "name": self.config.scenario.dataset_name,
                "text_column": self.config.scenario.text_column_name,
                "num_samples": self.config.scenario.num_samples,
            }
        )

        # Start metrics collection
        collector = self.metrics_collector
        if collector is not None:
            print("Starting metrics collection...")
            collector.start()

        # Check GPU availability and collect initial stats
        gpu_ids = self.config.backend.device_ids
        gpu_availability = GPUMonitor.check_multi_gpu_available(gpu_ids)
        if not gpu_availability["all_available"]:
            print(f"Warning: Some GPUs unavailable: {gpu_availability['unavailable_gpus']}")
        else:
            print(f"All GPUs available: {gpu_ids}")

        # Run inference on all prompts
        start_time = time.time()
        inference_results = []

        # Prepare generation kwargs
        gen_kwargs = {
            "max_tokens": self.config.scenario.generate_kwargs.get("max_new_tokens", 100),
            "temperature": 0.7,
        }

        # Add reasoning parameters if enabled
        if self.config.scenario.reasoning and self.config.scenario.reasoning_params:
            print(f"Reasoning enabled with params: {self.config.scenario.reasoning_params}")
            gen_kwargs["reasoning_params"] = self.config.scenario.reasoning_params

        print("Running inference...")
        backend = self.backend
        if backend is None:
            raise RuntimeError("Backend not initialized")

        consecutive_failures = 0
        max_consecutive_failures = 3  # Fail fast after 3 consecutive failures

        for i, prompt in enumerate(prompts):
            prompt_start = time.time()
            print(f"  Processing prompt {i + 1}/{len(prompts)}...", flush=True)

            result = backend.run_inference(prompt, **gen_kwargs)
            inference_results.append(result)

            prompt_time = time.time() - prompt_start

            # Check for failures and implement fail-fast logic
            if not result.get("success", False):
                consecutive_failures += 1
                error_msg = result.get("error", "Unknown error")
                print(f"    FAILED in {prompt_time:.1f}s: {error_msg}", flush=True)

                # Fail fast if we hit too many consecutive failures
                if consecutive_failures >= max_consecutive_failures:
                    print(
                        f"\n{'=' * 70}\n"
                        f"ERROR: {consecutive_failures} consecutive failures detected.\n"
                        f"Stopping benchmark early to avoid wasting resources.\n"
                        f"Last error: {error_msg}\n"
                        f"{'=' * 70}\n",
                        flush=True,
                    )
                    # Clean up and exit
                    if collector is not None:
                        print("Stopping metrics collection...")
                        collector.stop()
                    raise RuntimeError(
                        f"Benchmark failed: {consecutive_failures} consecutive inference failures. "
                        f"Last error: {error_msg}"
                    )
            else:
                # Reset counter on success
                consecutive_failures = 0
                print(f"    Completed in {prompt_time:.1f}s", flush=True)

        end_time = time.time()
        print(f"\nInference completed in {end_time - start_time:.2f} seconds")

        # Collect final GPU stats
        final_gpu_stats = GPUMonitor.get_multi_gpu_stats(gpu_ids)
        print("\nFinal GPU Statistics:")
        GPUMonitor.print_multi_gpu_info(gpu_ids)

        # Stop metrics collection
        energy_metrics: Dict[str, Any] = {}
        if collector is not None:
            print("Stopping metrics collection...")
            energy_metrics = collector.stop()

        # Aggregate results
        results = self._aggregate_results(
            inference_results, energy_metrics, end_time - start_time, final_gpu_stats
        )

        # Report results
        print("Reporting results...")
        reporter = self.reporter
        if reporter is None:
            raise RuntimeError("Reporter not initialized")

        reporter.report(results)

        print("\n=== Benchmark Complete ===")
        print(f"Total prompts: {len(prompts)}")
        print(f"Successful: {results['summary']['successful_prompts']}")
        print(f"Failed: {results['summary']['failed_prompts']}")
        print(f"Duration: {results['summary']['total_duration_seconds']:.2f}s")
        if energy_metrics:
            print(f"Energy: {energy_metrics.get('energy_wh', 0):.2f} Wh")
            print(f"CO2: {energy_metrics.get('emissions_g_co2eq', 0):.2f} g")

        return results

    def _aggregate_results(
        self,
        inference_results: List[Dict[str, Any]],
        energy_metrics: Dict[str, Any],
        total_duration: float,
        gpu_stats: Dict[int, Any],
    ) -> Dict[str, Any]:
        """Aggregate benchmark results.

        Args:
            inference_results: List of inference results
            energy_metrics: Energy metrics from collector
            total_duration: Total benchmark duration
            gpu_stats: Final GPU statistics per device

        Returns:
            Aggregated results dictionary
        """
        successful = [r for r in inference_results if r.get("success", False)]
        failed = [r for r in inference_results if not r.get("success", False)]

        # Calculate stats
        total_tokens = sum(r.get("total_tokens", 0) for r in successful)
        total_prompt_tokens = sum(r.get("prompt_tokens", 0) for r in successful)
        total_completion_tokens = sum(r.get("completion_tokens", 0) for r in successful)

        avg_latency = (
            sum(r.get("latency_seconds", 0) for r in successful) / len(successful)
            if successful
            else 0
        )

        # Calculate average TTFT (Time to First Token)
        ttft_values_raw = [
            r.get("time_to_first_token")
            for r in successful
            if r.get("time_to_first_token") is not None
        ]
        ttft_values = [float(v) for v in ttft_values_raw if v is not None]
        avg_ttft = sum(ttft_values) / len(ttft_values) if ttft_values else 0.0

        # Process GPU stats for reporting
        gpu_stats_summary = {}
        for gpu_id, stats in gpu_stats.items():
            if stats is not None:
                gpu_stats_summary[f"gpu_{gpu_id}"] = {
                    "utilization_percent": stats.utilization_percent,
                    "memory_used_mb": stats.memory_used_mb,
                    "memory_total_mb": stats.memory_total_mb,
                    "memory_percent": stats.memory_percent,
                    "temperature_c": stats.temperature_c,
                    "power_draw_w": stats.power_draw_w,
                }

        return {
            "config": {
                "name": self.config.name,
                "backend": self.config.backend.type,
                "model": self.config.backend.model,
                "dataset": self.config.scenario.dataset_name,
                "num_samples": self.config.scenario.num_samples,
                "device_ids": self.config.backend.device_ids,
            },
            "summary": {
                "total_prompts": len(inference_results),
                "successful_prompts": len(successful),
                "failed_prompts": len(failed),
                "total_duration_seconds": total_duration,
                "avg_latency_seconds": avg_latency,
                "avg_time_to_first_token": avg_ttft,
                "total_tokens": total_tokens,
                "total_prompt_tokens": total_prompt_tokens,
                "total_completion_tokens": total_completion_tokens,
                "throughput_tokens_per_second": (
                    total_tokens / total_duration if total_duration > 0 else 0
                ),
            },
            "energy": energy_metrics,
            "gpu_stats": gpu_stats_summary,
            "backend_info": self._get_backend_info(),
            "metrics_metadata": self._get_metrics_metadata(),
        }

    def _get_backend_info(self) -> Dict[str, Any]:
        if self.backend is None:
            raise RuntimeError("Backend not initialized")
        return self.backend.get_endpoint_info()

    def _get_metrics_metadata(self) -> Dict[str, Any]:
        if self.metrics_collector is None:
            return {}
        return self.metrics_collector.get_metadata()


def run_benchmark_from_config(
    config_path: str, overrides: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Run benchmark from configuration file.

    Args:
        config_path: Path to configuration file
        overrides: Optional configuration overrides

    Returns:
        Benchmark results
    """
    # Load config
    if overrides:
        config = ConfigParser.load_config_with_overrides(config_path, overrides)
    else:
        config = ConfigParser.load_config(config_path)

    # Create and run benchmark
    runner = BenchmarkRunner(config)
    return runner.run()
