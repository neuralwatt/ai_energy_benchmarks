"""Main benchmark runner for POC."""

import time
from typing import Dict, Any, List
from ai_energy_benchmarks.config.parser import BenchmarkConfig, ConfigParser
from ai_energy_benchmarks.backends.vllm import VLLMBackend
from ai_energy_benchmarks.backends.pytorch import PyTorchBackend
from ai_energy_benchmarks.datasets.huggingface import HuggingFaceDataset
from ai_energy_benchmarks.metrics.codecarbon import CodeCarbonCollector
from ai_energy_benchmarks.reporters.csv_reporter import CSVReporter


class BenchmarkRunner:
    """Main benchmark runner for POC phase."""

    def __init__(self, config: BenchmarkConfig):
        """Initialize benchmark runner.

        Args:
            config: Benchmark configuration
        """
        self.config = config
        self.backend = None
        self.dataset = None
        self.metrics_collector = None
        self.reporter = None

        # Initialize components
        self._initialize_backend()
        self._initialize_dataset()
        self._initialize_metrics()
        self._initialize_reporter()

    def _initialize_backend(self):
        """Initialize inference backend."""
        backend_type = self.config.backend.type

        if backend_type == 'vllm':
            self.backend = VLLMBackend(
                endpoint=self.config.backend.endpoint,
                model=self.config.backend.model
            )
        elif backend_type == 'pytorch':
            self.backend = PyTorchBackend(
                model=self.config.backend.model,
                device=self.config.backend.device,
                device_ids=self.config.backend.device_ids
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

        if self.config.metrics.type == 'codecarbon':
            self.metrics_collector = CodeCarbonCollector(
                project_name=self.config.metrics.project_name,
                output_dir=self.config.metrics.output_dir,
                country_iso_code=self.config.metrics.country_iso_code,
                region=self.config.metrics.region,
                gpu_ids=self.config.backend.device_ids
            )
            print("Initialized CodeCarbon metrics collector")
        else:
            raise ValueError(f"Unknown metrics type: {self.config.metrics.type}")

    def _initialize_reporter(self):
        """Initialize results reporter."""
        if self.config.reporter.type == 'csv':
            self.reporter = CSVReporter(
                output_file=self.config.reporter.output_file
            )
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
        if not self.backend.validate_environment():
            print("Backend validation failed")
            return False

        if not self.backend.health_check():
            print("Backend health check failed")
            return False

        # Validate dataset
        if not self.dataset.validate():
            print("Dataset validation failed")
            return False

        # Validate reporter
        if not self.reporter.validate():
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
        prompts = self.dataset.load({
            'name': self.config.scenario.dataset_name,
            'text_column': self.config.scenario.text_column_name,
            'num_samples': self.config.scenario.num_samples
        })

        # Start metrics collection
        if self.metrics_collector:
            print("Starting metrics collection...")
            self.metrics_collector.start()

        # Run inference on all prompts
        start_time = time.time()
        inference_results = []

        print("Running inference...")
        for i, prompt in enumerate(prompts):
            prompt_start = time.time()
            print(f"  Processing prompt {i+1}/{len(prompts)}...", flush=True)

            result = self.backend.run_inference(
                prompt,
                max_tokens=self.config.scenario.generate_kwargs.get('max_new_tokens', 100),
                temperature=0.7
            )
            inference_results.append(result)

            prompt_time = time.time() - prompt_start
            print(f"    Completed in {prompt_time:.1f}s", flush=True)

        end_time = time.time()
        print(f"\nInference completed in {end_time - start_time:.2f} seconds")

        # Stop metrics collection
        energy_metrics = {}
        if self.metrics_collector:
            print("Stopping metrics collection...")
            energy_metrics = self.metrics_collector.stop()

        # Aggregate results
        results = self._aggregate_results(
            inference_results,
            energy_metrics,
            end_time - start_time
        )

        # Report results
        print("Reporting results...")
        self.reporter.report(results)

        print(f"\n=== Benchmark Complete ===")
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
        total_duration: float
    ) -> Dict[str, Any]:
        """Aggregate benchmark results.

        Args:
            inference_results: List of inference results
            energy_metrics: Energy metrics from collector
            total_duration: Total benchmark duration

        Returns:
            Aggregated results dictionary
        """
        successful = [r for r in inference_results if r.get('success', False)]
        failed = [r for r in inference_results if not r.get('success', False)]

        # Calculate stats
        total_tokens = sum(r.get('total_tokens', 0) for r in successful)
        total_prompt_tokens = sum(r.get('prompt_tokens', 0) for r in successful)
        total_completion_tokens = sum(r.get('completion_tokens', 0) for r in successful)

        avg_latency = sum(r.get('latency_seconds', 0) for r in successful) / len(successful) if successful else 0

        return {
            'config': {
                'name': self.config.name,
                'backend': self.config.backend.type,
                'model': self.config.backend.model,
                'dataset': self.config.scenario.dataset_name,
                'num_samples': self.config.scenario.num_samples
            },
            'summary': {
                'total_prompts': len(inference_results),
                'successful_prompts': len(successful),
                'failed_prompts': len(failed),
                'total_duration_seconds': total_duration,
                'avg_latency_seconds': avg_latency,
                'total_tokens': total_tokens,
                'total_prompt_tokens': total_prompt_tokens,
                'total_completion_tokens': total_completion_tokens,
                'throughput_tokens_per_second': total_tokens / total_duration if total_duration > 0 else 0
            },
            'energy': energy_metrics,
            'backend_info': self.backend.get_endpoint_info(),
            'metrics_metadata': self.metrics_collector.get_metadata() if self.metrics_collector else {}
        }


def run_benchmark_from_config(config_path: str, overrides: Dict[str, Any] = None) -> Dict[str, Any]:
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
