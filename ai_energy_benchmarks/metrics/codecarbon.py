"""CodeCarbon metrics collector for comprehensive energy measurement."""

import os
from typing import Any, Dict, Optional, Protocol, cast

from ai_energy_benchmarks.metrics.base import MetricsCollector


class _TrackerProtocol(Protocol):
    """Simplified protocol describing the CodeCarbon tracker behaviour."""

    final_emissions_data: Any  # CodeCarbon supplies a rich data object we treat opaquely

    def start(self) -> None: ...

    def stop(self) -> Optional[float]:  # Tracker returns total emissions in kilograms
        ...


class CodeCarbonCollector(MetricsCollector):
    """CodeCarbon-based energy and emissions tracker.

    Provides comprehensive energy measurement including:
    - GPU energy (NVIDIA/AMD/Intel) - PRIMARY METRIC
    - CPU energy (RAPL)
    - RAM energy
    - Carbon emissions (CO2eq)

    Methodology aligned with AIEnergyScore:
    - Reports GPU-only energy as primary metric for standardized comparison
    - Full energy breakdown (GPU/CPU/RAM) available in detailed metrics
    """

    def __init__(
        self,
        project_name: str = "ai_energy_benchmarks",
        output_dir: str = "./emissions",
        country_iso_code: str = "USA",
        region: Optional[str] = None,
        gpu_ids: Optional[list] = None,
        save_to_file: bool = True,
        log_level: str = "warning",
    ):
        """Initialize CodeCarbon tracker.

        Args:
            project_name: Project name for tracking
            output_dir: Directory for emission reports
            country_iso_code: Country code for carbon intensity
            region: Specific region for carbon intensity
            gpu_ids: List of GPU IDs to monitor
            save_to_file: Whether to save emissions to CSV
            log_level: Logging level
        """
        self.project_name = project_name
        self.output_dir = output_dir
        self.country_iso_code = country_iso_code
        self.region = region
        self.gpu_ids = gpu_ids
        self.save_to_file = save_to_file
        self.log_level = log_level
        self.tracker: Optional[_TrackerProtocol] = None
        self._tracker_started: bool = False

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

    def start(self):
        """Start energy tracking."""
        try:
            from codecarbon import EmissionsTracker  # type: ignore[import-untyped]

            # CodeCarbon 3.0+ doesn't use country_iso_code/region
            # It uses tracking_mode and co2_signal_api_token instead
            tracker = cast(
                _TrackerProtocol,
                EmissionsTracker(
                    project_name=self.project_name,
                    output_dir=self.output_dir,
                    log_level=self.log_level,
                    save_to_file=self.save_to_file,
                    gpu_ids=self.gpu_ids,
                    tracking_mode="machine",  # Use machine mode for local tracking
                ),
            )

            self.tracker = tracker
            tracker.start()
            self._tracker_started = True
            print("CodeCarbon tracker started (version 3.0+, machine mode)")

        except ImportError:
            print("Warning: codecarbon not installed. Energy metrics will not be collected.")
            print("Install with: pip install codecarbon")
        except Exception as e:
            print(f"Error starting CodeCarbon tracker: {e}")
            import traceback

            traceback.print_exc()

    def stop(self) -> Dict[str, Any]:
        """Stop tracking and return metrics.

        Returns GPU-only energy as primary metric (aligned with AIEnergyScore).
        Full breakdown (GPU/CPU/RAM) available in detailed fields.

        Returns:
            Dict with energy and emissions data
        """
        if not self._tracker_started or self.tracker is None:
            return {
                "emissions_kg_co2eq": 0.0,
                "energy_kwh": 0.0,
                "energy_wh": 0.0,
                "gpu_energy_kwh": 0.0,
                "gpu_energy_wh": 0.0,
                "cpu_energy_kwh": 0.0,
                "cpu_energy_wh": 0.0,
                "ram_energy_kwh": 0.0,
                "ram_energy_wh": 0.0,
                "total_energy_kwh": 0.0,
                "total_energy_wh": 0.0,
                "duration_seconds": 0.0,
                "error": "Tracker not started or codecarbon not installed",
            }

        try:
            emissions_kg = self.tracker.stop()

            # Get detailed emissions data
            if hasattr(self.tracker, "final_emissions_data"):
                data = self.tracker.final_emissions_data

                # Extract component energies (in kWh)
                gpu_energy_kwh = data.gpu_energy if hasattr(data, "gpu_energy") else 0.0
                cpu_energy_kwh = data.cpu_energy if hasattr(data, "cpu_energy") else 0.0
                ram_energy_kwh = data.ram_energy if hasattr(data, "ram_energy") else 0.0
                total_energy_kwh = data.energy_consumed if hasattr(data, "energy_consumed") else 0.0

                duration = data.duration if hasattr(data, "duration") else 0.0
                carbon_intensity = data.emissions_rate if hasattr(data, "emissions_rate") else 0.0
            else:
                # Fallback if final_emissions_data not available
                gpu_energy_kwh = 0.0
                cpu_energy_kwh = 0.0
                ram_energy_kwh = 0.0
                total_energy_kwh = 0.0
                duration = 0.0
                carbon_intensity = 0.0

            self._tracker_started = False

            # Primary metric: GPU energy (aligned with AIEnergyScore methodology)
            # This enables standardized model comparison across different CPU/RAM configurations
            return {
                "emissions_kg_co2eq": emissions_kg or 0.0,
                "emissions_g_co2eq": (emissions_kg or 0.0) * 1000,
                # Primary energy metric (GPU-only, matches AIEnergyScore)
                "energy_kwh": gpu_energy_kwh,
                "energy_wh": gpu_energy_kwh * 1000,
                # Detailed energy breakdown
                "gpu_energy_kwh": gpu_energy_kwh,
                "gpu_energy_wh": gpu_energy_kwh * 1000,
                "cpu_energy_kwh": cpu_energy_kwh,
                "cpu_energy_wh": cpu_energy_kwh * 1000,
                "ram_energy_kwh": ram_energy_kwh,
                "ram_energy_wh": ram_energy_kwh * 1000,
                # Total system energy (GPU+CPU+RAM)
                "total_energy_kwh": total_energy_kwh,
                "total_energy_wh": total_energy_kwh * 1000,
                # Other metrics
                "duration_seconds": duration,
                "carbon_intensity_g_per_kwh": carbon_intensity,
                "project_name": self.project_name,
                "country": self.country_iso_code,
                "region": self.region,
            }

        except Exception as e:
            return {
                "emissions_kg_co2eq": 0.0,
                "energy_kwh": 0.0,
                "gpu_energy_kwh": 0.0,
                "cpu_energy_kwh": 0.0,
                "ram_energy_kwh": 0.0,
                "total_energy_kwh": 0.0,
                "error": f"Error stopping tracker: {e}",
            }

    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current metrics without stopping.

        Returns:
            Dict with current metrics (limited data available)
        """
        if not self._tracker_started:
            return {"error": "Tracker not started"}

        # CodeCarbon doesn't provide current metrics, only final
        return {"status": "tracking", "note": "Metrics available after stop()"}

    def get_metadata(self) -> Dict[str, Any]:
        """Get collector metadata.

        Returns:
            Dict with metadata
        """
        try:
            import codecarbon

            version = codecarbon.__version__
        except Exception:
            version = "unknown"

        return {
            "name": "codecarbon",
            "version": version,
            "capabilities": ["gpu_energy", "cpu_energy", "ram_energy", "carbon_emissions"],
            "project_name": self.project_name,
            "output_dir": self.output_dir,
        }
