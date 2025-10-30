"""CSV reporter for benchmark results."""

import csv
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple

from ai_energy_benchmarks.reporters.base import Reporter


class CSVReporter(Reporter):
    """CSV file reporter for benchmark results."""

    def __init__(self, output_file: str):
        """Initialize CSV reporter.

        Args:
            output_file: Path to output CSV file
        """
        self.output_file = output_file

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)

    def report(self, results: Dict[str, Any]) -> None:
        """Write benchmark results to CSV file.

        Args:
            results: Benchmark results dictionary
        """
        # Flatten nested dictionaries
        flattened = self._flatten_dict(results)

        # Add timestamp
        flattened["timestamp"] = datetime.now(timezone.utc).isoformat()

        # Check if file exists to determine if we need headers
        file_exists = os.path.exists(self.output_file)

        try:
            with open(self.output_file, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=flattened.keys())

                # Write header if new file
                if not file_exists:
                    writer.writeheader()

                # Write data
                writer.writerow(flattened)

            print(f"Results written to: {self.output_file}")

        except Exception as e:
            print(f"Error writing CSV: {e}")
            raise

    def validate(self) -> bool:
        """Validate reporter configuration.

        Returns:
            bool: True if output directory is writable
        """
        try:
            output_dir = os.path.dirname(os.path.abspath(self.output_file))
            os.makedirs(output_dir, exist_ok=True)
            return os.access(output_dir, os.W_OK)
        except Exception:
            return False

    def _flatten_dict(
        self, d: Dict[str, Any], parent_key: str = "", sep: str = "_"
    ) -> Dict[str, Any]:
        """Flatten nested dictionary.

        Args:
            d: Dictionary to flatten
            parent_key: Parent key prefix
            sep: Separator for nested keys

        Returns:
            Flattened dictionary
        """
        items: List[Tuple[str, Any]] = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k

            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            elif isinstance(v, list):
                # Convert lists to comma-separated strings
                items.append((new_key, ",".join(map(str, v))))
            else:
                items.append((new_key, v))

        return dict(items)
