"""Unit tests for CSV reporter."""

import csv
import os
import tempfile

from ai_energy_benchmarks.reporters.csv_reporter import CSVReporter


class TestCSVReporter:
    """Test CSV reporter."""

    def test_initialization(self):
        """Test reporter initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = os.path.join(tmpdir, "results.csv")
            reporter = CSVReporter(output_file)
            assert reporter.output_file == output_file

    def test_report_creates_file(self):
        """Test that report creates CSV file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = os.path.join(tmpdir, "results.csv")
            reporter = CSVReporter(output_file)

            results = {"name": "test", "value": 42, "nested": {"key": "value"}}

            reporter.report(results)
            assert os.path.exists(output_file)

    def test_report_flattens_nested_dict(self):
        """Test that nested dicts are flattened."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = os.path.join(tmpdir, "results.csv")
            reporter = CSVReporter(output_file)

            results = {"level1": {"level2": {"value": 123}}}

            reporter.report(results)

            # Read CSV and check flattened keys
            with open(output_file, "r") as f:
                reader = csv.DictReader(f)
                row = next(reader)
                assert "level1_level2_value" in row

    def test_validate_writable_directory(self):
        """Test validation of writable directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = os.path.join(tmpdir, "results.csv")
            reporter = CSVReporter(output_file)
            assert reporter.validate() is True

    def test_report_appends_to_existing_file(self):
        """Test that multiple reports append to file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = os.path.join(tmpdir, "results.csv")
            reporter = CSVReporter(output_file)

            reporter.report({"value": 1})
            reporter.report({"value": 2})

            # Check file has 2 data rows + 1 header
            with open(output_file, "r") as f:
                lines = f.readlines()
                assert len(lines) == 3  # header + 2 data rows
