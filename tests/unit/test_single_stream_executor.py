"""Tests for SingleStreamExecutor and its CLI."""

import json
from unittest.mock import MagicMock, patch

import pytest

from ai_energy_benchmarks.executors.single_stream import (
    GpuPowerSampler,
    SingleStreamExecutor,
)
from ai_energy_benchmarks.profiles import LoadProfileConfig


def _make_profile(request_count: int = 2) -> LoadProfileConfig:
    """Build a minimal single-stream profile for tests."""
    return LoadProfileConfig(
        name="single_stream_light",
        description="test",
        concurrency=1,
        request_count=request_count,
        input_token_range=(50, 100),
        output_token_range=(50, 100),
        cache_strategy="minimal",
    )


def _make_chat_response(prompt_tokens: int = 50, completion_tokens: int = 100) -> bytes:
    """Build a fake OpenAI-style chat completion JSON response body."""
    return json.dumps(
        {
            "id": "test",
            "choices": [{"message": {"content": "hi"}, "finish_reason": "stop"}],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }
    ).encode()


class _FakeUrlopenResponse:
    """Context-manager stub matching urllib.request.urlopen()'s interface."""

    def __init__(self, body: bytes, status: int = 200):
        self._body = body
        self.status = status

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self) -> bytes:
        return self._body


@pytest.fixture
def deterministic_sampler():
    """Force GpuPowerSampler to return a fixed avg_power_watts.

    Avoids depending on a real nvidia-smi in CI. The threading start/stop
    behaviour is bypassed entirely so tests don't sleep.
    """
    with (
        patch.object(GpuPowerSampler, "start", lambda self: None),
        patch.object(GpuPowerSampler, "stop", lambda self: None),
        patch.object(
            GpuPowerSampler,
            "avg_power_watts",
            new=property(lambda self: 250.0),
        ),
    ):
        yield


class TestGpuPowerSampler:
    """Tests for the standalone GpuPowerSampler."""

    def test_avg_power_with_no_samples_is_zero(self):
        """An idle sampler reports 0.0 W instead of dividing by zero."""
        sampler = GpuPowerSampler()
        assert sampler.avg_power_watts == 0.0

    def test_avg_power_averages_collected_samples(self):
        sampler = GpuPowerSampler()
        sampler.samples = [100.0, 200.0, 300.0]
        assert sampler.avg_power_watts == 200.0

    def test_sample_loop_handles_missing_nvidia_smi(self):
        """If nvidia-smi isn't on PATH, the loop drops samples silently.

        Regression-style: the sampler is used in mock test environments
        without an NVIDIA driver; a FileNotFoundError must not crash the
        background thread.
        """
        sampler = GpuPowerSampler(interval_s=0.01)
        with patch("subprocess.run", side_effect=FileNotFoundError("no nvidia-smi")):
            sampler.start()
            # Let the loop attempt one iteration, then stop
            import time

            time.sleep(0.05)
            sampler.stop()
        assert sampler.samples == []

    def test_sample_loop_handles_unparseable_output(self):
        """Bad output from nvidia-smi is dropped, not propagated."""
        sampler = GpuPowerSampler(interval_s=0.01)
        fake_result = MagicMock(returncode=0, stdout="not-a-float\n")
        with patch("subprocess.run", return_value=fake_result):
            sampler.start()
            import time

            time.sleep(0.05)
            sampler.stop()
        assert sampler.samples == []


class TestSingleStreamExecutor:
    """Tests for SingleStreamExecutor's request loop and aggregation."""

    def test_run_populates_energy_from_power_and_duration(self, deterministic_sampler):
        """energy_joules must equal avg_power_watts * request_duration_seconds.

        This is the core invariant of single-stream attribution.
        """
        profile = _make_profile(request_count=1)
        executor = SingleStreamExecutor(seed=42)

        with patch(
            "urllib.request.urlopen",
            return_value=_FakeUrlopenResponse(_make_chat_response(50, 100)),
        ):
            result = executor.run(profile, "http://localhost:8000/v1", "test-model")

        assert result.successful_requests == 1
        assert result.failed_requests == 0
        req = result.individual_results[0]
        assert req.energy_joules is not None
        # avg_power_watts is forced to 250.0 by the fixture
        assert req.avg_power_watts == 250.0
        # Allow some floating-point slop on the duration multiplication
        assert req.energy_joules == pytest.approx(250.0 * req.request_duration_seconds, rel=1e-9)

    def test_run_forces_concurrency_to_1(self, deterministic_sampler):
        """Even if a profile claims concurrency=4, single-stream must serialize.

        Per-request power attribution is only valid when requests don't
        overlap on the GPU; the executor must ignore the profile's
        concurrency field rather than silently produce invalid data.
        """
        profile = LoadProfileConfig(
            name="lying_profile",
            description="profile claims concurrency=4 — executor must ignore",
            concurrency=4,
            request_count=3,
            input_token_range=(50, 100),
            output_token_range=(50, 100),
            cache_strategy="minimal",
        )
        executor = SingleStreamExecutor(seed=42)
        with patch(
            "urllib.request.urlopen",
            return_value=_FakeUrlopenResponse(_make_chat_response()),
        ):
            result = executor.run(profile, "http://localhost:8000/v1", "test-model")

        assert result.concurrency == 1
        assert result.successful_requests == 3

    def test_run_records_http_error(self, deterministic_sampler):
        """HTTP failures land as RequestResult.error, not exceptions."""
        import urllib.error

        profile = _make_profile(request_count=1)
        executor = SingleStreamExecutor(seed=42)

        http_err = urllib.error.HTTPError(
            url="http://localhost:8000/v1/chat/completions",
            code=503,
            msg="Service Unavailable",
            hdrs=None,  # type: ignore[arg-type]
            fp=None,
        )
        with patch("urllib.request.urlopen", side_effect=http_err):
            result = executor.run(profile, "http://localhost:8000/v1", "test-model")

        assert result.successful_requests == 0
        assert result.failed_requests == 1
        assert result.individual_results[0].status_code == 503
        assert "HTTP 503" in (result.individual_results[0].error or "")

    def test_endpoint_v1_suffix_is_appended_when_missing(self, deterministic_sampler):
        """Passing 'http://host:8000' must produce requests to '/v1/chat/completions'."""
        profile = _make_profile(request_count=1)
        executor = SingleStreamExecutor(seed=42)

        captured_urls = []

        def _capture(req, *args, **kwargs):
            captured_urls.append(req.full_url)
            return _FakeUrlopenResponse(_make_chat_response())

        with patch("urllib.request.urlopen", side_effect=_capture):
            executor.run(profile, "http://localhost:8000", "test-model")

        assert captured_urls == ["http://localhost:8000/v1/chat/completions"]

    def test_endpoint_with_trailing_slash_is_normalized(self, deterministic_sampler):
        """'http://host:8000/v1/' must not double up the slashes."""
        profile = _make_profile(request_count=1)
        executor = SingleStreamExecutor(seed=42)

        captured_urls = []

        def _capture(req, *args, **kwargs):
            captured_urls.append(req.full_url)
            return _FakeUrlopenResponse(_make_chat_response())

        with patch("urllib.request.urlopen", side_effect=_capture):
            executor.run(profile, "http://localhost:8000/v1/", "test-model")

        assert captured_urls == ["http://localhost:8000/v1/chat/completions"]

    def test_api_key_adds_bearer_header(self, deterministic_sampler):
        """When api_key is set, the request must carry an Authorization header."""
        profile = _make_profile(request_count=1)
        executor = SingleStreamExecutor(seed=42)

        captured_headers = []

        def _capture(req, *args, **kwargs):
            captured_headers.append(dict(req.headers))
            return _FakeUrlopenResponse(_make_chat_response())

        with patch("urllib.request.urlopen", side_effect=_capture):
            executor.run(profile, "http://localhost:8000/v1", "test-model", api_key="sk-test")

        # urllib normalizes header names; check case-insensitively
        merged = {k.lower(): v for k, v in captured_headers[0].items()}
        assert merged.get("authorization") == "Bearer sk-test"

    def test_aggregate_tokens_per_joule_uses_total_tokens(self, deterministic_sampler):
        """tokens_per_joule = sum(total_tokens) / sum(energy_joules).

        Energy is consumed for both prefill (input) and decode (output);
        the aggregate must reflect that.
        """
        profile = _make_profile(request_count=2)
        executor = SingleStreamExecutor(seed=42)

        with patch(
            "urllib.request.urlopen",
            return_value=_FakeUrlopenResponse(
                _make_chat_response(prompt_tokens=30, completion_tokens=70)
            ),
        ):
            result = executor.run(profile, "http://localhost:8000/v1", "test-model")

        # Each request: total_tokens=100, energy = 250 * duration
        assert result.total_tokens == 200
        assert result.tokens_per_joule is not None
        assert result.total_energy_joules is not None
        assert result.tokens_per_joule == pytest.approx(
            result.total_tokens / result.total_energy_joules
        )


class TestCliOutputFormat:
    """Verify the CLI writes the JSONL shape downstream consumers expect."""

    def test_meta_header_has_required_fields(self, tmp_path):
        """JSONL `_meta` header must contain the fields downstream looks up
        when computing benchmark_source resolution and provenance.
        """
        from ai_energy_benchmarks.cli import single_stream as cli

        args = MagicMock(
            tensor_parallel=2,
            serving_engine="vllm",
            endpoint="http://localhost:8000/v1",
            power_sample_interval=0.1,
            gpu_type="H200",
            quantization=None,
            max_model_len=32768,
            gpu_memory_utilization=0.9,
        )
        meta = cli.build_meta_header("Qwen/Qwen3-32B", "single_stream_light", args)

        # Downstream's loader needs these to land at the right benchmark_source.
        assert meta["_meta"] is True
        assert meta["benchmark_source"] == "codecarbon_sweep"
        assert meta["measurement_mode"] == "local-power-sampling"
        assert meta["concurrency"] == 1
        # The display profile name is the short form (e.g. "light"), not "single_stream_light"
        assert meta["profile_name"] == "light"
        assert meta["tensor_parallel"] == 2
        assert meta["gpu_type"] == "H200"
        assert meta["max_model_len"] == 32768

    def test_request_row_has_energy_and_token_fields(self):
        """Per-request JSONL row must carry energy, power, duration,
        token counts, and the derived metrics downstream's loader uses.
        """
        from ai_energy_benchmarks.cli import single_stream as cli
        from ai_energy_benchmarks.executors.energy_aware import RequestResult

        req = RequestResult(
            prompt_tokens=30,
            completion_tokens=70,
            total_tokens=100,
            request_duration_seconds=0.5,
            energy_joules=125.0,
            energy_kwh=125.0 / 3_600_000.0,
            avg_power_watts=250.0,
            inference_duration_seconds=0.5,
            attribution_method="local-power-sampling",
            attribution_ratio=1.0,
            status_code=200,
        )
        row = cli.request_result_to_jsonl_row(req)
        assert row["energy_joules"] == 125.0
        assert row["avg_power_watts"] == 250.0
        assert row["duration_seconds"] == 0.5
        assert row["input_tokens"] == 30
        assert row["output_tokens"] == 70
        assert row["thinking_tokens"] == 0
        # Values in the JSONL row are rounded to 4 decimals for compact output;
        # use abs tolerance that accommodates that rounding.
        assert row["tokens_per_joule"] == pytest.approx(100 / 125.0, abs=1e-4)
        assert row["energy_per_useful_token"] == pytest.approx(125.0 / 70, abs=1e-4)

    def test_request_row_for_failed_request_is_error_only(self):
        """Failed requests must serialize without bogus zero energy fields
        that would skew downstream aggregation.
        """
        from ai_energy_benchmarks.cli import single_stream as cli
        from ai_energy_benchmarks.executors.energy_aware import RequestResult

        req = RequestResult(
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
            request_duration_seconds=1.2,
            error="Connection refused",
        )
        row = cli.request_result_to_jsonl_row(req)
        assert "error" in row
        assert row["error"] == "Connection refused"
        assert "energy_joules" not in row
        assert "tokens_per_joule" not in row

    def test_write_jsonl_output_layout(self, tmp_path):
        """File must start with a single meta row, then one row per request."""
        from ai_energy_benchmarks.cli import single_stream as cli
        from ai_energy_benchmarks.executors.energy_aware import RequestResult

        out = tmp_path / "out.jsonl"
        meta = {"_meta": True, "model": "test", "benchmark_source": "codecarbon_sweep"}
        results = [
            RequestResult(
                prompt_tokens=10,
                completion_tokens=20,
                total_tokens=30,
                request_duration_seconds=0.1,
                energy_joules=25.0,
                avg_power_watts=250.0,
            )
        ]
        cli.write_jsonl_output(out, meta, results)
        lines = out.read_text().strip().split("\n")
        assert len(lines) == 2
        assert json.loads(lines[0])["_meta"] is True
        assert json.loads(lines[1])["energy_joules"] == 25.0
