"""Tests for EnergyAwareExecutor."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ai_energy_benchmarks.executors.energy_aware import (
    EnergyAwareExecutor,
    ProfileResult,
    RequestResult,
    run_sync,
)
from ai_energy_benchmarks.profiles import LoadProfileConfig


class TestRequestResult:
    """Tests for RequestResult dataclass."""

    def test_request_result_basic(self):
        """Test creating a basic RequestResult."""
        result = RequestResult(
            prompt_tokens=10,
            completion_tokens=50,
            total_tokens=60,
            request_duration_seconds=0.5,
        )
        assert result.prompt_tokens == 10
        assert result.completion_tokens == 50
        assert result.total_tokens == 60
        assert result.request_duration_seconds == 0.5
        assert result.energy_joules is None
        assert result.error is None

    def test_request_result_with_energy(self):
        """Test RequestResult with energy data."""
        result = RequestResult(
            prompt_tokens=14,
            completion_tokens=37,
            total_tokens=51,
            request_duration_seconds=0.407,
            energy_joules=256.32,
            energy_kwh=7.1199e-05,
            avg_power_watts=630.5,
            inference_duration_seconds=0.407,
            attribution_method="counter_prorated_multi_gpu_8",
            attribution_ratio=1.0,
        )
        assert result.energy_joules == 256.32
        assert result.energy_kwh == 7.1199e-05
        assert result.avg_power_watts == 630.5
        assert result.attribution_method == "counter_prorated_multi_gpu_8"

    def test_request_result_with_error(self):
        """Test RequestResult with error."""
        result = RequestResult(
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
            request_duration_seconds=1.0,
            error="Connection timeout",
            status_code=None,
        )
        assert result.error == "Connection timeout"
        assert result.total_tokens == 0


class TestProfileResult:
    """Tests for ProfileResult dataclass."""

    def test_profile_result_basic(self):
        """Test creating a basic ProfileResult."""
        from datetime import datetime, timezone

        result = ProfileResult(
            profile_name="light",
            model="test-model",
            endpoint="http://localhost:8000/v1",
            timestamp=datetime.now(timezone.utc),
            request_count=20,
            successful_requests=18,
            failed_requests=2,
            concurrency=2,
            total_tokens=1000,
            total_prompt_tokens=200,
            total_completion_tokens=800,
            total_wall_clock_seconds=5.0,
            total_inference_seconds=4.0,
            tokens_per_second=160.0,
            energy_available=False,
        )
        assert result.profile_name == "light"
        assert result.successful_requests == 18
        assert result.tokens_per_second == 160.0
        assert result.energy_available is False
        assert result.wh_per_request is None

    def test_profile_result_with_energy(self):
        """Test ProfileResult with energy metrics."""
        from datetime import datetime, timezone

        result = ProfileResult(
            profile_name="light",
            model="test-model",
            endpoint="http://localhost:8000/v1",
            timestamp=datetime.now(timezone.utc),
            request_count=20,
            successful_requests=20,
            failed_requests=0,
            concurrency=2,
            total_tokens=1000,
            total_prompt_tokens=200,
            total_completion_tokens=800,
            total_wall_clock_seconds=5.0,
            total_inference_seconds=4.0,
            tokens_per_second=160.0,
            total_energy_joules=5126.4,
            total_energy_kwh=0.001424,
            wh_per_request=0.0712,
            tokens_per_joule=0.156,
            avg_power_watts=630.5,
            energy_available=True,
        )
        assert result.energy_available is True
        assert result.total_energy_joules == 5126.4
        assert result.wh_per_request == 0.0712

    def test_profile_result_to_dict(self):
        """Test ProfileResult.to_dict() method."""
        from datetime import datetime, timezone

        ts = datetime(2024, 12, 22, 14, 30, 0, tzinfo=timezone.utc)
        result = ProfileResult(
            profile_name="moderate",
            model="test-model",
            endpoint="https://api.example.com/v1",
            timestamp=ts,
            request_count=40,
            successful_requests=40,
            failed_requests=0,
            concurrency=4,
            total_tokens=2000,
            total_prompt_tokens=400,
            total_completion_tokens=1600,
            total_wall_clock_seconds=10.0,
            total_inference_seconds=8.0,
            tokens_per_second=160.0,
            total_energy_joules=10000.0,
            total_energy_kwh=0.00278,
            wh_per_request=0.0695,
            tokens_per_joule=0.16,
            avg_power_watts=625.0,
            energy_available=True,
        )
        d = result.to_dict()
        assert d["profile"] == "moderate"
        assert d["requests"] == 40
        assert d["energy_available"] is True
        assert d["total_energy_j"] == 10000.0
        assert d["tokens_per_joule"] == 0.16
        assert "timestamp" in d


class TestEnergyAwareExecutor:
    """Tests for EnergyAwareExecutor class."""

    def test_init_with_seed(self):
        """Test executor initialization with seed."""
        executor = EnergyAwareExecutor(seed=42)
        assert executor.seed == 42

    def test_init_without_seed(self):
        """Test executor initialization without seed."""
        executor = EnergyAwareExecutor()
        assert executor.seed is None

    def test_generate_prompts_count(self):
        """Test that correct number of prompts are generated."""
        executor = EnergyAwareExecutor(seed=42)
        prompts = executor._generate_prompts(count=10, input_token_range=(100, 200))
        assert len(prompts) == 10

    def test_generate_prompts_deterministic(self):
        """Test that prompts are deterministic with same seed."""
        executor1 = EnergyAwareExecutor(seed=42)
        executor2 = EnergyAwareExecutor(seed=42)
        prompts1 = executor1._generate_prompts(count=5, input_token_range=(100, 200))
        prompts2 = executor2._generate_prompts(count=5, input_token_range=(100, 200))
        assert prompts1 == prompts2

    def test_generate_prompts_different_seeds(self):
        """Test that different seeds produce different prompts."""
        executor1 = EnergyAwareExecutor(seed=42)
        executor2 = EnergyAwareExecutor(seed=123)
        prompts1 = executor1._generate_prompts(count=5, input_token_range=(100, 200))
        prompts2 = executor2._generate_prompts(count=5, input_token_range=(100, 200))
        # The base prompts are the same but unique IDs differ
        assert prompts1 != prompts2

    def test_generate_prompts_unique_prefix(self):
        """Test that each prompt has a unique prefix."""
        executor = EnergyAwareExecutor(seed=42)
        prompts = executor._generate_prompts(count=20, input_token_range=(50, 100))
        prefixes = [p.split("]")[0] for p in prompts]
        assert len(set(prefixes)) == 20  # All unique

    @pytest.mark.asyncio
    async def test_aggregate_results_no_energy(self):
        """Test aggregation when no energy data is available."""
        executor = EnergyAwareExecutor(seed=42)
        results = [
            RequestResult(
                prompt_tokens=10,
                completion_tokens=40,
                total_tokens=50,
                request_duration_seconds=0.5,
            ),
            RequestResult(
                prompt_tokens=10,
                completion_tokens=60,
                total_tokens=70,
                request_duration_seconds=0.6,
            ),
        ]
        profile_result = executor._aggregate_results(
            profile_name="test",
            model="test-model",
            endpoint="http://localhost:8000/v1",
            concurrency=2,
            results=results,
            wall_clock_seconds=1.0,
        )
        assert profile_result.successful_requests == 2
        assert profile_result.failed_requests == 0
        assert profile_result.total_completion_tokens == 100
        assert profile_result.energy_available is False
        assert profile_result.total_energy_joules is None

    @pytest.mark.asyncio
    async def test_aggregate_results_with_energy(self):
        """Test aggregation with energy data."""
        executor = EnergyAwareExecutor(seed=42)
        results = [
            RequestResult(
                prompt_tokens=10,
                completion_tokens=40,
                total_tokens=50,
                request_duration_seconds=0.5,
                energy_joules=200.0,
                energy_kwh=5.56e-05,
                avg_power_watts=600.0,
                inference_duration_seconds=0.33,
            ),
            RequestResult(
                prompt_tokens=10,
                completion_tokens=60,
                total_tokens=70,
                request_duration_seconds=0.6,
                energy_joules=300.0,
                energy_kwh=8.33e-05,
                avg_power_watts=650.0,
                inference_duration_seconds=0.46,
            ),
        ]
        profile_result = executor._aggregate_results(
            profile_name="test",
            model="test-model",
            endpoint="http://localhost:8000/v1",
            concurrency=2,
            results=results,
            wall_clock_seconds=1.0,
        )
        assert profile_result.energy_available is True
        assert profile_result.total_energy_joules == 500.0
        assert profile_result.avg_power_watts == 625.0  # (600 + 650) / 2
        # tokens_per_joule = 120 / 500 = 0.24 (uses total_tokens since energy is consumed for both prefill and decode)
        assert profile_result.tokens_per_joule == 0.24

    @pytest.mark.asyncio
    async def test_aggregate_results_with_errors(self):
        """Test aggregation with some failed requests."""
        executor = EnergyAwareExecutor(seed=42)
        results = [
            RequestResult(
                prompt_tokens=10,
                completion_tokens=40,
                total_tokens=50,
                request_duration_seconds=0.5,
            ),
            RequestResult(
                prompt_tokens=0,
                completion_tokens=0,
                total_tokens=0,
                request_duration_seconds=1.0,
                error="Connection timeout",
            ),
        ]
        profile_result = executor._aggregate_results(
            profile_name="test",
            model="test-model",
            endpoint="http://localhost:8000/v1",
            concurrency=2,
            results=results,
            wall_clock_seconds=1.5,
        )
        assert profile_result.successful_requests == 1
        assert profile_result.failed_requests == 1
        assert profile_result.total_completion_tokens == 40


class TestEnergyAwareExecutorIntegration:
    """Integration tests using mocked HTTP responses."""

    @pytest.fixture
    def mock_response_with_energy(self):
        """Create a mock response with energy data."""
        return {
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "created": 1234567890,
            "model": "test-model",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "Test response"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 14,
                "completion_tokens": 37,
                "total_tokens": 51,
            },
            "energy": {
                "energy_joules": 256.32,
                "energy_kwh": 7.1199e-05,
                "avg_power_watts": 630.5,
                "duration_seconds": 0.407,
                "attribution_method": "counter_prorated_multi_gpu_8",
                "attribution_ratio": 1.0,
            },
        }

    @pytest.fixture
    def mock_response_without_energy(self):
        """Create a mock response without energy data."""
        return {
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "created": 1234567890,
            "model": "test-model",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "Test response"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 14,
                "completion_tokens": 37,
                "total_tokens": 51,
            },
        }

    @pytest.mark.asyncio
    async def test_run_with_energy_response(self, mock_response_with_energy):
        """Test running executor with energy in response."""
        executor = EnergyAwareExecutor(seed=42)
        profile = LoadProfileConfig(
            name="test",
            concurrency=2,
            request_count=4,
            input_token_range=(50, 100),
            output_token_range=(50, 100),
        )

        # Create mock response context manager
        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.json = AsyncMock(return_value=mock_response_with_energy)

        # Mock aiohttp session - need to patch where it's used
        with patch(
            "ai_energy_benchmarks.executors.energy_aware.aiohttp.ClientSession"
        ) as mock_session_class:
            # Create proper async context manager for session
            mock_session = MagicMock()
            mock_session_class.return_value.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session_class.return_value.__aexit__ = AsyncMock(return_value=None)

            # Create proper async context manager for post
            mock_post_cm = MagicMock()
            mock_post_cm.__aenter__ = AsyncMock(return_value=mock_resp)
            mock_post_cm.__aexit__ = AsyncMock(return_value=None)
            mock_session.post = MagicMock(return_value=mock_post_cm)

            result = await executor.run(
                profile=profile,
                endpoint="http://localhost:8000/v1",
                model="test-model",
                api_key="test-key",
            )

        assert result.successful_requests == 4
        assert result.energy_available is True
        assert result.total_energy_joules == 256.32 * 4

    @pytest.mark.asyncio
    async def test_run_without_energy_response(self, mock_response_without_energy):
        """Test running executor without energy in response."""
        executor = EnergyAwareExecutor(seed=42)
        profile = LoadProfileConfig(
            name="test",
            concurrency=2,
            request_count=4,
            input_token_range=(50, 100),
            output_token_range=(50, 100),
        )

        # Create mock response
        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.json = AsyncMock(return_value=mock_response_without_energy)

        with patch(
            "ai_energy_benchmarks.executors.energy_aware.aiohttp.ClientSession"
        ) as mock_session_class:
            mock_session = MagicMock()
            mock_session_class.return_value.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session_class.return_value.__aexit__ = AsyncMock(return_value=None)

            mock_post_cm = MagicMock()
            mock_post_cm.__aenter__ = AsyncMock(return_value=mock_resp)
            mock_post_cm.__aexit__ = AsyncMock(return_value=None)
            mock_session.post = MagicMock(return_value=mock_post_cm)

            result = await executor.run(
                profile=profile,
                endpoint="http://localhost:8000/v1",
                model="test-model",
                api_key="test-key",
            )

        assert result.successful_requests == 4
        assert result.energy_available is False
        assert result.total_energy_joules is None
        assert result.total_completion_tokens == 37 * 4

    @pytest.mark.asyncio
    async def test_run_with_http_error(self):
        """Test handling HTTP errors."""
        executor = EnergyAwareExecutor(seed=42)
        profile = LoadProfileConfig(
            name="test",
            concurrency=1,
            request_count=2,
            input_token_range=(50, 100),
            output_token_range=(50, 100),
        )

        # Create mock error response
        mock_resp = AsyncMock()
        mock_resp.status = 401
        mock_resp.text = AsyncMock(return_value="Unauthorized")

        with patch(
            "ai_energy_benchmarks.executors.energy_aware.aiohttp.ClientSession"
        ) as mock_session_class:
            mock_session = MagicMock()
            mock_session_class.return_value.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session_class.return_value.__aexit__ = AsyncMock(return_value=None)

            mock_post_cm = MagicMock()
            mock_post_cm.__aenter__ = AsyncMock(return_value=mock_resp)
            mock_post_cm.__aexit__ = AsyncMock(return_value=None)
            mock_session.post = MagicMock(return_value=mock_post_cm)

            result = await executor.run(
                profile=profile,
                endpoint="http://localhost:8000/v1",
                model="test-model",
                api_key="bad-key",
            )

        assert result.successful_requests == 0
        assert result.failed_requests == 2
        assert result.individual_results[0].error is not None
        assert "401" in result.individual_results[0].error

    @pytest.mark.asyncio
    async def test_endpoint_normalization(self, mock_response_without_energy):
        """Test that endpoint URLs are normalized correctly."""
        executor = EnergyAwareExecutor(seed=42)
        profile = LoadProfileConfig(
            name="test",
            concurrency=1,
            request_count=1,
            input_token_range=(50, 100),
            output_token_range=(50, 100),
        )

        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.json = AsyncMock(return_value=mock_response_without_energy)

        captured_urls = []

        def create_post_mock(url, **kwargs):
            captured_urls.append(url)
            mock_post_cm = MagicMock()
            mock_post_cm.__aenter__ = AsyncMock(return_value=mock_resp)
            mock_post_cm.__aexit__ = AsyncMock(return_value=None)
            return mock_post_cm

        with patch(
            "ai_energy_benchmarks.executors.energy_aware.aiohttp.ClientSession"
        ) as mock_session_class:
            mock_session = MagicMock()
            mock_session_class.return_value.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session_class.return_value.__aexit__ = AsyncMock(return_value=None)
            mock_session.post = create_post_mock

            # Test with trailing slash
            await executor.run(
                profile=profile,
                endpoint="http://localhost:8000/v1/",
                model="test-model",
                api_key="test-key",
            )

        assert len(captured_urls) == 1
        assert captured_urls[0] == "http://localhost:8000/v1/chat/completions"


class TestRunSync:
    """Tests for the synchronous wrapper."""

    def test_run_sync_basic(self):
        """Test run_sync wrapper."""
        profile = LoadProfileConfig(
            name="test",
            concurrency=1,
            request_count=2,
            input_token_range=(50, 100),
            output_token_range=(50, 100),
        )

        mock_response = {
            "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
        }

        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.json = AsyncMock(return_value=mock_response)

        with patch(
            "ai_energy_benchmarks.executors.energy_aware.aiohttp.ClientSession"
        ) as mock_session_class:
            mock_session = MagicMock()
            mock_session_class.return_value.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session_class.return_value.__aexit__ = AsyncMock(return_value=None)

            mock_post_cm = MagicMock()
            mock_post_cm.__aenter__ = AsyncMock(return_value=mock_resp)
            mock_post_cm.__aexit__ = AsyncMock(return_value=None)
            mock_session.post = MagicMock(return_value=mock_post_cm)

            result = run_sync(
                profile=profile,
                endpoint="http://localhost:8000/v1",
                model="test-model",
                api_key="test-key",
                seed=42,
            )

        assert isinstance(result, ProfileResult)
        assert result.request_count == 2
