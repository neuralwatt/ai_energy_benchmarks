"""Unit tests for profiles module."""

import pytest

from ai_energy_benchmarks.profiles import LoadProfileConfig, MultiPhaseProfile
from ai_energy_benchmarks.profiles.definitions import (
    _get_profiles_registry,
    get_heavy_profile,
    get_light_profile,
    get_long_input_short_output_profile,
    get_long_tokens_profile,
    get_moderate_profile,
    get_multiphase_profile,
    get_pattern_profile,
    get_power_test_profile,
    get_profile,
    get_short_tokens_profile,
    get_stress_profile,
    is_multi_phase,
    list_profiles,
)


class TestLoadProfileConfig:
    """Tests for LoadProfileConfig dataclass."""

    def test_default_values(self):
        """Test that LoadProfileConfig has correct default values."""
        config = LoadProfileConfig(name="test")

        assert config.name == "test"
        assert config.description == ""
        assert config.concurrency == 1
        assert config.request_count == 100
        assert config.input_token_range == (100, 500)
        assert config.output_token_range == (100, 500)
        assert config.cache_strategy == "minimal"

    def test_custom_values(self):
        """Test LoadProfileConfig with custom values."""
        config = LoadProfileConfig(
            name="custom",
            description="Custom profile",
            concurrency=8,
            input_token_range=(200, 600),
            output_token_range=(500, 1500),
            request_count=50,
            cache_strategy="realistic",
        )

        assert config.name == "custom"
        assert config.description == "Custom profile"
        assert config.concurrency == 8
        assert config.input_token_range == (200, 600)
        assert config.output_token_range == (500, 1500)
        assert config.request_count == 50
        assert config.cache_strategy == "realistic"

    def test_to_dict(self):
        """Test LoadProfileConfig.to_dict() method."""
        config = LoadProfileConfig(
            name="test",
            description="Test profile",
            concurrency=4,
            input_token_range=(100, 300),
            output_token_range=(200, 400),
            request_count=100,
            cache_strategy="low",
        )

        result = config.to_dict()

        assert result["name"] == "test"
        assert result["description"] == "Test profile"
        assert result["concurrency"] == 4
        assert result["input_token_range"] == [100, 300]  # Converted to list
        assert result["output_token_range"] == [200, 400]  # Converted to list
        assert result["request_count"] == 100
        assert result["cache_strategy"] == "low"


class TestMultiPhaseProfile:
    """Tests for MultiPhaseProfile dataclass."""

    def test_default_values(self):
        """Test that MultiPhaseProfile has correct default values."""
        profile = MultiPhaseProfile(name="test")

        assert profile.name == "test"
        assert profile.description == ""
        assert profile.phases == []
        assert profile.cache_strategy == "minimal"

    def test_with_phases(self):
        """Test MultiPhaseProfile with phases."""
        phases = [
            LoadProfileConfig(name="phase1", concurrency=2),
            LoadProfileConfig(name="phase2", concurrency=4),
        ]

        profile = MultiPhaseProfile(
            name="multi",
            description="Multi-phase test",
            phases=phases,
            cache_strategy="realistic",
        )

        assert profile.name == "multi"
        assert profile.description == "Multi-phase test"
        assert len(profile.phases) == 2
        assert profile.phases[0].name == "phase1"
        assert profile.phases[1].name == "phase2"
        assert profile.cache_strategy == "realistic"

    def test_to_dict(self):
        """Test MultiPhaseProfile.to_dict() method."""
        phases = [
            LoadProfileConfig(name="light", concurrency=2, request_count=10),
            LoadProfileConfig(name="load", concurrency=4, request_count=20),
        ]

        profile = MultiPhaseProfile(
            name="test",
            description="Test multi-phase",
            phases=phases,
        )

        result = profile.to_dict()

        assert result["name"] == "test"
        assert result["description"] == "Test multi-phase"
        assert result["type"] == "multi-phase"
        assert len(result["phases"]) == 2
        assert result["phases"][0]["name"] == "light"
        assert result["phases"][1]["name"] == "load"
        assert result["cache_strategy"] == "minimal"


class TestProfileDefinitions:
    """Tests for profile definition functions."""

    def test_get_light_profile(self):
        """Test light profile definition."""
        profile = get_light_profile()

        assert isinstance(profile, LoadProfileConfig)
        assert profile.name == "light"
        assert profile.concurrency == 2
        assert profile.request_count == 20
        assert profile.cache_strategy == "realistic"

    def test_get_moderate_profile(self):
        """Test moderate profile definition."""
        profile = get_moderate_profile()

        assert isinstance(profile, LoadProfileConfig)
        assert profile.name == "moderate"
        assert profile.concurrency == 4
        assert profile.request_count == 40

    def test_get_heavy_profile(self):
        """Test heavy profile definition."""
        profile = get_heavy_profile()

        assert isinstance(profile, LoadProfileConfig)
        assert profile.name == "heavy"
        assert profile.concurrency == 8
        assert profile.request_count == 80

    def test_get_stress_profile(self):
        """Test stress profile definition."""
        profile = get_stress_profile()

        assert isinstance(profile, LoadProfileConfig)
        assert profile.name == "stress"
        assert profile.concurrency == 24
        assert profile.request_count == 240
        assert profile.cache_strategy == "minimal"

    def test_get_multiphase_profile(self):
        """Test multiphase profile definition."""
        profile = get_multiphase_profile()

        assert isinstance(profile, MultiPhaseProfile)
        assert profile.name == "multiphase"
        assert len(profile.phases) == 3
        assert profile.phases[0].name == "light"
        assert profile.phases[1].name == "moderate"
        assert profile.phases[2].name == "stress"

    def test_get_pattern_profile(self):
        """Test pattern profile definition."""
        profile = get_pattern_profile()

        assert isinstance(profile, MultiPhaseProfile)
        assert profile.name == "pattern"
        assert len(profile.phases) == 4

    def test_get_power_test_profile(self):
        """Test power_test profile definition."""
        profile = get_power_test_profile()

        assert isinstance(profile, MultiPhaseProfile)
        assert profile.name == "power_test"
        assert len(profile.phases) == 4

    def test_get_short_tokens_profile(self):
        """Test short_tokens profile definition."""
        profile = get_short_tokens_profile()

        assert isinstance(profile, LoadProfileConfig)
        assert profile.name == "short_tokens"
        assert profile.concurrency == 4
        assert profile.request_count == 20
        assert profile.input_token_range == (50, 150)
        assert profile.output_token_range == (50, 150)

    def test_get_long_input_short_output_profile(self):
        """Test long_input_short_output profile definition."""
        profile = get_long_input_short_output_profile()

        assert isinstance(profile, LoadProfileConfig)
        assert profile.name == "long_input_short_output"
        assert profile.concurrency == 4
        assert profile.request_count == 20
        assert profile.input_token_range == (1000, 2000)
        assert profile.output_token_range == (50, 150)

    def test_get_long_tokens_profile(self):
        """Test long_tokens profile definition."""
        profile = get_long_tokens_profile()

        assert isinstance(profile, LoadProfileConfig)
        assert profile.name == "long_tokens"
        assert profile.concurrency == 4
        assert profile.request_count == 20
        assert profile.input_token_range == (1000, 2000)
        assert profile.output_token_range == (1000, 2000)


class TestProfileRegistry:
    """Tests for profile registry functions."""

    def test_list_profiles(self):
        """Test list_profiles returns all profile names."""
        profiles = list_profiles()

        assert isinstance(profiles, list)
        assert "light" in profiles
        assert "moderate" in profiles
        assert "heavy" in profiles
        assert "stress" in profiles
        assert "multiphase" in profiles
        assert "pattern" in profiles
        assert "power_test" in profiles
        # Token length variation profiles
        assert "short_tokens" in profiles
        assert "long_input_short_output" in profiles
        assert "long_tokens" in profiles
        assert len(profiles) == 10

    def test_get_profile_valid(self):
        """Test get_profile returns correct profile for valid names."""
        for name in list_profiles():
            profile = get_profile(name)
            assert profile.name == name

    def test_get_profile_invalid(self):
        """Test get_profile raises ValueError for invalid name."""
        with pytest.raises(ValueError) as excinfo:
            get_profile("nonexistent")

        assert "Unknown profile: nonexistent" in str(excinfo.value)
        assert "Available profiles:" in str(excinfo.value)

    def test_is_multi_phase_single(self):
        """Test is_multi_phase returns False for single-phase profiles."""
        assert is_multi_phase(get_light_profile()) is False
        assert is_multi_phase(get_moderate_profile()) is False
        assert is_multi_phase(get_heavy_profile()) is False
        assert is_multi_phase(get_stress_profile()) is False
        # Token length profiles are also single-phase
        assert is_multi_phase(get_short_tokens_profile()) is False
        assert is_multi_phase(get_long_input_short_output_profile()) is False
        assert is_multi_phase(get_long_tokens_profile()) is False

    def test_is_multi_phase_multi(self):
        """Test is_multi_phase returns True for multi-phase profiles."""
        assert is_multi_phase(get_multiphase_profile()) is True
        assert is_multi_phase(get_pattern_profile()) is True
        assert is_multi_phase(get_power_test_profile()) is True

    def test_profiles_registry_populated(self):
        """Test profiles registry is populated correctly."""
        profiles = _get_profiles_registry()
        assert len(profiles) == 10
        assert all(name in profiles for name in list_profiles())


class TestProfileConsistency:
    """Tests for profile consistency and validity."""

    @pytest.mark.parametrize("profile_name", list_profiles())
    def test_profile_has_required_fields(self, profile_name):
        """Test all profiles have required fields."""
        profile = get_profile(profile_name)

        assert profile.name is not None
        assert profile.name == profile_name

    @pytest.mark.parametrize("profile_name", ["light", "moderate", "heavy", "stress"])
    def test_single_phase_profiles_have_request_count(self, profile_name):
        """Test single-phase profiles have request_count."""
        profile = get_profile(profile_name)
        assert isinstance(profile, LoadProfileConfig)

        assert profile.request_count is not None
        assert profile.request_count > 0

    @pytest.mark.parametrize("profile_name", ["multiphase", "pattern", "power_test"])
    def test_multi_phase_profiles_have_phases(self, profile_name):
        """Test multi-phase profiles have at least one phase."""
        profile = get_profile(profile_name)

        assert isinstance(profile, MultiPhaseProfile)
        assert len(profile.phases) > 0

    def test_token_ranges_are_valid(self):
        """Test all profiles have valid token ranges (min <= max)."""
        single_phase_profiles = [
            "light",
            "moderate",
            "heavy",
            "stress",
            "short_tokens",
            "long_input_short_output",
            "long_tokens",
        ]
        for name in single_phase_profiles:
            profile = get_profile(name)
            assert isinstance(profile, LoadProfileConfig)

            assert profile.input_token_range[0] <= profile.input_token_range[1]
            assert profile.output_token_range[0] <= profile.output_token_range[1]

    def test_concurrency_values_increasing(self):
        """Test concurrency increases from light to stress."""
        light = get_light_profile()
        moderate = get_moderate_profile()
        heavy = get_heavy_profile()
        stress = get_stress_profile()

        assert light.concurrency < moderate.concurrency
        assert moderate.concurrency < heavy.concurrency
        assert heavy.concurrency < stress.concurrency
