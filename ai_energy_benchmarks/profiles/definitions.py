"""Profile definitions for genai-perf load generation.

This module contains predefined load profiles ported from genai_perf_load_proper.py.
"""

from types import MappingProxyType
from typing import Dict, Optional, TypeGuard, Union, cast

from . import LoadProfileConfig, MultiPhaseProfile


def get_light_profile() -> LoadProfileConfig:
    """Get the light load profile (10-20% GPU)."""
    return LoadProfileConfig(
        name="light",
        description="Light load - 10-20% GPU",
        concurrency=2,
        request_count=20,
        input_token_range=(100, 300),
        output_token_range=(500, 1000),
        cache_strategy="realistic",
    )


def get_moderate_profile() -> LoadProfileConfig:
    """Get the moderate load profile (40-50% GPU)."""
    return LoadProfileConfig(
        name="moderate",
        description="Moderate load - 40-50% GPU",
        concurrency=4,
        request_count=40,
        input_token_range=(200, 400),
        output_token_range=(1000, 1500),
        cache_strategy="realistic",
    )


def get_heavy_profile() -> LoadProfileConfig:
    """Get the heavy load profile (70-80% GPU)."""
    return LoadProfileConfig(
        name="heavy",
        description="Heavy load - 70-80% GPU",
        concurrency=8,
        request_count=80,
        input_token_range=(300, 500),
        output_token_range=(1500, 2000),
        cache_strategy="low",
    )


def get_stress_profile() -> LoadProfileConfig:
    """Get the stress test profile (90-100% GPU - max power + throughput)."""
    return LoadProfileConfig(
        name="stress",
        description="Stress test - Optimized for max power + throughput",
        concurrency=24,
        request_count=240,
        input_token_range=(200, 600),
        output_token_range=(1000, 3000),
        cache_strategy="minimal",
    )


def get_multiphase_profile() -> MultiPhaseProfile:
    """Get the multiphase workload profile (multi-phase with realistic variability)."""
    return MultiPhaseProfile(
        name="multiphase",
        description="Multi-phase workload with realistic variability",
        phases=[
            LoadProfileConfig(
                name="light",
                description="Light phase",
                concurrency=2,
                request_count=10,
                input_token_range=(100, 300),
                output_token_range=(500, 1000),
            ),
            LoadProfileConfig(
                name="moderate",
                description="Moderate phase",
                concurrency=4,
                request_count=20,
                input_token_range=(200, 400),
                output_token_range=(1000, 1500),
            ),
            LoadProfileConfig(
                name="stress",
                description="Stress phase",
                concurrency=24,
                request_count=48,
                input_token_range=(200, 600),
                output_token_range=(1000, 3000),
            ),
        ],
        cache_strategy="minimal",
    )


def get_pattern_profile() -> MultiPhaseProfile:
    """Get the pattern test profile (light -> moderate -> heavy -> stress)."""
    return MultiPhaseProfile(
        name="pattern",
        description="Pattern test - light, moderate, heavy, stress phases",
        phases=[
            LoadProfileConfig(
                name="light",
                description="Light phase",
                concurrency=2,
                request_count=20,
                input_token_range=(100, 300),
                output_token_range=(500, 1000),
            ),
            LoadProfileConfig(
                name="moderate",
                description="Moderate phase",
                concurrency=8,
                request_count=40,
                input_token_range=(200, 400),
                output_token_range=(1000, 1500),
            ),
            LoadProfileConfig(
                name="heavy",
                description="Heavy phase",
                concurrency=24,
                request_count=80,
                input_token_range=(600, 1000),
                output_token_range=(3000, 6000),
            ),
            LoadProfileConfig(
                name="stress",
                description="Stress phase",
                concurrency=56,
                request_count=120,
                input_token_range=(1000, 5000),
                output_token_range=(1000, 3000),
            ),
        ],
        cache_strategy="minimal",
    )


def get_power_test_profile() -> MultiPhaseProfile:
    """Get the power test profile (extended phases for power measurement)."""
    return MultiPhaseProfile(
        name="power_test",
        description="Power test mode - extended phases for power consumption measurement",
        phases=[
            LoadProfileConfig(
                name="light",
                description="Light phase",
                concurrency=2,
                request_count=120,
                input_token_range=(100, 300),
                output_token_range=(500, 1000),
            ),
            LoadProfileConfig(
                name="moderate",
                description="Moderate phase",
                concurrency=4,
                request_count=240,
                input_token_range=(200, 400),
                output_token_range=(1000, 1500),
            ),
            LoadProfileConfig(
                name="heavy",
                description="Heavy phase",
                concurrency=8,
                request_count=480,
                input_token_range=(300, 500),
                output_token_range=(1500, 2000),
            ),
            LoadProfileConfig(
                name="stress",
                description="Stress phase",
                concurrency=20,
                request_count=600,
                input_token_range=(50, 200),
                output_token_range=(3000, 6000),
            ),
        ],
        cache_strategy="minimal",
    )


# Token length variation profiles - same request count, varying token lengths
def get_short_tokens_profile() -> LoadProfileConfig:
    """Get the short tokens profile (short input, short output)."""
    return LoadProfileConfig(
        name="short_tokens",
        description="Short tokens - short input and output for baseline energy measurement",
        concurrency=4,
        request_count=20,
        input_token_range=(50, 150),
        output_token_range=(50, 150),
        cache_strategy="minimal",
    )


def get_long_input_short_output_profile() -> LoadProfileConfig:
    """Get the long input, short output profile."""
    return LoadProfileConfig(
        name="long_input_short_output",
        description="Long input, short output - measures prefill-heavy workload energy",
        concurrency=4,
        request_count=20,
        input_token_range=(1000, 2000),
        output_token_range=(50, 150),
        cache_strategy="minimal",
    )


def get_long_tokens_profile() -> LoadProfileConfig:
    """Get the long tokens profile (long input, long output)."""
    return LoadProfileConfig(
        name="long_tokens",
        description="Long tokens - long input and output for maximum energy measurement",
        concurrency=4,
        request_count=20,
        input_token_range=(1000, 2000),
        output_token_range=(1000, 2000),
        cache_strategy="minimal",
    )


# Profile registry - lazy initialized and cached
_PROFILES_CACHE: Optional[Dict[str, Union[LoadProfileConfig, MultiPhaseProfile]]] = None


def _get_profiles_registry() -> MappingProxyType:
    """Get the profile registry, initializing it on first access.

    Returns:
        Immutable mapping proxy of profiles
    """
    global _PROFILES_CACHE
    if _PROFILES_CACHE is None:
        _PROFILES_CACHE = {
            "light": get_light_profile(),
            "moderate": get_moderate_profile(),
            "heavy": get_heavy_profile(),
            "stress": get_stress_profile(),
            "multiphase": get_multiphase_profile(),
            "pattern": get_pattern_profile(),
            "power_test": get_power_test_profile(),
            # Token length variation profiles
            "short_tokens": get_short_tokens_profile(),
            "long_input_short_output": get_long_input_short_output_profile(),
            "long_tokens": get_long_tokens_profile(),
        }
    return MappingProxyType(_PROFILES_CACHE)


def get_profile(name: str) -> Union[LoadProfileConfig, MultiPhaseProfile]:
    """Get a profile by name.

    Args:
        name: Profile name (light, moderate, heavy, stress, multiphase, pattern, power_test)

    Returns:
        LoadProfileConfig or MultiPhaseProfile

    Raises:
        ValueError: If profile name is not found
    """
    profiles = _get_profiles_registry()
    if name not in profiles:
        available = ", ".join(profiles.keys())
        raise ValueError(f"Unknown profile: {name}. Available profiles: {available}")
    return cast(Union[LoadProfileConfig, MultiPhaseProfile], profiles[name])


def list_profiles() -> list:
    """List all available profile names.

    Returns:
        List of profile names
    """
    return list(_get_profiles_registry().keys())


def is_multi_phase(
    profile: Union[LoadProfileConfig, MultiPhaseProfile],
) -> TypeGuard[MultiPhaseProfile]:
    """Check if a profile is multi-phase.

    Args:
        profile: Profile to check

    Returns:
        True if multi-phase, False otherwise
    """
    return isinstance(profile, MultiPhaseProfile)


# Backwards compatibility: expose PROFILES as an immutable mapping
PROFILES = _get_profiles_registry()


__all__ = [
    "get_profile",
    "list_profiles",
    "is_multi_phase",
    "get_light_profile",
    "get_moderate_profile",
    "get_heavy_profile",
    "get_stress_profile",
    "get_multiphase_profile",
    "get_pattern_profile",
    "get_power_test_profile",
    "get_short_tokens_profile",
    "get_long_input_short_output_profile",
    "get_long_tokens_profile",
    "PROFILES",
]
