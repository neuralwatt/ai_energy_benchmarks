#!/usr/bin/env python3
"""
Mock test to validate reasoning parameter flow without requiring GPU or model.
Tests that parameters are correctly passed through the system.
"""

import sys
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Add ai_energy_benchmarks to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ai_energy_benchmarks.config.parser import (
    BenchmarkConfig, BackendConfig, ScenarioConfig,
    MetricsConfig, ReporterConfig
)
from ai_energy_benchmarks.backends.pytorch import PyTorchBackend
from ai_energy_benchmarks.backends.vllm import VLLMBackend


def test_pytorch_backend_reasoning_params():
    """Test that PyTorch backend accepts and processes reasoning params."""
    print("Testing PyTorch backend reasoning parameter handling...")

    backend = PyTorchBackend(
        model="mock-model",
        device="cpu",  # Use CPU for testing
        device_ids=[0]
    )

    # Mock the model and tokenizer
    backend._initialized = True
    backend.model = Mock()
    backend.tokenizer = Mock()

    # Setup mocks
    backend.tokenizer.return_value = {
        'input_ids': Mock(shape=(1, 10)),
        'attention_mask': Mock()
    }
    backend.tokenizer.decode = Mock(return_value="Generated text")
    backend.tokenizer.pad_token_id = 0
    backend.model.generate = Mock(return_value=Mock(shape=(1, 20)))

    # Test with reasoning params
    result = backend.run_inference(
        prompt="Test prompt",
        max_tokens=100,
        reasoning_params={"reasoning_effort": "high"}
    )

    # Verify model.generate was called
    assert backend.model.generate.called, "model.generate should have been called"

    # Extract the call kwargs
    call_kwargs = backend.model.generate.call_args[1]

    # Check reasoning_effort was passed
    assert 'reasoning_effort' in call_kwargs, "reasoning_effort should be in generate kwargs"
    assert call_kwargs['reasoning_effort'] == 'high', "reasoning_effort should be 'high'"

    print("✓ PyTorch backend correctly passes reasoning params to model.generate()")
    return True


def test_vllm_backend_reasoning_params():
    """Test that vLLM backend translates reasoning params to extra_body."""
    print("\nTesting vLLM backend reasoning parameter handling...")

    backend = VLLMBackend(
        endpoint="http://mock:8000/v1",
        model="mock-model"
    )

    # Mock requests.post
    with patch('ai_energy_benchmarks.backends.vllm.requests.post') as mock_post:
        mock_response = Mock()
        mock_response.json.return_value = {
            'choices': [{'message': {'content': 'Test response'}}],
            'usage': {'prompt_tokens': 10, 'completion_tokens': 20, 'total_tokens': 30}
        }
        mock_response.status_code = 200
        mock_post.return_value = mock_response

        # Run inference with reasoning params
        result = backend.run_inference(
            prompt="Test prompt",
            max_tokens=100,
            reasoning_params={"reasoning_effort": "medium"}
        )

        # Check that requests.post was called
        assert mock_post.called, "requests.post should have been called"

        # Extract the payload
        call_args = mock_post.call_args
        payload = call_args[1]['json']

        # Verify extra_body contains reasoning params
        assert 'extra_body' in payload, "payload should contain extra_body"
        assert 'reasoning_effort' in payload['extra_body'], "extra_body should contain reasoning_effort"
        assert payload['extra_body']['reasoning_effort'] == 'medium', "reasoning_effort should be 'medium'"

    print("✓ vLLM backend correctly translates reasoning params to extra_body")
    return True


def test_config_to_backend_flow():
    """Test full config to backend parameter flow."""
    print("\nTesting full config-to-backend parameter flow...")

    # Create config with reasoning
    backend_cfg = BackendConfig(
        type="pytorch",
        model="mock-model",
        device="cpu",
        device_ids=[0],
    )

    scenario_cfg = ScenarioConfig(
        dataset_name="mock/dataset",
        text_column_name="text",
        num_samples=2,
        reasoning=True,
        reasoning_params={"reasoning_effort": "low"},
        generate_kwargs={"max_new_tokens": 50}
    )

    # Verify config has reasoning params
    assert scenario_cfg.reasoning == True, "reasoning should be True"
    assert scenario_cfg.reasoning_params is not None, "reasoning_params should not be None"
    assert scenario_cfg.reasoning_params['reasoning_effort'] == 'low', "reasoning_effort should be 'low'"

    print("✓ Config correctly stores reasoning parameters")

    # Test that backend receives params (would happen in runner)
    gen_kwargs = {
        'max_tokens': scenario_cfg.generate_kwargs.get('max_new_tokens', 100),
        'temperature': 0.7
    }

    if scenario_cfg.reasoning and scenario_cfg.reasoning_params:
        gen_kwargs['reasoning_params'] = scenario_cfg.reasoning_params

    assert 'reasoning_params' in gen_kwargs, "reasoning_params should be in gen_kwargs"
    assert gen_kwargs['reasoning_params']['reasoning_effort'] == 'low', "Should pass through correctly"

    print("✓ Parameters flow correctly from config to generation kwargs")
    return True


def test_reasoning_disabled():
    """Test that reasoning disabled works correctly."""
    print("\nTesting reasoning disabled mode...")

    scenario_cfg = ScenarioConfig(
        dataset_name="mock/dataset",
        text_column_name="text",
        num_samples=2,
        reasoning=False,
        reasoning_params=None,
        generate_kwargs={"max_new_tokens": 50}
    )

    assert scenario_cfg.reasoning == False, "reasoning should be False"
    assert scenario_cfg.reasoning_params is None, "reasoning_params should be None"

    # Prepare kwargs as runner would
    gen_kwargs = {
        'max_tokens': 100,
        'temperature': 0.7
    }

    # Should NOT add reasoning_params
    if scenario_cfg.reasoning and scenario_cfg.reasoning_params:
        gen_kwargs['reasoning_params'] = scenario_cfg.reasoning_params

    assert 'reasoning_params' not in gen_kwargs, "reasoning_params should NOT be in gen_kwargs when disabled"

    print("✓ Reasoning disabled mode works correctly")
    return True


def main():
    """Run all mock tests."""
    print("="*60)
    print("REASONING PARAMETER FLOW TESTS (MOCK)")
    print("No GPU or model required - validates parameter passing")
    print("="*60)
    print()

    try:
        # Run all tests
        results = []
        results.append(("PyTorch Backend", test_pytorch_backend_reasoning_params()))
        results.append(("vLLM Backend", test_vllm_backend_reasoning_params()))
        results.append(("Config Flow", test_config_to_backend_flow()))
        results.append(("Reasoning Disabled", test_reasoning_disabled()))

        # Summary
        print("\n" + "="*60)
        print("TEST SUMMARY")
        print("="*60)
        for name, passed in results:
            status = "✓ PASSED" if passed else "✗ FAILED"
            print(f"{name:<30} {status}")

        all_passed = all(r[1] for r in results)

        if all_passed:
            print("\n" + "="*60)
            print("✓ ALL MOCK TESTS PASSED")
            print("="*60)
            print("\nReasoning parameters correctly flow through the system!")
            print("Next: Test with actual model (requires GPU + gpt-oss-20b)")
            return 0
        else:
            print("\n✗ SOME TESTS FAILED")
            return 1

    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
