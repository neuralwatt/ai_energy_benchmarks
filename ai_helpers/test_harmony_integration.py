#!/usr/bin/env python3
"""
Test Harmony formatting integration in ai_energy_benchmarks.

This validates that both PyTorch and vLLM backends automatically
apply Harmony formatting for gpt-oss models.
"""

import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ai_energy_benchmarks.backends.pytorch import PyTorchBackend
from ai_energy_benchmarks.backends.vllm import VLLMBackend


def test_harmony_auto_detection():
    """Test that Harmony formatting is auto-detected for gpt-oss models."""
    print("="*60)
    print("TEST: Harmony Auto-Detection")
    print("="*60)

    # Test 1: gpt-oss-20b should enable Harmony
    print("\n✅ Test 1: gpt-oss-20b (should enable Harmony)")
    backend_pytorch = PyTorchBackend(model="openai/gpt-oss-20b")
    backend_vllm = VLLMBackend(endpoint="http://localhost:8000/v1", model="openai/gpt-oss-20b")

    assert backend_pytorch.use_harmony == True, "PyTorch should auto-enable Harmony for gpt-oss-20b"
    assert backend_vllm.use_harmony == True, "vLLM should auto-enable Harmony for gpt-oss-20b"
    print("  ✓ PyTorch backend: Harmony enabled")
    print("  ✓ vLLM backend: Harmony enabled")

    # Test 2: gpt-oss-120b should enable Harmony
    print("\n✅ Test 2: gpt-oss-120b (should enable Harmony)")
    backend_pytorch = PyTorchBackend(model="openai/gpt-oss-120b")
    backend_vllm = VLLMBackend(endpoint="http://localhost:8000/v1", model="openai/gpt-oss-120b")

    assert backend_pytorch.use_harmony == True, "PyTorch should auto-enable Harmony for gpt-oss-120b"
    assert backend_vllm.use_harmony == True, "vLLM should auto-enable Harmony for gpt-oss-120b"
    print("  ✓ PyTorch backend: Harmony enabled")
    print("  ✓ vLLM backend: Harmony enabled")

    # Test 3: Other models should NOT enable Harmony
    print("\n✅ Test 3: llama-3.3-70b (should NOT enable Harmony)")
    backend_pytorch = PyTorchBackend(model="nvidia/Llama-3.3-70B-Instruct-FP8")
    backend_vllm = VLLMBackend(endpoint="http://localhost:8000/v1", model="nvidia/Llama-3.3-70B-Instruct-FP8")

    assert backend_pytorch.use_harmony == False, "PyTorch should NOT enable Harmony for Llama"
    assert backend_vllm.use_harmony == False, "vLLM should NOT enable Harmony for Llama"
    print("  ✓ PyTorch backend: Harmony disabled")
    print("  ✓ vLLM backend: Harmony disabled")

    # Test 4: Manual override
    print("\n✅ Test 4: Manual override (force Harmony for non-gpt-oss)")
    backend_pytorch = PyTorchBackend(model="nvidia/Llama-3.3-70B-Instruct-FP8", use_harmony=True)
    backend_vllm = VLLMBackend(endpoint="http://localhost:8000/v1", model="nvidia/Llama-3.3-70B-Instruct-FP8", use_harmony=True)

    assert backend_pytorch.use_harmony == True, "PyTorch should respect manual override"
    assert backend_vllm.use_harmony == True, "vLLM should respect manual override"
    print("  ✓ PyTorch backend: Harmony enabled (manual override)")
    print("  ✓ vLLM backend: Harmony enabled (manual override)")

    print("\n" + "="*60)
    print("✅ All auto-detection tests passed!")
    print("="*60)


def test_harmony_formatting():
    """Test that Harmony formatting produces correct format."""
    print("\n" + "="*60)
    print("TEST: Harmony Formatting Structure")
    print("="*60)

    backend_pytorch = PyTorchBackend(model="openai/gpt-oss-20b")
    backend_vllm = VLLMBackend(endpoint="http://localhost:8000/v1", model="openai/gpt-oss-20b")

    test_prompt = "Explain quantum computing"

    # Test with different reasoning efforts
    for effort in ["low", "medium", "high"]:
        print(f"\n✅ Testing {effort} reasoning effort:")

        # PyTorch
        pytorch_formatted = backend_pytorch.format_harmony_prompt(test_prompt, effort)
        assert "<|start|>system<|message|>" in pytorch_formatted
        assert f"Reasoning: {effort}" in pytorch_formatted
        assert "# Valid channels: analysis, commentary, final" in pytorch_formatted
        assert f"<|start|>user<|message|>{test_prompt}<|end|>" in pytorch_formatted
        print(f"  ✓ PyTorch: Correct Harmony format with {effort} reasoning")

        # vLLM
        vllm_formatted = backend_vllm.format_harmony_prompt(test_prompt, effort)
        assert "<|start|>system<|message|>" in vllm_formatted
        assert f"Reasoning: {effort}" in vllm_formatted
        assert "# Valid channels: analysis, commentary, final" in vllm_formatted
        assert f"<|start|>user<|message|>{test_prompt}<|end|>" in vllm_formatted
        print(f"  ✓ vLLM: Correct Harmony format with {effort} reasoning")

        # Verify both backends produce identical format
        assert pytorch_formatted == vllm_formatted, "PyTorch and vLLM should produce identical Harmony format"
        print(f"  ✓ Both backends produce identical format")

    # Show example
    print("\n" + "="*60)
    print("Example Harmony Format (high reasoning):")
    print("="*60)
    example = backend_pytorch.format_harmony_prompt("What is machine learning?", "high")
    print(example)

    print("\n" + "="*60)
    print("✅ All formatting tests passed!")
    print("="*60)


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("AI ENERGY BENCHMARKS - HARMONY INTEGRATION TESTS")
    print("="*60)

    try:
        test_harmony_auto_detection()
        test_harmony_formatting()

        print("\n" + "="*60)
        print("🎉 ALL TESTS PASSED!")
        print("="*60)
        print("\nKey Features:")
        print("✅ Automatic detection for gpt-oss models")
        print("✅ Proper Harmony format structure")
        print("✅ Reasoning effort integration")
        print("✅ Manual override support")
        print("✅ Consistent across PyTorch and vLLM backends")
        print("\nNow run test_reasoning_levels.py to test with actual models!")

        return 0

    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        return 1
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
