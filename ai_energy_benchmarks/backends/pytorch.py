"""PyTorch backend implementation for local model inference."""

import time
from typing import Dict, Any, Optional
from ai_energy_benchmarks.backends.base import Backend


class PyTorchBackend(Backend):
    """PyTorch backend for local model inference with transformers."""

    def __init__(
        self,
        model: str,
        device: str = "cuda",
        device_ids: list = None,
        torch_dtype: str = "auto",
        device_map: str = "auto",
        max_memory: Optional[Dict] = None,
        use_harmony: bool = None
    ):
        """Initialize PyTorch backend.

        Args:
            model: HuggingFace model name or path
            device: Device to use (cuda/cpu)
            device_ids: List of GPU device IDs
            torch_dtype: Torch dtype (auto/float16/bfloat16/float32)
            device_map: Device map strategy (auto/balanced/sequential)
            max_memory: Max memory per device
            use_harmony: Enable Harmony formatting for gpt-oss models (auto-detects if None)
        """
        self.model_name = model
        self.device = device
        self.device_ids = device_ids or [0]
        self.torch_dtype = torch_dtype
        self.device_map = device_map
        self.max_memory = max_memory

        # Auto-detect Harmony formatting for gpt-oss models
        if use_harmony is None:
            self.use_harmony = 'gpt-oss' in model.lower()
        else:
            self.use_harmony = use_harmony

        self.model = None
        self.tokenizer = None
        self._initialized = False

    def validate_environment(self) -> bool:
        """Check if PyTorch and transformers are available.

        Returns:
            bool: True if environment is ready
        """
        try:
            import torch

            # Check CUDA availability if needed
            if self.device == "cuda":
                if not torch.cuda.is_available():
                    print("CUDA not available")
                    return False

                # Check if requested GPUs are available
                gpu_count = torch.cuda.device_count()
                for device_id in self.device_ids:
                    if device_id >= gpu_count:
                        print(f"GPU {device_id} not available (found {gpu_count} GPUs)")
                        return False

            # Check transformers
            try:
                import transformers
                return True
            except ImportError:
                print("transformers library not available")
                return False

        except ImportError:
            print("PyTorch not available")
            return False

    def _initialize_model(self):
        """Initialize model and tokenizer."""
        if self._initialized:
            return

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            print(f"Loading model: {self.model_name}")
            print(f"Device: {self.device}, Device Map: {self.device_map}")

            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )

            # Set padding token if not set
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            # Determine dtype - use "auto" to let transformers choose best dtype
            if self.torch_dtype == "auto":
                dtype = "auto"  # Let transformers decide (will use bfloat16 if available)
            elif self.torch_dtype == "float16":
                dtype = torch.float16
            elif self.torch_dtype == "bfloat16":
                dtype = torch.bfloat16
            else:
                dtype = torch.float32

            # Load model (use 'dtype' instead of deprecated 'torch_dtype')
            load_kwargs = {
                "trust_remote_code": True,
                "device_map": self.device_map,
            }

            # Add dtype parameter
            if dtype == "auto":
                load_kwargs["torch_dtype"] = "auto"
            else:
                load_kwargs["torch_dtype"] = dtype

            if self.max_memory:
                load_kwargs["max_memory"] = self.max_memory

            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                **load_kwargs
            )

            # Set to eval mode
            self.model.eval()

            self._initialized = True
            print(f"Model loaded successfully on {self.device}")
            print(f"Model dtype: {self.model.dtype}")

        except Exception as e:
            print(f"Error loading model: {e}")
            raise

    def format_harmony_prompt(self, text: str, reasoning_effort: str = "high") -> str:
        """Format a prompt using OpenAI Harmony formatting for gpt-oss models.

        Args:
            text: The user's question/prompt text
            reasoning_effort: Reasoning level (low, medium, high)

        Returns:
            Harmony-formatted prompt with system message and user message
        """
        # Harmony format structure with system and user messages
        harmony_prompt = (
            "<|start|>system<|message|>"
            "You are a helpful AI assistant.\n"
            f"Reasoning: {reasoning_effort}\n"
            "# Valid channels: analysis, commentary, final"
            "<|end|>\n"
            f"<|start|>user<|message|>{text}<|end|>"
        )
        return harmony_prompt

    def health_check(self) -> bool:
        """Check if backend is healthy.

        Returns:
            bool: True if model is loaded and ready
        """
        if not self._initialized:
            try:
                self._initialize_model()
                return True
            except:
                return False
        return self.model is not None

    def get_endpoint_info(self) -> Dict[str, Any]:
        """Get backend information.

        Returns:
            Dict with backend configuration and status
        """
        return {
            "backend": "pytorch",
            "model": self.model_name,
            "device": self.device,
            "device_ids": self.device_ids,
            "torch_dtype": self.torch_dtype,
            "device_map": self.device_map,
            "initialized": self._initialized,
            "healthy": self.health_check()
        }

    def run_inference(
        self,
        prompt: str,
        max_tokens: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        reasoning_params: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Run inference on a single prompt.

        Args:
            prompt: Input text prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            reasoning_params: Optional reasoning parameters for thinking models
            **kwargs: Additional generation parameters

        Returns:
            Dict with response text and metadata
        """
        # Initialize model if needed
        if not self._initialized:
            self._initialize_model()

        start_time = time.time()

        try:
            import torch

            # Apply Harmony formatting if enabled for gpt-oss models
            original_prompt = prompt
            if self.use_harmony:
                reasoning_effort = "high"  # Default
                if reasoning_params and 'reasoning_effort' in reasoning_params:
                    reasoning_effort = reasoning_params['reasoning_effort']
                prompt = self.format_harmony_prompt(prompt, reasoning_effort)
                print(f"  Using Harmony format with {reasoning_effort} reasoning")
                print(f"  Prompt preview (first 200 chars): {prompt[:200]}...")
            elif reasoning_params and 'reasoning_effort' in reasoning_params:
                # Legacy: simple reasoning prefix (deprecated)
                effort = reasoning_params['reasoning_effort']
                use_prompt_based = reasoning_params.get('use_prompt_based_reasoning', False)

                if use_prompt_based:
                    # Old format (kept for backward compatibility)
                    prompt = f"Reasoning:{effort}\n\n{prompt}"
                    print(f"  Using legacy prompt-based reasoning ({effort} effort)")
                    print(f"  Prompt preview: {prompt}")

            # Tokenize input
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True
            )

            # Move to device
            if self.device == "cuda":
                inputs = {k: v.cuda() for k, v in inputs.items()}

            prompt_tokens = inputs['input_ids'].shape[1]

            # Build generation kwargs
            gen_kwargs = {
                'max_new_tokens': max_tokens,
                'temperature': temperature,
                'top_p': top_p,
                'top_k': top_k,
                'do_sample': temperature > 0,
                'pad_token_id': self.tokenizer.pad_token_id,
            }

            # Add reasoning parameters if provided
            if reasoning_params:
                # Handle different reasoning parameter formats
                if 'reasoning_effort' in reasoning_params:
                    effort = reasoning_params['reasoning_effort']
                    use_prompt_based = reasoning_params.get('use_prompt_based_reasoning', False)

                    # Only apply parameter mapping if NOT using prompt-based approach
                    if not use_prompt_based:
                        print(f"Using reasoning effort: {effort}")

                        # Map reasoning effort to actual generation parameters
                        # This simulates OpenAI-style reasoning by adjusting sampling parameters
                        # Note: max_new_tokens is NOT modified - it stays at user-configured value
                        if effort == 'low':
                            # Fast, concise generation
                            gen_kwargs['temperature'] = 0.9
                            gen_kwargs['do_sample'] = True
                        elif effort == 'medium':
                            # Balanced generation
                            gen_kwargs['temperature'] = 0.7
                            gen_kwargs['top_p'] = 0.9
                            gen_kwargs['do_sample'] = True
                        elif effort == 'high':
                            # Thorough, detailed generation
                            gen_kwargs['temperature'] = 0.5
                            gen_kwargs['top_p'] = 0.95
                            gen_kwargs['top_k'] = 50
                            gen_kwargs['do_sample'] = True

                    # Note: Don't pass 'reasoning_effort' or 'use_prompt_based_reasoning' to model.generate()
                    # We've already used them above

                # Pass through other reasoning parameters (like thinking_budget for DeepSeek-R1)
                for key, value in reasoning_params.items():
                    if key not in gen_kwargs and key not in ['reasoning_effort', 'use_prompt_based_reasoning']:
                        gen_kwargs[key] = value

            # Merge additional kwargs
            gen_kwargs.update(kwargs)

            # Generate
            with torch.no_grad():
                try:
                    outputs = self.model.generate(
                        **inputs,
                        **gen_kwargs
                    )
                except (TypeError, ValueError) as e:
                    error_msg = str(e)
                    # Check if error is about unused model_kwargs (model doesn't support reasoning params)
                    if "model_kwargs" in error_msg or "unexpected keyword argument" in error_msg or "not used by the model" in error_msg:
                        # Model doesn't support reasoning parameters, retry without them
                        # Note: We don't print this message when using prompt-based reasoning
                        # because the reasoning is in the prompt, not the parameters
                        if not (reasoning_params and reasoning_params.get('use_prompt_based_reasoning')):
                            print(f"  Note: Model doesn't support reasoning parameters, running without them")

                        # Remove known reasoning-related parameters
                        reasoning_keys = ['reasoning_effort', 'thinking_budget', 'cot_depth', 'use_prompt_based_reasoning']
                        filtered_kwargs = {k: v for k, v in gen_kwargs.items()
                                          if k not in reasoning_keys}

                        outputs = self.model.generate(
                            **inputs,
                            **filtered_kwargs
                        )
                    else:
                        # Different error, re-raise
                        raise

            # Decode output
            generated_text = self.tokenizer.decode(
                outputs[0],
                skip_special_tokens=True
            )

            # Extract only the generated portion (remove prompt)
            prompt_text = self.tokenizer.decode(
                inputs['input_ids'][0],
                skip_special_tokens=True
            )
            completion_text = generated_text[len(prompt_text):].strip()

            completion_tokens = outputs.shape[1] - prompt_tokens
            total_tokens = outputs.shape[1]

            end_time = time.time()

            # Debug: log token generation stats when reasoning is enabled
            if reasoning_params:
                effort = reasoning_params.get('reasoning_effort', 'unknown')
                print(f"    Generated {completion_tokens} tokens ({effort} effort)")
                print(f"    Full text: {generated_text}, Total tokens: {total_tokens}")
            return {
                "text": completion_text,
                "full_text": generated_text,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
                "latency_seconds": end_time - start_time,
                "model": self.model_name,
                "success": True,
                "error": None
            }

        except Exception as e:
            end_time = time.time()
            return {
                "text": "",
                "success": False,
                "error": str(e),
                "latency_seconds": end_time - start_time
            }

    def cleanup(self):
        """Clean up model and free GPU memory."""
        if self.model is not None:
            try:
                import torch
                del self.model
                del self.tokenizer
                if self.device == "cuda":
                    torch.cuda.empty_cache()
                self._initialized = False
                print("Model cleaned up and GPU memory freed")
            except Exception as e:
                print(f"Error during cleanup: {e}")
