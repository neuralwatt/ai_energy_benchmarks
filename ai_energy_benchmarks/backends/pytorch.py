"""PyTorch backend implementation for local model inference."""

import importlib.util
import time
import warnings
from typing import Any, Dict, List, Optional

from ai_energy_benchmarks.backends.base import Backend
from ai_energy_benchmarks.formatters.registry import FormatterRegistry


class PyTorchBackend(Backend):
    """PyTorch backend for local model inference with transformers."""

    def __init__(
        self,
        model: str,
        device: str = "cuda",
        device_ids: Optional[List[int]] = None,
        torch_dtype: str = "auto",
        device_map: str = "auto",
        max_memory: Optional[Dict[str, Any]] = None,
        use_harmony: Optional[bool] = None,
    ):
        """Initialize PyTorch backend.

        Args:
            model: HuggingFace model name or path
            device: Device to use (cuda/cpu)
            device_ids: List of GPU device IDs
            torch_dtype: Torch dtype (auto/float16/bfloat16/float32)
            device_map: Device map strategy (auto/balanced/sequential)
            max_memory: Max memory per device
            use_harmony: DEPRECATED - Enable Harmony formatting for gpt-oss models (auto-detects if None)
        """
        self.model_name = model
        self.device = device
        self.device_ids = list(device_ids) if device_ids is not None else [0]
        self.torch_dtype = torch_dtype
        self.device_map = device_map
        self.max_memory = max_memory

        # DEPRECATED: Backward compatibility
        self._legacy_use_harmony: Optional[bool]
        if use_harmony is not None:
            warnings.warn(
                "use_harmony parameter is deprecated. "
                "Reasoning format is now auto-detected via FormatterRegistry. "
                "This parameter will be removed in v2.0.",
                DeprecationWarning,
                stacklevel=2,
            )
            self._legacy_use_harmony = use_harmony
        else:
            self._legacy_use_harmony = None

        # Auto-detect Harmony formatting for gpt-oss models (legacy)
        detected_use_harmony = (
            use_harmony if use_harmony is not None else "gpt-oss" in model.lower()
        )
        self.use_harmony: bool = bool(detected_use_harmony)

        # Initialize formatter registry
        self.formatter_registry = FormatterRegistry()
        self.formatter = self.formatter_registry.get_formatter(model)

        self.model: Any = None
        self.tokenizer: Any = None
        self._initialized: bool = False

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

            # Check transformers availability without importing the package fully
            if importlib.util.find_spec("transformers") is None:
                print("transformers library not available")
                return False

            return True

        except ImportError:
            print("PyTorch not available")
            return False

    def _initialize_model(self):
        """Initialize model and tokenizer with retry logic for network timeouts."""
        if self._initialized:
            return

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        print(f"Loading model: {self.model_name}")
        print(f"Device: {self.device}, Device Map: {self.device_map}")

        # Retry configuration
        max_retries = 3
        base_delay = 2  # seconds
        max_delay = 30  # seconds

        # Load tokenizer with retry
        for attempt in range(max_retries):
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name,
                    trust_remote_code=True,
                )
                break
            except Exception as e:
                error_str = str(e).lower()
                is_timeout = "timeout" in error_str or "timed out" in error_str
                is_network = "connection" in error_str or "network" in error_str

                if (is_timeout or is_network) and attempt < max_retries - 1:
                    delay = min(base_delay * (2**attempt), max_delay)
                    print(f"  Tokenizer loading failed (attempt {attempt + 1}/{max_retries}): {e}")
                    print(f"  Retrying in {delay}s...")
                    time.sleep(delay)
                else:
                    print(f"Error loading tokenizer: {e}")
                    raise

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        requested_dtype = self.torch_dtype
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }

        requested_torch_dtype: Optional[torch.dtype]
        if requested_dtype == "auto":
            requested_torch_dtype = None
            torch_dtype_param: Any = "auto"
        else:
            requested_torch_dtype = dtype_map.get(requested_dtype, torch.float32)
            torch_dtype_param = requested_torch_dtype

        load_kwargs = {
            "trust_remote_code": True,
            "device_map": self.device_map,
            "torch_dtype": torch_dtype_param,
        }

        if self.max_memory:
            load_kwargs["max_memory"] = self.max_memory

        # Load model with retry
        for attempt in range(max_retries):
            try:
                self.model = AutoModelForCausalLM.from_pretrained(self.model_name, **load_kwargs)
                break
            except Exception as e:
                error_str = str(e).lower()
                is_timeout = "timeout" in error_str or "timed out" in error_str
                is_network = "connection" in error_str or "network" in error_str
                is_oom = "out of memory" in error_str or "oom" in error_str

                # Don't retry OOM errors
                if is_oom:
                    print(f"Error loading model (OOM - not retrying): {e}")
                    raise

                if (is_timeout or is_network) and attempt < max_retries - 1:
                    delay = min(base_delay * (2**attempt), max_delay)
                    print(f"  Model loading failed (attempt {attempt + 1}/{max_retries}): {e}")
                    print(f"  Retrying in {delay}s...")
                    time.sleep(delay)
                else:
                    print(f"Error loading model: {e}")
                    raise

        self.model.eval()

        actual_dtype = getattr(self.model, "dtype", None)
        print(f"Requested torch_dtype: {requested_dtype}")
        if actual_dtype is not None:
            print(f"Model dtype in use: {actual_dtype}")
            if requested_torch_dtype is None:
                print("Model dtype was auto-selected.")
            elif requested_torch_dtype is not None and actual_dtype == requested_torch_dtype:
                print("Model dtype matches requested torch_dtype.")
            else:
                print("Model dtype differs from requested torch_dtype.")
        else:
            print("Model dtype could not be determined.")

        self._initialized = True
        print(f"Model loaded successfully on {self.device}")
        print(f"Model dtype: {self.model.dtype}")

    def format_harmony_prompt(self, text: str, reasoning_effort: str = "high") -> str:
        """DEPRECATED: Format a prompt using OpenAI Harmony formatting for gpt-oss models.

        Args:
            text: The user's question/prompt text
            reasoning_effort: Reasoning level (low, medium, high)

        Returns:
            Harmony-formatted prompt with system message and user message
        """
        warnings.warn(
            "format_harmony_prompt() is deprecated. Use FormatterRegistry instead.",
            DeprecationWarning,
            stacklevel=2,
        )
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
            except Exception:
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
            "healthy": self.health_check(),
        }

    def run_inference(
        self,
        prompt: str,
        max_tokens: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        reasoning_params: Optional[Dict[str, Any]] = None,
        enable_streaming: bool = False,  # Disabled by default to avoid hangs
        **kwargs,
    ) -> Dict[str, Any]:
        """Run inference on a single prompt.

        Args:
            prompt: Input text prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            reasoning_params: Optional reasoning parameters for thinking models
            enable_streaming: Enable streaming for TTFT tracking (default: True)
            **kwargs: Additional generation parameters

        Returns:
            Dict with response text and metadata (includes time_to_first_token if streaming enabled)
        """
        # Initialize model if needed
        if not self._initialized:
            self._initialize_model()

        start_time = time.time()
        ttft: Optional[float] = None

        try:
            import torch

            # Check if tokenizer supports chat template
            has_chat_template = (
                hasattr(self.tokenizer, "chat_template")
                and self.tokenizer.chat_template is not None
            )

            # Determine if we should use chat template
            # Special handling for different model types:
            # - Qwen/Hunyuan: Use chat template with enable_thinking parameter
            # - gpt-oss: Use HarmonyFormatter (self.formatter is set)
            # - Others with chat template but no formatter: Use chat template
            is_qwen = "qwen" in self.model_name.lower()
            is_hunyuan = "hunyuan" in self.model_name.lower()

            # Use chat template if:
            # 1. Model has chat template AND
            # 2. (No formatter assigned OR model is Qwen/Hunyuan which need chat template)
            use_chat_template = (
                has_chat_template
                and not self._legacy_use_harmony
                and (not self.formatter or is_qwen or is_hunyuan)
            )

            if use_chat_template:
                # Use chat template for models that require it (e.g., Qwen, Hunyuan, Llama-3, etc.)
                messages = [{"role": "user", "content": prompt}]

                # Add model-specific chat template parameters
                chat_kwargs = {"add_generation_prompt": True}

                # Handle Qwen thinking mode
                if "qwen" in self.model_name.lower():
                    # For Qwen models, enable_thinking is passed to chat template
                    # Default depends on model type:
                    # - Qwen3 mixed models: True by default
                    # - Thinking-only models: always enabled (parameter ignored)
                    enable_thinking = True  # Default
                    if reasoning_params:
                        # Check for explicit thinking control
                        if "enable_thinking" in reasoning_params:
                            # Handle string "true"/"false" from config
                            val = reasoning_params["enable_thinking"]
                            enable_thinking = (
                                val if isinstance(val, bool) else str(val).lower() == "true"
                            )
                        elif "reasoning" in reasoning_params:
                            val = reasoning_params["reasoning"]
                            enable_thinking = (
                                val if isinstance(val, bool) else str(val).lower() == "true"
                            )
                    chat_kwargs["enable_thinking"] = enable_thinking
                    print(
                        f"  Using Qwen chat template (thinking={'enabled' if enable_thinking else 'disabled'})"
                    )
                # Handle Hunyuan thinking mode
                elif "hunyuan" in self.model_name.lower():
                    # For Hunyuan models, enable_thinking controls CoT reasoning
                    enable_thinking = True  # Default
                    if reasoning_params:
                        # Check for explicit thinking control
                        if "enable_thinking" in reasoning_params:
                            enable_thinking = reasoning_params["enable_thinking"]
                        elif reasoning_params.get("reasoning") is False:
                            enable_thinking = False
                    chat_kwargs["enable_thinking"] = enable_thinking
                    print(
                        f"  Using Hunyuan chat template (thinking={'enabled' if enable_thinking else 'disabled'})"
                    )
                else:
                    print(f"  Using chat template for {self.model_name}")

                # Apply chat template and tokenize
                try:
                    print(f"  Applying chat template with kwargs: {chat_kwargs}")
                    tokenized_inputs = self.tokenizer.apply_chat_template(
                        messages, tokenize=True, return_tensors="pt", **chat_kwargs
                    )
                    # apply_chat_template returns input_ids directly, wrap in dict
                    if not isinstance(tokenized_inputs, dict):
                        tokenized_inputs = {"input_ids": tokenized_inputs}
                    print(
                        f"  Chat template applied successfully, input shape: {tokenized_inputs['input_ids'].shape}"
                    )
                except TypeError as e:
                    # Fallback if model doesn't support certain kwargs
                    print(f"  Chat template warning: {e}, retrying without extra params")
                    tokenized_inputs = self.tokenizer.apply_chat_template(
                        messages, tokenize=True, return_tensors="pt", add_generation_prompt=True
                    )
                    if not isinstance(tokenized_inputs, dict):
                        tokenized_inputs = {"input_ids": tokenized_inputs}
                    print(
                        f"  Fallback chat template applied, input shape: {tokenized_inputs['input_ids'].shape}"
                    )
            else:
                # Format prompt using registry formatter (new approach)
                if self.formatter:
                    # Use new formatter from registry
                    prompt = self.formatter.format_prompt(prompt, reasoning_params)
                elif self._legacy_use_harmony:
                    # DEPRECATED: Legacy Harmony formatting
                    reasoning_effort = "high"  # Default
                    if reasoning_params and "reasoning_effort" in reasoning_params:
                        reasoning_effort = reasoning_params["reasoning_effort"]
                    prompt = self.format_harmony_prompt(prompt, reasoning_effort)
                    print(f"  Using Harmony format with {reasoning_effort} reasoning")
                    print(f"  Prompt preview (first 200 chars): {prompt[:200]}...")
                elif reasoning_params and "reasoning_effort" in reasoning_params:
                    # Legacy: simple reasoning prefix (deprecated)
                    effort = reasoning_params["reasoning_effort"]
                    use_prompt_based = reasoning_params.get("use_prompt_based_reasoning", False)

                    if use_prompt_based:
                        # Old format (kept for backward compatibility)
                        prompt = f"Reasoning:{effort}\n\n{prompt}"
                        print(f"  Using legacy prompt-based reasoning ({effort} effort)")
                        print(f"  Prompt preview: {prompt}")

                # Tokenize input (standard path)
                tokenized_inputs = self.tokenizer(
                    prompt, return_tensors="pt", padding=True, truncation=True
                )

            inputs: Dict[str, Any] = dict(tokenized_inputs)

            if self.device == "cuda":
                inputs = {k: v.cuda() for k, v in inputs.items()}

            # Filter out token_type_ids if present (some models like Hunyuan don't support it)
            if "token_type_ids" in inputs:
                inputs.pop("token_type_ids")

            prompt_tokens = inputs["input_ids"].shape[1]

            # Build generation kwargs
            gen_kwargs = {
                "max_new_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "do_sample": temperature > 0,
                "pad_token_id": self.tokenizer.pad_token_id,
            }

            # Add formatter-provided generation parameters
            if self.formatter:
                extra_gen_params = self.formatter.get_generation_params(reasoning_params)
                gen_kwargs.update(extra_gen_params)

            # Add reasoning parameters if provided (legacy path)
            if reasoning_params and not self.formatter:
                # Handle different reasoning parameter formats
                if "reasoning_effort" in reasoning_params:
                    effort = reasoning_params["reasoning_effort"]
                    use_prompt_based = reasoning_params.get("use_prompt_based_reasoning", False)

                    # Only apply parameter mapping if NOT using prompt-based approach
                    if not use_prompt_based:
                        print(f"Using reasoning effort: {effort}")

                        # Map reasoning effort to actual generation parameters
                        # This simulates OpenAI-style reasoning by adjusting sampling parameters
                        # Note: max_new_tokens is NOT modified - it stays at user-configured value
                        if effort == "low":
                            # Fast, concise generation
                            gen_kwargs["temperature"] = 0.9
                            gen_kwargs["do_sample"] = True
                        elif effort == "medium":
                            # Balanced generation
                            gen_kwargs["temperature"] = 0.7
                            gen_kwargs["top_p"] = 0.9
                            gen_kwargs["do_sample"] = True
                        elif effort == "high":
                            # Thorough, detailed generation
                            gen_kwargs["temperature"] = 0.5
                            gen_kwargs["top_p"] = 0.95
                            gen_kwargs["top_k"] = 50
                            gen_kwargs["do_sample"] = True

                    # Note: Don't pass 'reasoning_effort' or 'use_prompt_based_reasoning' to model.generate()
                    # We've already used them above

                # Pass through other reasoning parameters (like thinking_budget for DeepSeek-R1)
                for key, value in reasoning_params.items():
                    if key not in gen_kwargs and key not in [
                        "reasoning_effort",
                        "use_prompt_based_reasoning",
                    ]:
                        gen_kwargs[key] = value

            # Merge additional kwargs
            gen_kwargs.update(kwargs)

            # Generate with streaming for TTFT tracking
            generated_text: str
            completion_text: str
            completion_tokens: int
            total_tokens: int

            if enable_streaming:
                # Use TextIteratorStreamer for accurate TTFT tracking
                try:
                    from threading import Thread

                    from transformers import TextIteratorStreamer

                    # Try streaming generation with reasoning parameters
                    streamer = TextIteratorStreamer(
                        self.tokenizer, skip_prompt=True, skip_special_tokens=True
                    )
                    generation_kwargs = {**inputs, **gen_kwargs, "streamer": streamer}

                    # Wrapper to capture exceptions in thread
                    thread_exception = None

                    def generation_target():
                        nonlocal thread_exception
                        try:
                            self.model.generate(**generation_kwargs)
                        except Exception as e:
                            thread_exception = e

                    # Start generation in a separate thread
                    thread = Thread(target=generation_target)
                    thread.start()

                    # Collect generated tokens and measure TTFT
                    generated_tokens = []
                    first_token = True

                    for new_text in streamer:
                        if first_token and new_text:
                            ttft = time.time() - start_time
                            first_token = False
                        generated_tokens.append(new_text)

                    thread.join()

                    # Check if thread raised an exception
                    if thread_exception is not None:
                        error_msg = str(thread_exception)
                        # Check if error is about reasoning parameters
                        if isinstance(thread_exception, (TypeError, ValueError)) and (
                            "model_kwargs" in error_msg
                            or "unexpected keyword argument" in error_msg
                            or "not used by the model" in error_msg
                        ):
                            # Model doesn't support reasoning parameters, retry without them
                            print(
                                "  Note: Model doesn't support reasoning parameters, retrying without them"
                            )

                            # Remove known reasoning-related parameters
                            reasoning_keys = [
                                "reasoning_effort",
                                "thinking_budget",
                                "cot_depth",
                                "use_prompt_based_reasoning",
                                "enable_thinking",
                                "reasoning",
                            ]
                            filtered_kwargs = {
                                k: v for k, v in gen_kwargs.items() if k not in reasoning_keys
                            }

                            # Retry with filtered parameters
                            streamer = TextIteratorStreamer(
                                self.tokenizer, skip_prompt=True, skip_special_tokens=True
                            )
                            generation_kwargs = {**inputs, **filtered_kwargs, "streamer": streamer}
                            thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
                            thread.start()

                            generated_tokens = []
                            first_token = True
                            for new_text in streamer:
                                if first_token and new_text:
                                    ttft = time.time() - start_time
                                    first_token = False
                                generated_tokens.append(new_text)
                            thread.join()
                        else:
                            # Different error, re-raise
                            raise thread_exception

                    completion_text = "".join(generated_tokens)

                    # Get the full output for token counting
                    # Re-run without streamer to get output tensor (needed for accurate token counts)
                    with torch.no_grad():
                        try:
                            outputs = self.model.generate(**inputs, **gen_kwargs)
                        except (TypeError, ValueError) as e:
                            error_msg = str(e)
                            # Check if error is about unused model_kwargs (model doesn't support reasoning params)
                            if (
                                "model_kwargs" in error_msg
                                or "unexpected keyword argument" in error_msg
                                or "not used by the model" in error_msg
                            ):
                                # Model doesn't support reasoning parameters, retry without them
                                if not (
                                    reasoning_params
                                    and reasoning_params.get("use_prompt_based_reasoning")
                                ):
                                    print(
                                        "  Note: Model doesn't support reasoning parameters, running without them"
                                    )

                                # Remove known reasoning-related parameters
                                reasoning_keys = [
                                    "reasoning_effort",
                                    "thinking_budget",
                                    "cot_depth",
                                    "use_prompt_based_reasoning",
                                    "enable_thinking",
                                    "reasoning",
                                ]
                                filtered_kwargs = {
                                    k: v for k, v in gen_kwargs.items() if k not in reasoning_keys
                                }

                                outputs = self.model.generate(**inputs, **filtered_kwargs)
                            else:
                                # Different error, re-raise
                                raise

                    generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                    completion_tokens = outputs.shape[1] - prompt_tokens
                    total_tokens = outputs.shape[1]

                except (ImportError, Exception) as e:
                    # Fallback to non-streaming if TextIteratorStreamer is not available
                    print(
                        f"  Streaming not available ({type(e).__name__}), using standard generation"
                    )
                    enable_streaming = False

            if not enable_streaming:
                # Standard non-streaming generation
                with torch.no_grad():
                    try:
                        outputs = self.model.generate(**inputs, **gen_kwargs)
                    except (TypeError, ValueError) as e:
                        error_msg = str(e)
                        # Check if error is about unused model_kwargs (model doesn't support reasoning params)
                        if (
                            "model_kwargs" in error_msg
                            or "unexpected keyword argument" in error_msg
                            or "not used by the model" in error_msg
                        ):
                            # Model doesn't support reasoning parameters, retry without them
                            # Note: We don't print this message when using prompt-based reasoning
                            # because the reasoning is in the prompt, not the parameters
                            if not (
                                reasoning_params
                                and reasoning_params.get("use_prompt_based_reasoning")
                            ):
                                print(
                                    "  Note: Model doesn't support reasoning parameters, running without them"
                                )

                            # Remove known reasoning-related parameters
                            reasoning_keys = [
                                "reasoning_effort",
                                "thinking_budget",
                                "cot_depth",
                                "use_prompt_based_reasoning",
                                "enable_thinking",
                                "reasoning",
                            ]
                            filtered_kwargs = {
                                k: v for k, v in gen_kwargs.items() if k not in reasoning_keys
                            }

                            outputs = self.model.generate(**inputs, **filtered_kwargs)
                        else:
                            # Different error, re-raise
                            raise

                # Decode output
                generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                completion_tokens = outputs.shape[1] - prompt_tokens
                total_tokens = outputs.shape[1]

            # Extract only the generated portion (remove prompt) for non-streaming path
            if not enable_streaming or completion_text is None:
                prompt_text = self.tokenizer.decode(
                    inputs["input_ids"][0], skip_special_tokens=True
                )
                completion_text = generated_text[len(prompt_text) :].strip()

            end_time = time.time()

            # For non-streaming, estimate TTFT based on generation time and token count
            # This is an approximation: total_time / total_tokens ≈ time_per_token
            if ttft is None and completion_tokens > 0:
                generation_time = end_time - start_time
                # Estimate TTFT as the time to generate first token
                # Assume roughly linear generation rate
                ttft = generation_time / completion_tokens
                print(
                    f"  Estimated TTFT: {ttft:.4f}s (non-streaming mode, {completion_tokens} tokens)"
                )

            # Debug: log token generation stats when reasoning is enabled
            if reasoning_params:
                effort = reasoning_params.get("reasoning_effort", "unknown")
                print(f"    Generated {completion_tokens} tokens ({effort} effort)")
                print(f"    Full text: {generated_text}, Total tokens: {total_tokens}")

            # Detect and warn about zero-token generation
            if completion_tokens == 0:
                warning_msg = (
                    f"WARNING: Model generated 0 tokens. "
                    f"This may indicate:\n"
                    f"  - Incorrect prompt formatting (chat template issue)\n"
                    f"  - Model immediately generated EOS token\n"
                    f"  - Input was truncated to empty\n"
                    f"  Prompt preview: {prompt_text[:200]}...\n"
                    f"  Generated text: '{generated_text}'"
                )
                print(warning_msg)

                # Return as failure with detailed error message
                return {
                    "text": completion_text,
                    "full_text": generated_text,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": total_tokens,
                    "latency_seconds": end_time - start_time,
                    "time_to_first_token": ttft,
                    "model": self.model_name,
                    "success": False,  # Mark as failure
                    "error": "Zero tokens generated - likely chat template or formatting issue",
                }

            result = {
                "text": completion_text,
                "full_text": generated_text,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
                "latency_seconds": end_time - start_time,
                "time_to_first_token": ttft,
                "model": self.model_name,
                "success": True,
                "error": None,
            }
            return result

        except Exception as e:
            end_time = time.time()
            error_msg = str(e)

            # Log detailed error information
            print(f"ERROR during inference: {error_msg}")
            print(f"  Exception type: {type(e).__name__}")
            if hasattr(e, "__traceback__"):
                import traceback

                print(f"  Traceback: {traceback.format_exc()}")

            return {
                "text": "",
                "success": False,
                "error": error_msg,
                "latency_seconds": end_time - start_time,
                "time_to_first_token": None,
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
