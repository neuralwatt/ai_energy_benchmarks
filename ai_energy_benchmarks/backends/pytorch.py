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
        max_memory: Optional[Dict] = None
    ):
        """Initialize PyTorch backend.

        Args:
            model: HuggingFace model name or path
            device: Device to use (cuda/cpu)
            device_ids: List of GPU device IDs
            torch_dtype: Torch dtype (auto/float16/bfloat16/float32)
            device_map: Device map strategy (auto/balanced/sequential)
            max_memory: Max memory per device
        """
        self.model_name = model
        self.device = device
        self.device_ids = device_ids or [0]
        self.torch_dtype = torch_dtype
        self.device_map = device_map
        self.max_memory = max_memory

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

            # Determine dtype
            if self.torch_dtype == "auto":
                dtype = torch.float16 if self.device == "cuda" else torch.float32
            elif self.torch_dtype == "float16":
                dtype = torch.float16
            elif self.torch_dtype == "bfloat16":
                dtype = torch.bfloat16
            else:
                dtype = torch.float32

            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=dtype,
                device_map=self.device_map,
                trust_remote_code=True,
                max_memory=self.max_memory
            )

            # Set to eval mode
            self.model.eval()

            self._initialized = True
            print(f"Model loaded successfully on {self.device}")

        except Exception as e:
            print(f"Error loading model: {e}")
            raise

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
        **kwargs
    ) -> Dict[str, Any]:
        """Run inference on a single prompt.

        Args:
            prompt: Input text prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
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

            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    do_sample=temperature > 0,
                    pad_token_id=self.tokenizer.pad_token_id,
                    **kwargs
                )

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
