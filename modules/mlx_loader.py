import platform
import traceback

import modules.shared as shared
from modules.logging_colors import logger

# Constants for MLX configuration
DEFAULT_MAX_TOKENS = 512  # Default maximum tokens for generation

# Mapping from webui cache_type values to mlx-lm kv_bits
CACHE_TYPE_TO_KV_BITS = {
    'q2': 2,
    'q3': 3,
    'q4_0': 4,
    'q4': 4,
    'q5': 5,
    'q6': 6,
    'q7': 7,
    'q8_0': 8,
    'q8': 8,
}


def is_apple_silicon():
    """Check if running on Apple Silicon"""
    return platform.system() == "Darwin" and platform.machine() == "arm64"


class MLXModel:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.model_name = None
        self.last_prompt_token_count = 0

    @classmethod
    def from_pretrained(cls, model_name):
        """Load MLX model from path or HuggingFace repository"""

        if not is_apple_silicon():
            logger.warning("MLX backend is only supported on Apple Silicon. Falling back to Transformers.")
            return None

        try:
            from mlx_lm import load
        except ImportError:
            logger.error("mlx-lm not found. Please install with: pip install mlx-lm")
            return None

        instance = cls()
        instance.model_name = model_name

        try:
            # Determine the model path/name
            model_path = cls._resolve_model_path(model_name)

            logger.info(f"Loading MLX model: {model_path}")
            tokenizer_config = {"trust_remote_code": True}
            model, tokenizer = load(model_path, tokenizer_config=tokenizer_config)

            instance.model = model
            instance.tokenizer = tokenizer

            logger.info(f"Successfully loaded MLX model: {model_name}")
            return instance

        except Exception as e:
            error_msg = str(e)
            if "not supported" in error_msg.lower():
                logger.error(f"MLX model {model_name} uses an unsupported model type: {error_msg}")
                logger.error("Consider using a different loader or updating mlx-lm to a newer version")
            else:
                logger.error(f"Failed to load MLX model {model_name}: {error_msg}")
                traceback.print_exc()
            return None

    @staticmethod
    def _resolve_model_path(model_name):
        """Resolve model path using the shared resolve_model_path utility,
        with additional MLX-specific HuggingFace repo name handling."""
        from modules.utils import resolve_model_path

        model_path = resolve_model_path(model_name)

        if model_path.exists():
            return str(model_path)
        elif '/' in model_name:
            # Already has repo/model format (e.g., "mlx-community/Llama-3.2-3B-Instruct-4bit")
            return model_name
        elif '_' in model_name and not model_name.startswith('_'):
            # Handle repo_name format - convert first underscore to slash
            # e.g., "mlx-community_model-name" -> "mlx-community/model-name"
            parts = model_name.split('_', 1)
            if len(parts) == 2:
                return f"{parts[0]}/{parts[1]}"
            return model_name
        else:
            # Default to mlx-community for standalone model names
            return f"mlx-community/{model_name}"

    def _is_stopped(self, state):
        """Check if generation should stop (supports both global and per-request stop)."""
        if shared.stop_everything:
            return True
        stop_event = state.get('stop_event') if isinstance(state, dict) else None
        if stop_event is not None and stop_event.is_set():
            return True
        return False

    def _create_mlx_sampler(self, state):
        """Create MLX sampler with webui parameters"""
        try:
            from mlx_lm.sample_utils import make_sampler

            # Extract sampling parameters from state
            temperature = state.get('temperature', 1.0)
            top_p = state.get('top_p', 1.0)
            top_k = state.get('top_k', 0)  # 0 means no top_k limit
            min_p = state.get('min_p', 0.0)

            # Handle dynamic temperature
            if state.get('dynamic_temperature', False):
                temp_low = state.get('dynatemp_low', 1.0)
                temp_high = state.get('dynatemp_high', 1.0)
                temperature = (temp_low + temp_high) / 2

            # XTC sampling parameters
            xtc_probability = state.get('xtc_probability', 0.0)
            xtc_threshold = state.get('xtc_threshold', 0.1)

            # Ensure temperature is not zero (causes issues with MLX)
            if temperature <= 0.0:
                temperature = 0.01

            # Log sampling parameters for debugging
            if shared.args.verbose:
                logger.info(f"MLX Sampler - temp: {temperature}, top_p: {top_p}, top_k: {top_k}, min_p: {min_p}")

            # Create the sampler
            sampler = make_sampler(
                temp=temperature,
                top_p=top_p if top_p < 1.0 else 0.0,
                top_k=int(top_k) if top_k > 0 else 0,
                min_p=min_p,
                min_tokens_to_keep=1,
                xtc_probability=xtc_probability,
                xtc_threshold=xtc_threshold,
                xtc_special_tokens=[]
            )

            return sampler

        except ImportError:
            logger.warning("MLX sampling utilities not available, using default sampler")
            return None
        except Exception as e:
            logger.error(f"Failed to create MLX sampler: {e}")
            return None

    def _create_logits_processors(self, state):
        """Create logits processors for repetition penalty, ban_eos_token, etc."""
        processors = []

        try:
            from mlx_lm.sample_utils import make_repetition_penalty

            # Repetition penalty
            repetition_penalty = state.get('repetition_penalty', 1.0)
            if repetition_penalty != 1.0:
                context_size = state.get('repetition_penalty_range', 1024)
                rep_processor = make_repetition_penalty(
                    penalty=repetition_penalty,
                    context_size=context_size
                )
                processors.append(rep_processor)

        except ImportError:
            logger.warning("MLX repetition penalty not available")
        except Exception as e:
            logger.error(f"Failed to create repetition penalty processor: {e}")

        # Ban EOS token
        if state.get('ban_eos_token', False) and self.tokenizer is not None:
            eos_token_id = getattr(self.tokenizer, 'eos_token_id', None)
            if eos_token_id is not None:
                try:
                    import mlx.core as mx

                    def ban_eos_processor(tokens, logits):
                        logits[..., eos_token_id] = -float('inf')
                        return logits

                    processors.append(ban_eos_processor)
                except ImportError:
                    pass

        return processors if processors else None

    def _map_parameters(self, state):
        """Map text-generation-webui parameters to MLX parameters"""
        mlx_params = {}

        # Max tokens
        if 'max_new_tokens' in state and state['max_new_tokens'] > 0:
            mlx_params['max_tokens'] = state['max_new_tokens']
        else:
            mlx_params['max_tokens'] = DEFAULT_MAX_TOKENS

        # Create custom sampler with advanced parameters
        sampler = self._create_mlx_sampler(state)
        if sampler:
            mlx_params['sampler'] = sampler

        # Create logits processors
        logits_processors = self._create_logits_processors(state)
        if logits_processors:
            mlx_params['logits_processors'] = logits_processors

        # Context size -> max_kv_size
        ctx_size = getattr(shared.args, 'ctx_size', 0)
        if ctx_size > 0:
            mlx_params['max_kv_size'] = ctx_size

        # KV cache quantization from cache_type
        cache_type = getattr(shared.args, 'cache_type', 'fp16')
        kv_bits = CACHE_TYPE_TO_KV_BITS.get(cache_type)
        if kv_bits is not None:
            mlx_params['kv_bits'] = kv_bits

        # Seed handling
        seed = state.get('seed', -1)
        if seed != -1:
            try:
                import mlx.core as mx
                mx.random.seed(seed)
            except Exception as e:
                logger.warning(f"Failed to set MLX random seed: {e}")

        return mlx_params

    def generate(self, prompt, state):
        """Non-streaming generation using stream_generate"""
        try:
            from mlx_lm import stream_generate
        except ImportError:
            logger.error("mlx-lm not found. Please install with: pip install mlx-lm")
            return ""

        if self.model is None or self.tokenizer is None:
            logger.error("MLX model not loaded")
            return ""

        try:
            mlx_params = self._map_parameters(state)

            # Track prompt token count for stats
            prompt_tokens = self.tokenizer.encode(prompt)
            self.last_prompt_token_count = len(prompt_tokens)

            # Auto max new tokens
            if state.get('auto_max_new_tokens', False):
                mlx_params['max_tokens'] = state.get('truncation_length', 2048) - self.last_prompt_token_count

            result_text = ""
            for response in stream_generate(
                self.model,
                self.tokenizer,
                prompt=prompt,
                **mlx_params
            ):
                if self._is_stopped(state):
                    break
                result_text += response.text

            return result_text

        except Exception as e:
            logger.error(f"MLX generation failed: {str(e)}")
            traceback.print_exc()
            return ""

    def generate_with_streaming(self, prompt, state):
        """Streaming generation using stream_generate.

        Uses the high-level stream_generate API which handles multi-byte
        characters, prompt caching, and generation statistics natively.
        """
        try:
            from mlx_lm import stream_generate
        except ImportError:
            logger.error("mlx-lm not found. Please install with: pip install mlx-lm")
            yield ""
            return

        if self.model is None or self.tokenizer is None:
            logger.error("MLX model not loaded")
            yield ""
            return

        try:
            mlx_params = self._map_parameters(state)

            # Track prompt token count for stats
            prompt_tokens = self.tokenizer.encode(prompt)
            self.last_prompt_token_count = len(prompt_tokens)

            # Auto max new tokens
            if state.get('auto_max_new_tokens', False):
                mlx_params['max_tokens'] = state.get('truncation_length', 2048) - self.last_prompt_token_count

            cumulative_text = ""
            for response in stream_generate(
                self.model,
                self.tokenizer,
                prompt=prompt,
                **mlx_params
            ):
                if self._is_stopped(state):
                    break

                cumulative_text += response.text
                yield cumulative_text

        except Exception as e:
            logger.error(f"MLX streaming generation failed: {str(e)}")
            traceback.print_exc()
            yield ""

    def encode(self, text, add_bos_token=False, **kwargs):
        """Encode text to tokens"""
        if self.tokenizer is None:
            import torch
            return torch.tensor([[]], dtype=torch.long)

        try:
            tokens = self.tokenizer.encode(text)

            import torch
            tokens_tensor = torch.tensor([tokens], dtype=torch.long)
            return tokens_tensor
        except Exception as e:
            logger.error(f"MLX tokenization failed: {str(e)}")
            import torch
            return torch.tensor([[]], dtype=torch.long)

    def decode(self, token_ids, skip_special_tokens=True, **kwargs):
        """Decode tokens to text"""
        if self.tokenizer is None:
            return ""

        try:
            text = self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)
            return text
        except Exception as e:
            logger.error(f"MLX detokenization failed: {str(e)}")
            return ""

    def unload(self):
        """Unload the model to free memory"""
        self.model = None
        self.tokenizer = None

        try:
            import mlx.core as mx
            mx.metal.clear_cache()
        except Exception:
            pass

        logger.info("MLX model unloaded")
