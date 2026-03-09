import platform
import traceback

import modules.shared as shared
from modules.logging_colors import logger

# Constants for MLX configuration
MLX_TOP_P_DISABLED = 0.0  # MLX expects 0.0 to disable top_p
DEFAULT_MAX_TOKENS = 512  # Default maximum tokens for generation


def is_apple_silicon():
    """Check if running on Apple Silicon"""
    return platform.system() == "Darwin" and platform.machine() == "arm64"


class MLXModel:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.model_name = None

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
                top_p=top_p if top_p < 1.0 else MLX_TOP_P_DISABLED,
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
        """Create logits processors for repetition penalty, etc."""
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

        # Seed handling
        seed = state.get('seed', -1)
        if seed != -1:
            try:
                import mlx.core as mx
                mx.random.seed(seed)
            except Exception as e:
                logger.warning(f"Failed to set MLX random seed: {e}")

        return mlx_params

    def _prepare_prompt(self, prompt):
        """Prepare prompt with chat template if available"""
        if self.tokenizer.chat_template is not None:
            messages = [{"role": "user", "content": prompt}]
            formatted_prompt = self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False
            )
            return formatted_prompt
        return prompt

    @staticmethod
    def _extract_token_id(token):
        """Extract integer token ID from MLX array or direct integer."""
        if hasattr(token, 'item'):
            return int(token.item())
        return int(token)

    def generate(self, prompt, state):
        """Non-streaming generation with advanced sampling"""
        try:
            from mlx_lm.generate import generate_step
            import mlx.core as mx
        except ImportError:
            logger.error("mlx-lm not found. Please install with: pip install mlx-lm")
            return ""

        if self.model is None or self.tokenizer is None:
            logger.error("MLX model not loaded")
            return ""

        try:
            formatted_prompt = self._prepare_prompt(prompt)
            prompt_tokens = self.tokenizer.encode(formatted_prompt)
            prompt_array = mx.array(prompt_tokens)

            mlx_params = self._map_parameters(state)
            max_tokens = mlx_params.pop('max_tokens', DEFAULT_MAX_TOKENS)

            generated_tokens = []

            for token, logprobs in generate_step(
                prompt_array,
                self.model,
                max_tokens=max_tokens,
                **mlx_params
            ):
                token_id = self._extract_token_id(token)
                generated_tokens.append(token_id)

                if self._is_stopped(state):
                    break

            if generated_tokens:
                return self.tokenizer.decode(generated_tokens)
            return ""

        except Exception as e:
            logger.error(f"MLX generation failed: {str(e)}")
            traceback.print_exc()
            return ""

    def generate_with_streaming(self, prompt, state):
        """True streaming generation using MLX generate_step.

        Uses full decode of accumulated tokens on each step to correctly
        handle multi-byte characters (e.g., CJK, emoji) that span multiple
        tokens.
        """
        try:
            from mlx_lm.generate import generate_step
            import mlx.core as mx
        except ImportError:
            logger.error("mlx-lm not found. Please install with: pip install mlx-lm")
            yield ""
            return

        if self.model is None or self.tokenizer is None:
            logger.error("MLX model not loaded")
            yield ""
            return

        try:
            formatted_prompt = self._prepare_prompt(prompt)
            prompt_tokens = self.tokenizer.encode(formatted_prompt)
            prompt_array = mx.array(prompt_tokens)

            mlx_params = self._map_parameters(state)
            max_tokens = mlx_params.pop('max_tokens', DEFAULT_MAX_TOKENS)

            generated_tokens = []
            prev_text = ""

            for token, logprobs in generate_step(
                prompt_array,
                self.model,
                max_tokens=max_tokens,
                **mlx_params
            ):
                token_id = self._extract_token_id(token)
                generated_tokens.append(token_id)

                # Decode all accumulated tokens to properly handle
                # multi-byte characters that span token boundaries
                current_text = self.tokenizer.decode(generated_tokens)

                # Only yield when new text is available (avoids yielding
                # on partial multi-byte sequences)
                if current_text != prev_text:
                    prev_text = current_text
                    yield current_text

                if self._is_stopped(state):
                    break

            # Final yield with complete text
            if generated_tokens:
                final_text = self.tokenizer.decode(generated_tokens)
                if final_text != prev_text:
                    yield final_text

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

    def decode(self, token_ids, **kwargs):
        """Decode tokens to text"""
        if self.tokenizer is None:
            return ""

        try:
            text = self.tokenizer.decode(token_ids)
            return text
        except Exception as e:
            logger.error(f"MLX detokenization failed: {str(e)}")
            return ""

    def unload(self):
        """Unload the model to free memory"""
        self.model = None
        self.tokenizer = None
        logger.info("MLX model unloaded")
