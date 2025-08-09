import traceback
from pathlib import Path
from typing import Any, List, Tuple

from exllamav3 import Cache, Config, Generator, Model, Tokenizer
from exllamav3.cache import CacheLayer_fp16, CacheLayer_quant

from extensions.openai.image_utils import (
    convert_image_attachments_to_pil,
    convert_openai_messages_to_images
)
from modules import shared
from modules.logging_colors import logger

try:
    import flash_attn
except Exception:
    logger.warning('Failed to load flash-attention due to the following error:\n')
    traceback.print_exc()


class Exllamav3Model:
    def __init__(self):
        pass

    @classmethod
    def from_pretrained(cls, path_to_model):
        path_to_model = Path(f'{shared.args.model_dir}') / Path(path_to_model)

        # Reset global MMTokenAllocator to prevent token ID corruption when switching models
        from exllamav3.tokenizer.mm_embedding import (
            FIRST_MM_EMBEDDING_INDEX,
            global_allocator
        )
        global_allocator.next_token_index = FIRST_MM_EMBEDDING_INDEX
        logger.info("Reset MMTokenAllocator for clean multimodal token allocation")

        config = Config.from_directory(str(path_to_model))
        model = Model.from_config(config)

        # Calculate the closest multiple of 256 at or above the chosen value
        max_tokens = shared.args.ctx_size
        if max_tokens % 256 != 0:
            adjusted_tokens = ((max_tokens // 256) + 1) * 256
            logger.warning(f"max_num_tokens must be a multiple of 256. Adjusting from {max_tokens} to {adjusted_tokens}")
            max_tokens = adjusted_tokens

        # Parse cache type (ExLlamaV2 pattern)
        cache_type = shared.args.cache_type.lower()
        cache_kwargs = {}
        if cache_type == 'fp16':
            layer_type = CacheLayer_fp16
        elif cache_type.startswith('q'):
            layer_type = CacheLayer_quant
            if '_' in cache_type:
                # Different bits for k and v (e.g., q4_q8)
                k_part, v_part = cache_type.split('_')
                k_bits = int(k_part[1:])
                v_bits = int(v_part[1:])
            else:
                # Same bits for k and v (e.g., q4)
                k_bits = v_bits = int(cache_type[1:])

            # Validate bit ranges
            if not (2 <= k_bits <= 8 and 2 <= v_bits <= 8):
                logger.warning(f"Invalid quantization bits: k_bits={k_bits}, v_bits={v_bits}. Must be between 2 and 8. Falling back to fp16.")
                layer_type = CacheLayer_fp16
            else:
                cache_kwargs = {'k_bits': k_bits, 'v_bits': v_bits}
        else:
            logger.warning(f"Unrecognized cache type: {cache_type}. Falling back to fp16.")
            layer_type = CacheLayer_fp16

        cache = Cache(model, max_num_tokens=max_tokens, layer_type=layer_type, **cache_kwargs)

        load_params = {'progressbar': True}
        if shared.args.gpu_split:
            split = [float(alloc) for alloc in shared.args.gpu_split.split(",")]
            load_params['use_per_device'] = split

        model.load(**load_params)

        tokenizer = Tokenizer.from_config(config)

        # Load vision model component (ExLlamaV3 native)
        vision_model = None
        try:
            logger.info("Loading vision model component...")
            vision_model = Model.from_config(config, component="vision")
            vision_model.load(progressbar=True)
            logger.info("Vision model loaded successfully")
        except Exception as e:
            logger.warning(f"Vision model loading failed (multimodal disabled): {e}")

        generator = Generator(
            model=model,
            cache=cache,
            tokenizer=tokenizer,
        )

        result = cls()
        result.model = model
        result.cache = cache
        result.tokenizer = tokenizer
        result.generator = generator
        result.config = config
        result.max_tokens = max_tokens
        result.vision_model = vision_model

        return result

    def is_multimodal(self) -> bool:
        """Check if this model supports multimodal input."""
        return hasattr(self, 'vision_model') and self.vision_model is not None

    def _process_images_for_generation(self, prompt: str, state: dict) -> Tuple[str, List[Any]]:
        """
        Process all possible image inputs and return modified prompt + embeddings.
        Returns: (processed_prompt, image_embeddings)
        """
        if not self.is_multimodal():
            return prompt, []

        # Collect images from various sources using shared utilities
        pil_images = []

        # From webui image_attachments (preferred format)
        if 'image_attachments' in state and state['image_attachments']:
            pil_images.extend(convert_image_attachments_to_pil(state['image_attachments']))

        # From OpenAI API raw_images
        elif 'raw_images' in state and state['raw_images']:
            pil_images.extend(state['raw_images'])

        # From OpenAI API messages format
        elif 'messages' in state and state['messages']:
            pil_images.extend(convert_openai_messages_to_images(state['messages']))

        if not pil_images:
            return prompt, []

        # ExLlamaV3-specific: Generate embeddings
        try:
            # Use pre-computed embeddings if available (proper MMEmbedding lifetime)
            if 'image_embeddings' in state and state['image_embeddings']:
                # Use existing embeddings - this preserves MMEmbedding lifetime
                image_embeddings = state['image_embeddings']
            else:
                # Do not reset the cache/allocator index; it causes token ID conflicts during generation.

                logger.info(f"Processing {len(pil_images)} image(s) with ExLlamaV3 vision model")
                image_embeddings = [
                    self.vision_model.get_image_embeddings(tokenizer=self.tokenizer, image=img)
                    for img in pil_images
                ]

            # ExLlamaV3-specific: Handle prompt processing with placeholders
            placeholders = [ie.text_alias for ie in image_embeddings]

            if '<__media__>' in prompt:
                # Web chat: Replace <__media__> placeholders
                for alias in placeholders:
                    prompt = prompt.replace('<__media__>', alias, 1)
                logger.info(f"Replaced {len(placeholders)} <__media__> placeholder(s)")
            else:
                # API: Prepend embedding aliases
                combined_placeholders = "\n".join(placeholders)
                prompt = combined_placeholders + "\n" + prompt
                logger.info(f"Prepended {len(placeholders)} embedding(s) to prompt")

            return prompt, image_embeddings

        except Exception as e:
            logger.error(f"Failed to process images: {e}")
            return prompt, []

    def generate_with_streaming(self, prompt, state):
        """
        Generate text with streaming using native ExLlamaV3 API
        """
        from exllamav3 import Job
        from exllamav3.generator.sampler.presets import ComboSampler

        # Process images and modify prompt (ExLlamaV3-specific)
        prompt, image_embeddings = self._process_images_for_generation(prompt, state)

        sampler = ComboSampler(
            rep_p=state.get('repetition_penalty', 1.0),
            freq_p=state.get('frequency_penalty', 0.0),
            pres_p=state.get('presence_penalty', 0.0),
            temperature=state.get('temperature', 0.7),
            min_p=state.get('min_p', 0.0),
            top_k=state.get('top_k', 0),
            top_p=state.get('top_p', 1.0),
        )

        # Encode prompt with embeddings (ExLlamaV3-specific)
        if image_embeddings:
            input_ids = self.tokenizer.encode(
                prompt,
                encode_special_tokens=True,
                embeddings=image_embeddings,
            )
        else:
            input_ids = self.tokenizer.encode(prompt, encode_special_tokens=True)

        # Get stop conditions from state (webui format) - keep as strings like ExLlamaV3 examples
        stop_conditions = []
        if 'stopping_strings' in state and state['stopping_strings']:
            # Use strings directly (ExLlamaV3 handles the conversion internally)
            stop_conditions.extend(state['stopping_strings'])

        # Add EOS token ID as ExLlamaV3 examples do
        if hasattr(self.tokenizer, 'eos_token_id') and self.tokenizer.eos_token_id is not None:
            stop_conditions.append(self.tokenizer.eos_token_id)

        job = Job(
            input_ids=input_ids,
            max_new_tokens=state.get('max_new_tokens', 500),
            decode_special_tokens=True,
            embeddings=image_embeddings if image_embeddings else None,
            sampler=sampler,
            stop_conditions=stop_conditions if stop_conditions else None,
        )

        # Stream generation
        self.generator.enqueue(job)

        response_text = ""
        try:
            while self.generator.num_remaining_jobs():
                results = self.generator.iterate()
                for result in results:
                    if "eos" in result and result["eos"]:
                        break

                    chunk = result.get("text", "")
                    if chunk:
                        response_text += chunk
                        yield response_text
        finally:
            # No cleanup needed. MMEmbedding lifetime is managed by Python.
            # Cache and page table resets are unnecessary and can cause token ID conflicts.
            pass

    def generate(self, prompt, state):
        """
        Generate text using native ExLlamaV3 API (non-streaming)
        """
        output = self.generator.generate(
            prompt=prompt,
            max_new_tokens=state.get('max_new_tokens', 500),
            temperature=state.get('temperature', 0.7),
            top_p=state.get('top_p', 1.0),
            top_k=state.get('top_k', 0),
            repetition_penalty=state.get('repetition_penalty', 1.0),
            frequency_penalty=state.get('frequency_penalty', 0.0),
            presence_penalty=state.get('presence_penalty', 0.0),
            min_p=state.get('min_p', 0.0),
        )

        return output

    def encode(self, string, **kwargs):
        return self.tokenizer.encode(string, **kwargs)

    def decode(self, ids, **kwargs):
        return self.tokenizer.decode(ids, **kwargs)

    @property
    def last_prompt_token_count(self):
        # This would need to be tracked during generation
        return 0

    def unload(self):
        logger.info("Unloading ExLlamaV3 model components...")

        if hasattr(self, 'vision_model') and self.vision_model is not None:
            try:
                del self.vision_model
            except Exception as e:
                logger.warning(f"Error unloading vision model: {e}")
            self.vision_model = None

        if hasattr(self, 'model') and self.model is not None:
            try:
                self.model.unload()
                del self.model
            except Exception as e:
                logger.warning(f"Error unloading main model: {e}")
            self.model = None

        if hasattr(self, 'cache') and self.cache is not None:
            self.cache = None

        if hasattr(self, 'generator') and self.generator is not None:
            self.generator = None

        if hasattr(self, 'tokenizer') and self.tokenizer is not None:
            self.tokenizer = None

        # Force GPU memory cleanup
        import gc

        import torch
        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

        logger.info("ExLlamaV3 model fully unloaded")
