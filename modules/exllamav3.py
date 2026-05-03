import math
import queue
import threading
from pathlib import Path
from typing import Any, List, Tuple

import torch

from exllamav3 import Cache, Config, Generator, Model, Tokenizer
from exllamav3.cache import CacheLayer_fp16, CacheLayer_quant
from exllamav3.generator import Job
from exllamav3.generator.filter import Filter
from exllamav3.generator.sampler import (
    CustomSampler,
    SS_AdaptiveP,
    SS_Argmax,
    SS_MinP,
    SS_PresFreqP,
    SS_RepP,
    SS_Sample,
    SS_Temperature,
    SS_TopK,
    SS_TopP
)
from modules import shared
from modules.image_utils import (
    convert_image_attachments_to_pil,
    convert_openai_messages_to_images
)
from modules.logging_colors import logger
from modules.text_generation import get_max_prompt_length

try:
    import flash_attn
except Exception:
    logger.warning('Failed to load flash-attention due to the following error:', exc_info=True)


class LogitBiasFilter(Filter):
    """Filter subclass that applies a static additive logit bias mask."""

    def __init__(self, tokenizer, logit_bias_dict):
        super().__init__(tokenizer=tokenizer, trigger_token=None, prefix_str=None, eos_after_completed=False)
        self.logit_bias_dict = logit_bias_dict
        self._mask = None

    def reset(self): pass
    def accept_token(self, token): pass
    def is_completed(self): return False
    def use_background_worker(self): return False

    def get_next_logit_mask(self):
        if self._mask is None:
            self._mask = torch.zeros((1, self.vocab_size), dtype=self.logits_dtype)
            for token_id_str, bias in self.logit_bias_dict.items():
                token_id = int(token_id_str)
                if 0 <= token_id < self.vocab_size:
                    self._mask[0, token_id] = bias
        return self._mask


class ConcurrentGenerator:
    def __init__(self, generator):
        self.generator = generator
        self.lock = threading.Lock()
        self.job_queues = {}
        self.active = True
        self.has_jobs = threading.Event()
        self.thread = threading.Thread(target=self._iterate_loop, daemon=True)
        self.thread.start()

    def _iterate_loop(self):
        while self.active:
            self.has_jobs.wait(timeout=0.5)
            with self.lock:
                if not self.job_queues:
                    self.has_jobs.clear()
                    continue
                try:
                    results = self.generator.iterate()
                except Exception:
                    logger.exception("Exception in ConcurrentGenerator iterate loop")
                    for q in self.job_queues.values():
                        q.put(None)
                    self.job_queues.clear()
                    self.generator.clear_queue()
                    self.has_jobs.clear()
                    continue
                for result in results:
                    job = result["job"]
                    q = self.job_queues.get(job)
                    if q:
                        q.put(result)
                        if result.get("eos"):
                            self.job_queues.pop(job, None)
                if not self.job_queues:
                    self.has_jobs.clear()

    def submit(self, job) -> queue.Queue:
        q = queue.Queue()
        with self.lock:
            self.job_queues[job] = q
            self.generator.enqueue(job)
        self.has_jobs.set()
        return q

    def cancel(self, job):
        with self.lock:
            if job in self.job_queues:
                self.generator.cancel(job)
                self.job_queues[job].put(None)
                del self.job_queues[job]

    def stop(self):
        self.active = False
        self.has_jobs.set()
        self.thread.join(timeout=5)


class Exllamav3Model:

    @property
    def device(self) -> torch.device:
        return torch.device(0)

    @classmethod
    def from_pretrained(cls, path_to_model):
        path_to_model = Path(f'{shared.args.model_dir}') / Path(path_to_model)

        # Reset global MMTokenAllocator to prevent token ID corruption when switching models
        from exllamav3.tokenizer.mm_embedding import (
            FIRST_MM_EMBEDDING_INDEX,
            global_allocator
        )
        global_allocator.next_token_index = FIRST_MM_EMBEDDING_INDEX

        config = Config.from_directory(str(path_to_model))
        model = Model.from_config(config)

        # Adjust to the closest multiple of 256 at or above the chosen value
        max_tokens = shared.args.ctx_size
        if max_tokens % 256 != 0:
            adjusted_tokens = ((max_tokens // 256) + 1) * 256
            logger.warning(f"max_num_tokens must be a multiple of 256. Adjusting from {max_tokens} to {adjusted_tokens}")
            max_tokens = adjusted_tokens

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
        split = None
        if shared.args.gpu_split:
            split = [float(alloc) for alloc in shared.args.gpu_split.split(",")]
            load_params['use_per_device'] = split

        # Tensor-parallelism
        if shared.args.enable_tp:
            load_params['tensor_p'] = True
            load_params['tp_backend'] = shared.args.tp_backend

        # Load vision and draft before the main model so autosplit
        # accounts for their VRAM usage.

        # Load vision model component (ExLlamaV3 native)
        vision_model = None
        if "vision_config" in config.config_dict:
            logger.info("Vision component detected in model config. Attempting to load...")
            try:
                vision_model = Model.from_config(config, component="vision")
                vision_model.load(progressbar=True)
                logger.info("Vision model loaded successfully.")
            except Exception as e:
                logger.warning(f"Vision model loading failed (multimodal disabled): {e}")
        else:
            logger.info("No vision component in model config. Skipping multimodal setup.")

        # Initialize draft model for speculative decoding
        draft_model = None
        draft_cache = None
        if shared.args.model_draft and shared.args.model_draft.lower() not in ["", "none"]:
            logger.info(f"Loading draft model for speculative decoding: {shared.args.model_draft}")

            draft_path = Path(shared.args.model_draft)
            if not draft_path.is_dir():
                draft_path = Path(f'{shared.args.model_dir}') / Path(shared.args.model_draft)

            if not draft_path.is_dir():
                logger.warning(f"Draft model not found at {draft_path}, speculative decoding disabled.")
            else:
                draft_config = Config.from_directory(str(draft_path))
                draft_model = Model.from_config(draft_config)
                draft_cache = Cache(draft_model, max_num_tokens=max_tokens, layer_type=layer_type, **cache_kwargs)

                draft_load_params = {'progressbar': True}
                if split:
                    draft_load_params['use_per_device'] = split

                draft_model.load(**draft_load_params)
                logger.info(f"Draft model loaded successfully. Max speculative tokens: {shared.args.draft_max}")

        # Load main model last
        model.load(**load_params)
        tokenizer = Tokenizer.from_config(config)

        generator = Generator(
            model=model,
            cache=cache,
            tokenizer=tokenizer,
            draft_model=draft_model,
            draft_cache=draft_cache,
            num_draft_tokens=shared.args.draft_max if draft_model is not None else 0,
        )

        result = cls()
        result.model = model
        result.cache = cache
        result.tokenizer = tokenizer
        result.generator = generator
        result.parallel_generator = ConcurrentGenerator(generator)
        result.config = config
        result.max_tokens = max_tokens
        result.vision_model = vision_model
        result.draft_model = draft_model
        result.draft_cache = draft_cache

        return result, result

    def is_multimodal(self) -> bool:
        """Check if this model supports multimodal input."""
        return hasattr(self, 'vision_model') and self.vision_model is not None

    def _process_images_for_generation(self, prompt: str, state: dict) -> Tuple[str, List[Any]]:
        """
        Process all possible image inputs and return modified prompt + embeddings.
        Returns: (processed_prompt, image_embeddings)
        """
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

        try:
            if 'image_embeddings' in state and state['image_embeddings']:
                image_embeddings = state['image_embeddings']
            else:
                # Do not reset the cache/allocator index; it causes token ID conflicts during generation.
                logger.info(f"Processing {len(pil_images)} image(s) with ExLlamaV3 vision model")
                image_embeddings = [
                    self.vision_model.get_image_embeddings(tokenizer=self.tokenizer, image=img)
                    for img in pil_images
                ]

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

        if shared.is_multimodal:
            # Process images and modify prompt (ExLlamaV3-specific)
            prompt, image_embeddings = self._process_images_for_generation(prompt, state)
        else:
            image_embeddings = []

        # Greedy decoding is a special case
        if state['temperature'] == 0:
            sampler = CustomSampler([SS_Argmax()])
        else:
            # 1. Collect active samplers (unordered)
            unordered_samplers = []

            penalty_range = state['repetition_penalty_range']
            if penalty_range <= 0:
                penalty_range = int(10e7)  # Use large number for "full context"
            rep_decay = 0  # Not a configurable parameter

            if state['repetition_penalty'] != 1.0:
                unordered_samplers.append(SS_RepP(state['repetition_penalty'], penalty_range, rep_decay))
            if state['presence_penalty'] != 0.0 or state['frequency_penalty'] != 0.0:
                unordered_samplers.append(SS_PresFreqP(state['presence_penalty'], state['frequency_penalty'], penalty_range, rep_decay))

            if state['top_k'] > 0:
                unordered_samplers.append(SS_TopK(state['top_k']))
            if state['top_p'] < 1.0:
                unordered_samplers.append(SS_TopP(state['top_p']))
            if state['min_p'] > 0.0:
                unordered_samplers.append(SS_MinP(state['min_p']))

            unordered_samplers.append(SS_Temperature(state['temperature']))

            # 2. Sort samplers by priority
            class_name_to_nickname = {
                'SS_RepP': 'repetition_penalty',
                'SS_PresFreqP': 'presence_frequency_penalty',
                'SS_TopK': 'top_k',
                'SS_TopP': 'top_p',
                'SS_MinP': 'min_p',
                'SS_Temperature': 'temperature',
            }

            default_priority = ['repetition_penalty', 'presence_frequency_penalty', 'top_k', 'top_p', 'min_p', 'temperature']
            sampler_priority = list(state.get('sampler_priority') or default_priority)

            if state['temperature_last'] and 'temperature' in sampler_priority:
                sampler_priority.append(sampler_priority.pop(sampler_priority.index('temperature')))

            # The preset system uses separate 'presence_penalty' and
            # 'frequency_penalty', but ExLlamaV3 has a single combined
            # SS_PresFreqP sampler. Normalize to the combined name.
            sampler_priority = ['presence_frequency_penalty' if x in ('presence_penalty', 'frequency_penalty') else x for x in sampler_priority]

            def custom_sort_key(sampler_obj):
                class_name = sampler_obj.__class__.__name__
                nickname = class_name_to_nickname.get(class_name)
                if nickname and nickname in sampler_priority:
                    return sampler_priority.index(nickname)
                return -1

            ordered_samplers = sorted(unordered_samplers, key=custom_sort_key)

            # 3. Add final sampling stage and build the sampler
            if state.get('adaptive_target', 0) > 0:
                ordered_samplers.append(SS_AdaptiveP(state['adaptive_target'], state['adaptive_decay']))
            else:
                ordered_samplers.append(SS_Sample())

            sampler = CustomSampler(ordered_samplers)

        input_ids = self.tokenizer.encode(
            prompt,
            add_bos=state['add_bos_token'],
            encode_special_tokens=True,
            embeddings=image_embeddings,
        )

        input_ids = input_ids[:, -get_max_prompt_length(state):]

        self._last_prompt_token_count = input_ids.shape[-1]

        if state['auto_max_new_tokens']:
            max_new_tokens = state['truncation_length'] - self._last_prompt_token_count
        else:
            max_new_tokens = state['max_new_tokens']

        eos_ids = [eid for eid in self.config.eos_token_id_list if eid is not None]

        stop_conditions = [] if state['ban_eos_token'] else list(eos_ids)

        filters = []
        logit_bias = state.get('logit_bias')
        if logit_bias:
            filters.append(LogitBiasFilter(self.tokenizer, logit_bias))

        # Suppress EOS tokens via logit bias so they are never sampled
        if state['ban_eos_token'] and eos_ids:
            eos_bias = {str(eid): float('-inf') for eid in eos_ids}
            filters.append(LogitBiasFilter(self.tokenizer, eos_bias))

        return_top_tokens = max(state.get('logprobs') or 0, 0)

        seed = state.get('seed', -1)
        job = Job(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            decode_special_tokens=not state['skip_special_tokens'],
            embeddings=image_embeddings if image_embeddings else None,
            sampler=sampler,
            seed=seed if seed >= 0 else None,
            stop_conditions=stop_conditions if stop_conditions else None,
            filters=filters if filters else None,
            return_top_tokens=return_top_tokens,
            return_probs=return_top_tokens > 0,
        )

        response_text = ""
        stop_event = state.get('stop_event')
        self.last_completion_probabilities = []

        result_queue = self.parallel_generator.submit(job)
        try:
            while True:
                if shared.stop_everything or (stop_event and stop_event.is_set()):
                    break
                try:
                    result = result_queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                if result is None or result.get("eos"):
                    if result is not None and return_top_tokens > 0:
                        self._capture_logprobs(result)
                    break
                chunk = result.get("text", "")
                if return_top_tokens > 0:
                    self._capture_logprobs(result)

                if chunk:
                    response_text += chunk
                    yield response_text
        finally:
            self.parallel_generator.cancel(job)

    def _capture_logprobs(self, result):
        """Convert ExLlamav3 top-k token data to the shared logprobs format."""
        top_k_tokens = result.get("top_k_tokens")
        top_k_probs = result.get("top_k_probs")
        if top_k_tokens is None or top_k_probs is None:
            return

        if not hasattr(self, '_id_to_piece'):
            self._id_to_piece = self.tokenizer.get_id_to_piece_list(True)

        id_to_piece = self._id_to_piece

        # Bulk-convert tensors to Python lists to avoid per-element .item() calls
        tk_tokens = top_k_tokens[0].tolist()   # (seq_len, k)
        tk_probs = top_k_probs[0].tolist()     # (seq_len, k)
        sampled_ids = result.get("token_ids")
        sampled_probs = result.get("token_probs")
        s_ids = sampled_ids[0].tolist() if sampled_ids is not None else None
        s_probs = sampled_probs[0].tolist() if sampled_probs is not None else None

        def _piece(tid):
            s = id_to_piece[tid] if tid < len(id_to_piece) else f"<{tid}>"
            return s.replace('\u2581', ' ')

        def _logprob(prob):
            return math.log(prob) if prob > 0 else float("-inf")

        for seq_idx in range(len(tk_tokens)):
            entry = {"top_logprobs": []}
            for k_idx in range(len(tk_tokens[seq_idx])):
                token_id = tk_tokens[seq_idx][k_idx]
                prob = tk_probs[seq_idx][k_idx]
                entry["top_logprobs"].append({"token": _piece(token_id), "logprob": _logprob(prob)})

            # Record the actually sampled token at the entry level so
            # format_completion_logprobs uses it instead of top_logprobs[0]
            # (they differ with non-greedy sampling).
            if s_ids is not None:
                entry["token"] = _piece(s_ids[seq_idx])
                entry["logprob"] = _logprob(s_probs[seq_idx]) if s_probs is not None else None

            self.last_completion_probabilities.append(entry)

    def generate(self, prompt, state):
        output = ""
        for chunk in self.generate_with_streaming(prompt, state):
            output = chunk

        return output

    def get_prompt_logits(self, input_ids):
        """Return logits for all positions via a single no-cache forward pass.

        Used by prompt logprobs computation. Returns (1, seq_len, vocab) on CPU in float32.
        """
        input_ids_tensor = input_ids if isinstance(input_ids, torch.Tensor) else torch.tensor(input_ids, dtype=torch.long)
        input_ids_tensor = input_ids_tensor.view(1, -1).cpu()
        with torch.inference_mode():
            output = self.model.forward(
                input_ids=input_ids_tensor,
                params={"attn_mode": "flash_attn_nc"}
            ).cpu().float()
            # Mask padding slots beyond the real vocab so they can't appear in top-k
            output[..., self.model.config.vocab_size:] = float("-inf")
            return output

    def get_logits(self, token_ids, **kwargs):
        """
        Process a batch of token_ids and return the logits for the last token.
        Uses flash_attn_nc (no cache) for correct results with recurrent models.
        """
        logits = self.model.forward(
            input_ids=token_ids,
            params={"attn_mode": "flash_attn_nc"}
        )

        return logits[:, -1:, :].float().cpu()

    def encode(self, string, **kwargs):
        add_bos = kwargs.pop('add_bos', True)
        if add_bos and self.tokenizer.bos_token and string.startswith(self.tokenizer.bos_token):
            add_bos = False
        return self.tokenizer.encode(string, add_bos=add_bos, **kwargs)

    def decode(self, ids, **kwargs):
        if isinstance(ids, torch.Tensor) and ids.dim() == 0:
            ids = ids.view(1)

        return self.tokenizer.decode(ids, **kwargs)

    @property
    def last_prompt_token_count(self):
        return getattr(self, '_last_prompt_token_count', 0)

    def unload(self):
        logger.info("Unloading ExLlamaV3 model components...")

        if self.parallel_generator is not None:
            try:
                self.parallel_generator.stop()
            except Exception as e:
                logger.warning(f"Error stopping parallel generator: {e}")

        if self.draft_model is not None:
            try:
                self.draft_model.unload()
            except Exception as e:
                logger.warning(f"Error unloading draft model: {e}")

        if self.model is not None:
            try:
                self.model.unload()
            except Exception as e:
                logger.warning(f"Error unloading main model: {e}")

        self.parallel_generator = None
        self.vision_model = None
        self.draft_model = None
        self.draft_cache = None
        self.model = None
        self.cache = None
        self.generator = None
        self.tokenizer = None
