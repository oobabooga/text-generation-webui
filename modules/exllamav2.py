import json
import traceback
from pathlib import Path

import torch
from exllamav2 import (
    ExLlamaV2,
    ExLlamaV2Cache,
    ExLlamaV2Cache_8bit,
    ExLlamaV2Cache_Q4,
    ExLlamaV2Cache_Q6,
    ExLlamaV2Cache_Q8,
    ExLlamaV2Cache_TP,
    ExLlamaV2Config,
    ExLlamaV2Tokenizer
)
from exllamav2.generator import ExLlamaV2Sampler, ExLlamaV2StreamingGenerator

from modules import shared
from modules.logging_colors import logger
from modules.text_generation import get_max_prompt_length

try:
    import flash_attn
except Exception:
    logger.warning('Failed to load flash-attention due to the following error:\n')
    traceback.print_exc()


class Exllamav2Model:
    def __init__(self):
        pass

    @classmethod
    def from_pretrained(self, path_to_model):

        path_to_model = Path(f'{shared.args.model_dir}') / Path(path_to_model)

        config = ExLlamaV2Config()
        config.model_dir = str(path_to_model)
        config.prepare()

        config.max_seq_len = shared.args.ctx_size
        config.scale_pos_emb = shared.args.compress_pos_emb
        config.scale_alpha_value = shared.args.alpha_value
        config.no_flash_attn = shared.args.no_flash_attn
        config.no_xformers = shared.args.no_xformers
        config.no_sdpa = shared.args.no_sdpa
        config.num_experts_per_token = int(shared.args.num_experts_per_token)

        model = ExLlamaV2(config)

        split = None
        if shared.args.gpu_split:
            split = [float(alloc) for alloc in shared.args.gpu_split.split(",")]

        if shared.args.enable_tp:
            model.load_tp(split)
        elif not shared.args.autosplit:
            model.load(split)

        # Determine the correct cache type
        kv_cache_type = shared.args.cache_type.lower()

        if kv_cache_type == 'fp16':
            cache_type = ExLlamaV2Cache
        elif kv_cache_type == 'fp8':
            cache_type = ExLlamaV2Cache_8bit
        elif kv_cache_type == 'q8':
            cache_type = ExLlamaV2Cache_Q8
        elif kv_cache_type == 'q6':
            cache_type = ExLlamaV2Cache_Q6
        elif kv_cache_type == 'q4':
            cache_type = ExLlamaV2Cache_Q4
        else:
            raise ValueError(f"Invalid cache type for ExLlamaV2: {cache_type}. Valid options are: fp16, fp8, q8, q6, q4.")

        # Use TP if specified
        if shared.args.enable_tp:
            cache = ExLlamaV2Cache_TP(model, base=cache_type)
        else:
            cache = cache_type(model, lazy=shared.args.autosplit)

        if shared.args.autosplit and not shared.args.enable_tp:
            model.load_autosplit(cache)

        tokenizer = ExLlamaV2Tokenizer(config)

        # Initialize draft model for speculative decoding
        draft_model = None
        draft_cache = None

        if shared.args.model_draft and shared.args.model_draft.lower() not in ["none", ""]:
            logger.info(f"Loading draft model for speculative decoding: {shared.args.model_draft}")

            # Find the draft model path
            draft_path = Path(shared.args.model_draft)
            if not draft_path.exists():
                draft_path = Path(f'{shared.args.model_dir}') / Path(shared.args.model_draft)

            draft_config = ExLlamaV2Config()
            draft_config.model_dir = str(draft_path)
            draft_config.prepare()
            draft_config.arch_compat_overrides()

            # Set context size for draft model
            if shared.args.ctx_size_draft > 0:
                draft_config.max_seq_len = shared.args.ctx_size_draft
            else:
                draft_config.max_seq_len = config.max_seq_len

            draft_model = ExLlamaV2(draft_config)
            draft_cache = cache_type(draft_model, lazy=True)
            draft_model.load_autosplit(draft_cache)

            logger.info(f"Draft model loaded successfully with max_draft={shared.args.draft_max}")

        generator = ExLlamaV2StreamingGenerator(
            model,
            cache,
            tokenizer,
            draft_model=draft_model,
            draft_cache=draft_cache,
            num_speculative_tokens=shared.args.draft_max if draft_model is not None else 0
        )

        result = self()
        result.model = model
        result.cache = cache
        result.tokenizer = tokenizer
        result.generator = generator
        result.loras = None
        result.draft_model = draft_model
        result.draft_cache = draft_cache
        return result, result

    def encode(self, string, **kwargs):
        return self.tokenizer.encode(string, add_bos=True, encode_special_tokens=True)

    def decode(self, ids, **kwargs):
        if isinstance(ids, list):
            ids = torch.tensor([ids])
        elif isinstance(ids, torch.Tensor) and ids.numel() == 1:
            ids = ids.view(1, -1)

        return self.tokenizer.decode(ids, decode_special_tokens=True)[0]

    def get_logits(self, token_ids, **kwargs):
        self.cache.current_seq_len = 0
        if token_ids.shape[-1] > 1:
            self.model.forward(token_ids[:, :-1], self.cache, input_mask=None, preprocess_only=True, loras=self.loras)

        return self.model.forward(token_ids[:, -1:], self.cache, input_mask=None, loras=self.loras, **kwargs).float().cpu()

    def generate_with_streaming(self, prompt, state):
        settings = ExLlamaV2Sampler.Settings()

        settings.token_repetition_penalty = state['repetition_penalty']
        settings.token_repetition_range = -1 if state['repetition_penalty_range'] <= 0 else state['repetition_penalty_range']

        settings.token_frequency_penalty = state['frequency_penalty']
        settings.token_presence_penalty = state['presence_penalty']

        settings.temperature = state['temperature']
        settings.smoothing_factor = state['smoothing_factor']
        settings.min_temp = state['dynatemp_low'] if state['dynamic_temperature'] else 0
        settings.max_temp = state['dynatemp_high'] if state['dynamic_temperature'] else 0
        settings.temp_exponent = state['dynatemp_exponent']
        settings.top_k = state['top_k']
        settings.top_p = state['top_p']
        settings.top_a = state['top_a']
        settings.min_p = state['min_p']
        settings.tfs = state['tfs']
        settings.typical = state['typical_p']

        settings.temperature_last = state['temperature_last']

        settings.mirostat = state['mirostat_mode'] == 2
        settings.mirostat_tau = state['mirostat_tau']
        settings.mirostat_eta = state['mirostat_eta']

        if state['ban_eos_token']:
            settings.disallow_tokens(self.tokenizer, [self.tokenizer.eos_token_id])

        if state['custom_token_bans']:
            to_ban = [int(x) for x in state['custom_token_bans'].split(',')]
            if len(to_ban) > 0:
                settings.disallow_tokens(self.tokenizer, to_ban)

        settings.dry_allowed_length = state['dry_allowed_length']
        settings.dry_base = state['dry_base']
        settings.dry_multiplier = state['dry_multiplier']

        # Dry sequence breakers processing
        if state['dry_multiplier'] > 0 and state['dry_sequence_breakers']:
            dry_sequence_breakers = state['dry_sequence_breakers']

            # Support both JSON array notation and comma-separated strings.
            if not dry_sequence_breakers.startswith("["):
                dry_sequence_breakers = "[" + dry_sequence_breakers + "]"

            sequence_breaker_strings = json.loads(dry_sequence_breakers)
            # Prefix with 'a' to get the correct encoding of the token at the end of a text.
            sequence_breakers = {
                self.encode(f"a{s}")[0, -1].item() for s in sequence_breaker_strings
            }

            settings.dry_sequence_breakers = sequence_breakers

        settings.xtc_probability = state['xtc_probability']
        settings.xtc_threshold = state['xtc_threshold']

        ids = self.tokenizer.encode(prompt, add_bos=state['add_bos_token'], encode_special_tokens=True)
        ids = ids[:, -get_max_prompt_length(state):]

        if state['auto_max_new_tokens']:
            max_new_tokens = state['truncation_length'] - ids.shape[-1]
        else:
            max_new_tokens = state['max_new_tokens']

        # Reset speculative decoding stats if using a draft model
        if hasattr(self, 'draft_model') and self.draft_model is not None:
            self.generator.reset_sd_stats()

        self.generator.begin_stream(ids, settings, loras=self.loras)

        decoded_text = ''
        for i in range(max_new_tokens):
            chunk, eos, _ = self.generator.stream()
            if eos or shared.stop_everything:
                break

            decoded_text += chunk
            yield decoded_text

        # Log speculative decoding stats if using draft model
        if hasattr(self, 'draft_model') and self.draft_model is not None:
            efficiency, accuracy, total_tokens, total_draft_tokens, accepted_draft_tokens = self.generator.get_sd_stats()
            logger.info(f"Speculative decoding: accepted={accepted_draft_tokens}/{total_draft_tokens} tokens")

    def generate(self, prompt, state):
        output = ''
        for output in self.generate_with_streaming(prompt, state):
            pass

        return output
