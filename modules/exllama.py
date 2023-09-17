from pathlib import Path

import torch
import torch.nn.functional as F
from torch import version as torch_version

from modules import shared
from modules.logging_colors import logger
from modules.models import clear_torch_cache
from modules.text_generation import get_max_prompt_length

try:
    from exllama.generator import ExLlamaGenerator
    from exllama.model import ExLlama, ExLlamaCache, ExLlamaConfig
    from exllama.tokenizer import ExLlamaTokenizer
except:
    logger.warning('exllama module failed to import. Will attempt to import from repositories/.')
    try:
        from modules.relative_imports import RelativeImport

        with RelativeImport("repositories/exllama"):
            from generator import ExLlamaGenerator
            from model import ExLlama, ExLlamaCache, ExLlamaConfig
            from tokenizer import ExLlamaTokenizer
    except:
        logger.error(
            "Could not find repositories/exllama. Please ensure that exllama"
            " (https://github.com/turboderp/exllama) is cloned inside repositories/ and is up to date."
        )
        raise


class ExllamaModel:
    def __init__(self):
        pass

    @classmethod
    def from_pretrained(self, path_to_model):

        path_to_model = Path(f'{shared.args.model_dir}') / Path(path_to_model)
        tokenizer_model_path = path_to_model / "tokenizer.model"
        model_config_path = path_to_model / "config.json"

        # Find the model checkpoint
        model_path = None
        for ext in ['.safetensors', '.pt', '.bin']:
            found = list(path_to_model.glob(f"*{ext}"))
            if len(found) > 0:
                if len(found) > 1:
                    logger.warning(f'More than one {ext} model has been found. The last one will be selected. It could be wrong.')

                model_path = found[-1]
                break

        config = ExLlamaConfig(str(model_config_path))
        config.model_path = str(model_path)
        config.max_seq_len = shared.args.max_seq_len
        config.compress_pos_emb = shared.args.compress_pos_emb
        if shared.args.gpu_split:
            config.set_auto_map(shared.args.gpu_split)
            config.gpu_peer_fix = True

        if shared.args.alpha_value > 1 and shared.args.rope_freq_base == 0:
            config.alpha_value = shared.args.alpha_value
            config.calculate_rotary_embedding_base()
        elif shared.args.rope_freq_base > 0:
            config.rotary_embedding_base = shared.args.rope_freq_base

        if torch_version.hip:
            config.rmsnorm_no_half2 = True
            config.rope_no_half2 = True
            config.matmul_no_half2 = True
            config.silu_no_half2 = True

        model = ExLlama(config)
        tokenizer = ExLlamaTokenizer(str(tokenizer_model_path))
        cache = ExLlamaCache(model)
        generator = ExLlamaGenerator(model, tokenizer, cache)

        result = self()
        result.config = config
        result.model = model
        result.cache = cache
        result.tokenizer = tokenizer
        result.generator = generator
        return result, result

    def generate_with_streaming(self, prompt, state):

        # The cache batch size must be 2 for CFG and 1 otherwise
        if state['guidance_scale'] == 1:
            if self.cache.batch_size == 2:
                del self.cache
                clear_torch_cache()
                self.cache = ExLlamaCache(self.model)
                self.generator = ExLlamaGenerator(self.model, self.tokenizer, self.cache)
        else:
            if self.cache.batch_size == 1:
                del self.cache
                clear_torch_cache()
                self.cache = ExLlamaCache(self.model, batch_size=2)
                self.generator = ExLlamaGenerator(self.model, self.tokenizer, self.cache)

        self.generator.settings.temperature = state['temperature']
        self.generator.settings.top_p = state['top_p']
        self.generator.settings.top_k = state['top_k']
        self.generator.settings.typical = state['typical_p']
        self.generator.settings.token_repetition_penalty_max = state['repetition_penalty']
        self.generator.settings.token_repetition_penalty_sustain = -1 if state['repetition_penalty_range'] <= 0 else state['repetition_penalty_range']
        if state['ban_eos_token']:
            self.generator.disallow_tokens([self.tokenizer.eos_token_id])
        else:
            self.generator.disallow_tokens(None)

        if state['custom_token_bans']:
            to_ban = [int(x) for x in state['custom_token_bans'].split(',')]
            if len(to_ban) > 0:
                self.generator.disallow_tokens(to_ban)

        # Case 1: no CFG
        if state['guidance_scale'] == 1:
            self.generator.end_beam_search()

            # Tokenizing the input
            ids = self.generator.tokenizer.encode(prompt, max_seq_len=self.model.config.max_seq_len)
            if state['add_bos_token']:
                ids = torch.cat(
                    [torch.tensor([[self.tokenizer.bos_token_id]]).to(ids.device),
                     ids], dim=1
                ).to(torch.int64)
            ids = ids[:, -get_max_prompt_length(state):]
            if state['auto_max_new_tokens']:
                max_new_tokens = state['truncation_length'] - ids.shape[-1]
            else:
                max_new_tokens = state['max_new_tokens']

            self.generator.gen_begin_reuse(ids)
            initial_len = self.generator.sequence[0].shape[0]
            has_leading_space = False

            for i in range(max_new_tokens):
                token = self.generator.gen_single_token()
                if i == 0 and self.generator.tokenizer.tokenizer.IdToPiece(int(token)).startswith('▁'):
                    has_leading_space = True

                decoded_text = self.generator.tokenizer.decode(self.generator.sequence[0][initial_len:])
                if has_leading_space:
                    decoded_text = ' ' + decoded_text

                yield decoded_text
                if token.item() == self.generator.tokenizer.eos_token_id or shared.stop_everything:
                    break

        # Case 2: CFG
        # Copied from https://github.com/turboderp/exllama/blob/master/example_cfg.py
        else:
            alpha = state['guidance_scale']
            prompts = [prompt, state['negative_prompt'] or '']

            ids, mask = self.tokenizer.encode(
                prompts,
                return_mask=True,
                max_seq_len=self.model.config.max_seq_len,
                add_bos=state['add_bos_token']
            )
            if state['auto_max_new_tokens']:
                max_new_tokens = state['truncation_length'] - ids[0].shape[-1]
            else:
                max_new_tokens = state['max_new_tokens']

            self.generator.gen_begin(ids, mask=mask)
            initial_len = self.generator.sequence[0].shape[0]
            has_leading_space = False

            for i in range(max_new_tokens):
                logits = self.model.forward(self.generator.sequence[:, -1:], self.cache, input_mask=mask)
                self.generator.apply_rep_penalty(logits)

                logits = F.log_softmax(logits, dim=-1)
                logits_mixed = alpha * logits[0] + (1 - alpha) * logits[1]

                token, _ = self.generator.sample_current(logits_mixed)
                if i == 0 and self.generator.tokenizer.tokenizer.IdToPiece(int(token)).startswith('▁'):
                    has_leading_space = True

                decoded_text = self.generator.tokenizer.decode(self.generator.sequence[0][initial_len:])
                if has_leading_space:
                    decoded_text = ' ' + decoded_text

                yield decoded_text
                if token.item() == self.tokenizer.eos_token_id or shared.stop_everything:
                    break

                batch_token = token.repeat(2, 1)
                self.generator.gen_accept_token(batch_token)

    def generate(self, prompt, state):
        output = ''
        for output in self.generate_with_streaming(prompt, state):
            pass

        return output

    def encode(self, string, **kwargs):
        return self.tokenizer.encode(string, max_seq_len=self.model.config.max_seq_len, add_bos=True)

    def decode(self, ids, **kwargs):
        if isinstance(ids, list):
            ids = torch.tensor([ids])
        elif isinstance(ids, torch.Tensor) and ids.numel() == 1:
            ids = ids.view(1, -1)

        return self.tokenizer.decode(ids)[0]

    def get_logits(self, token_ids, **kwargs):
        self.cache.current_seq_len = 0
        self.model.forward(token_ids[:, :-1], self.cache, input_mask=None, preprocess_only=True)
        return self.model.forward(token_ids[:, -1:], self.cache, **kwargs).float().cpu()
