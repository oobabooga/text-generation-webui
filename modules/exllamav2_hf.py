import os
import traceback
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch
from exllamav2 import (
    ExLlamaV2,
    ExLlamaV2Cache,
    ExLlamaV2Cache_8bit,
    ExLlamaV2Cache_Q4,
    ExLlamaV2Cache_Q6,
    ExLlamaV2Cache_Q8,
    ExLlamaV2Cache_TP,
    ExLlamaV2Config
)
from torch.nn import CrossEntropyLoss
from transformers import (
    GenerationConfig,
    GenerationMixin,
    PretrainedConfig,
    PreTrainedModel
)
from transformers.modeling_outputs import CausalLMOutputWithPast

from modules import shared
from modules.logging_colors import logger

try:
    import flash_attn
except Exception:
    logger.warning('Failed to load flash-attention due to the following error:\n')
    traceback.print_exc()


class Exllamav2HF(PreTrainedModel, GenerationMixin):
    def __init__(self, config: ExLlamaV2Config):
        super().__init__(PretrainedConfig())
        self.ex_config = config
        self.loras = None
        self.generation_config = GenerationConfig()

        self.ex_model = ExLlamaV2(config)

        split = None
        if shared.args.gpu_split:
            split = [float(alloc) for alloc in shared.args.gpu_split.split(",")]

        if shared.args.enable_tp:
            self.ex_model.load_tp(split)
        elif not shared.args.autosplit:
            self.ex_model.load(split)

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
            self.ex_cache = ExLlamaV2Cache_TP(self.ex_model, base=cache_type)
        else:
            self.ex_cache = cache_type(self.ex_model, lazy=shared.args.autosplit)

        if shared.args.autosplit and not shared.args.enable_tp:
            self.ex_model.load_autosplit(self.ex_cache)

        self.past_seq = None
        if shared.args.cfg_cache:
            if shared.args.cache_8bit:
                self.ex_cache_negative = ExLlamaV2Cache_8bit(self.ex_model)
            elif shared.args.cache_4bit:
                self.ex_cache_negative = ExLlamaV2Cache_Q4(self.ex_model)
            else:
                self.ex_cache_negative = ExLlamaV2Cache(self.ex_model)

            self.past_seq_negative = None

    def _validate_model_class(self):
        pass

    def _validate_model_kwargs(self, model_kwargs: Dict[str, Any]):
        pass

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        return {'input_ids': input_ids, **kwargs}

    @property
    def device(self) -> torch.device:
        return torch.device(0)

    def __call__(self, *args, **kwargs):
        use_cache = kwargs.get('use_cache', True)
        labels = kwargs.get('labels', None)
        past_key_values = kwargs.get('past_key_values', None)

        if len(args) > 0:
            if not shared.args.cfg_cache:
                logger.error("Please enable the cfg-cache option to use CFG with ExLlamav2_HF.")
                return

            input_ids = args[0]
            is_negative = True
            past_seq = self.past_seq_negative
            ex_cache = self.ex_cache_negative
        else:
            input_ids = kwargs['input_ids']
            is_negative = False
            past_seq = self.past_seq
            ex_cache = self.ex_cache

        seq = input_ids[0].tolist()
        if is_negative and past_key_values is not None:
            seq = past_key_values + seq

        seq_tensor = torch.tensor(seq)
        reset = True

        # Make the forward call
        if labels is None:
            if past_seq is not None:
                min_length = min(past_seq.shape[0], seq_tensor.shape[0])
                indices = torch.nonzero(~torch.eq(past_seq[:min_length], seq_tensor[:min_length]))
                if len(indices) > 0:
                    longest_prefix = indices[0].item()
                else:
                    longest_prefix = min_length

                if longest_prefix > 0:
                    reset = False
                    ex_cache.current_seq_len = longest_prefix
                    if len(seq_tensor) - longest_prefix > 1:
                        self.ex_model.forward(seq_tensor[longest_prefix:-1].view(1, -1), ex_cache, preprocess_only=True, loras=self.loras)
                    elif len(seq_tensor) == longest_prefix:
                        # Very tricky: if the prefix we are reusing *is* the input_ids, then we have to back up the cache pointer by one,
                        # because we feed input_ids[-1] to forward() below, but that last token is already in the cache!
                        ex_cache.current_seq_len -= 1

            if reset:
                ex_cache.current_seq_len = 0
                if len(seq_tensor) > 1:
                    self.ex_model.forward(seq_tensor[:-1].view(1, -1), ex_cache, preprocess_only=True, loras=self.loras)

            logits = self.ex_model.forward(seq_tensor[-1:].view(1, -1), ex_cache, loras=self.loras).to(input_ids.device).float()
        else:
            ex_cache.current_seq_len = 0
            logits = self.ex_model.forward(seq_tensor.view(1, -1), ex_cache, last_id_only=False, loras=self.loras).float()

        if is_negative:
            self.past_seq_negative = seq_tensor
        else:
            self.past_seq = seq_tensor

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, logits.shape[-1])
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        return CausalLMOutputWithPast(logits=logits, past_key_values=seq if use_cache else None, loss=loss)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Optional[Union[str, os.PathLike]], *model_args, **kwargs):
        assert len(model_args) == 0 and len(kwargs) == 0, "extra args is currently not supported"
        if isinstance(pretrained_model_name_or_path, str):
            pretrained_model_name_or_path = Path(pretrained_model_name_or_path)

        pretrained_model_name_or_path = Path(f'{shared.args.model_dir}') / Path(pretrained_model_name_or_path)

        config = ExLlamaV2Config()
        config.model_dir = str(pretrained_model_name_or_path)
        config.prepare()

        config.max_seq_len = shared.args.ctx_size
        config.scale_pos_emb = shared.args.compress_pos_emb
        config.scale_alpha_value = shared.args.alpha_value
        config.no_flash_attn = shared.args.no_flash_attn
        config.no_xformers = shared.args.no_xformers
        config.no_sdpa = shared.args.no_sdpa
        config.num_experts_per_token = int(shared.args.num_experts_per_token)

        return Exllamav2HF(config)
