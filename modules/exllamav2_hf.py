import os
import traceback
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch
from exllamav2 import ExLlamaV2, ExLlamaV2Cache, ExLlamaV2Config
from torch.nn import CrossEntropyLoss
from transformers import GenerationConfig, PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast

from modules import shared
from modules.logging_colors import logger

try:
    import flash_attn
except ModuleNotFoundError:
    logger.warning(
        'You are running ExLlamaV2 without flash-attention. This will cause the VRAM usage '
        'to be a lot higher than it could be.\n'
        'Try installing flash-attention following the instructions here: '
        'https://github.com/Dao-AILab/flash-attention#installation-and-features'
    )
    pass
except Exception:
    logger.warning('Failed to load flash-attention due to the following error:\n')
    traceback.print_exc()


class Exllamav2HF(PreTrainedModel):
    def __init__(self, config: ExLlamaV2Config):
        super().__init__(PretrainedConfig())
        self.ex_config = config
        self.ex_model = ExLlamaV2(config)
        split = None
        if shared.args.gpu_split:
            split = [float(alloc) for alloc in shared.args.gpu_split.split(",")]

        self.ex_model.load(split)
        self.generation_config = GenerationConfig()
        self.loras = None

        self.ex_cache = ExLlamaV2Cache(self.ex_model)
        self.past_seq = None

        if shared.args.cfg_cache:
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

        config.max_seq_len = shared.args.max_seq_len
        config.scale_pos_emb = shared.args.compress_pos_emb
        config.scale_alpha_value = shared.args.alpha_value

        return Exllamav2HF(config)
