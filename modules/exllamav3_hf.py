import os
import traceback
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch
from exllamav3 import Cache, Config, Model
from exllamav3.cache import CacheLayer_fp16, CacheLayer_quant
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


class Exllamav3HF(PreTrainedModel, GenerationMixin):
    def __init__(self, model_dir):
        hf_config = PretrainedConfig.from_pretrained(model_dir)
        super().__init__(hf_config)

        exl3_config = Config.from_directory(model_dir)

        self.generation_config = GenerationConfig()
        self.ex_model = Model.from_config(exl3_config)

        # Calculate the closest multiple of 256 at or above the chosen value
        max_tokens = shared.args.ctx_size
        if max_tokens % 256 != 0:
            adjusted_tokens = ((max_tokens // 256) + 1) * 256
            logger.warning(f"max_num_tokens must be a multiple of 256. Adjusting from {max_tokens} to {adjusted_tokens}")
            max_tokens = adjusted_tokens

        # Parse cache type
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

        self.ex_cache = Cache(self.ex_model, max_num_tokens=max_tokens, layer_type=layer_type, **cache_kwargs)

        # Create load parameters dictionary
        load_params = {'progressbar': True}
        if shared.args.gpu_split:
            split = [float(alloc) for alloc in shared.args.gpu_split.split(",")]
            load_params['use_per_device'] = split

        # Tensor-parallelism
        if shared.args.enable_tp:
            load_params['tensor_p'] = True
            load_params['tp_backend'] = shared.args.tp_backend

        self.ex_model.load(**load_params)
        self.past_seq = None
        self.max_tokens = max_tokens

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

        # Reset the internal sequence state for standalone calls (logit viewer)
        # or the very first step of a new generation.
        if past_key_values is None:
            self.past_seq = None
            self.past_seq_negative = None

        if len(args) > 0:
            if not shared.args.cfg_cache:
                logger.error("Please enable the cfg-cache option to use CFG with ExLlamav3_HF.")
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
        if is_negative and past_key_values is not None and isinstance(past_key_values, list):
             seq = past_key_values + seq

        seq_tensor = torch.tensor(seq)
        reset = True

        # Maximum number of tokens to process in a single forward pass
        max_chunk_size = 256

        if past_seq is not None:
            min_length = min(past_seq.shape[0], seq_tensor.shape[0])
            indices = torch.nonzero(~torch.eq(past_seq[:min_length], seq_tensor[:min_length]))
            if len(indices) == 0 and seq_tensor.shape[0] > past_seq.shape[0]:
                reset = False

        # Create a single `params` dictionary that will be used and modified
        # in-place across all `forward` calls within this function.
        params = {
            "attn_mode": "flash_attn",
            "cache": ex_cache,
            "batch_shape": (1, self.max_tokens),
            "reconstruct": False,
            "past_len": 0
        }

        # Make the forward call
        if labels is None:
            # If it's an efficient continuation, process only the new tokens
            if not reset:
                params["past_len"] = past_seq.shape[0]
                tokens_to_process = seq_tensor[past_seq.shape[0]:]
            # Otherwise, process the whole sequence from scratch
            else:
                tokens_to_process = seq_tensor

            # Process all but the last token of the sequence/sub-sequence
            if tokens_to_process.shape[0] > 1:
                prefix_to_process = tokens_to_process[:-1]

                # Process in chunks if the number of tokens is large
                for i in range(0, prefix_to_process.shape[0], max_chunk_size):
                    chunk = prefix_to_process[i:i + max_chunk_size]
                    self.ex_model.forward(input_ids=chunk.view(1, -1), params=params)
                    params["past_len"] += chunk.shape[0]

            # Process the last token to get logits
            last_token = tokens_to_process[-1:].view(1, -1)
            logits = self.ex_model.forward(input_ids=last_token, params=params).to(input_ids.device).float()
        else:
            # When processing with labels, handle as a complete sequence
            params["attn_mode"] = "flash_attn_nc"
            logits = self.ex_model.forward(input_ids=seq_tensor.view(1,-1), params=params).float()


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

        return Exllamav3HF(pretrained_model_name_or_path)

    def unload(self):
        """Properly unload the ExllamaV3 model and free GPU memory."""
        if hasattr(self, 'ex_model') and self.ex_model is not None:
            self.ex_model.unload()
            self.ex_model = None

        if hasattr(self, 'ex_cache') and self.ex_cache is not None:
            self.ex_cache = None

        # Clean up any additional ExllamaV3 resources
        if hasattr(self, 'past_seq'):
            self.past_seq = None
        if hasattr(self, 'past_seq_negative'):
            self.past_seq_negative = None
        if hasattr(self, 'ex_cache_negative'):
            self.ex_cache_negative = None
