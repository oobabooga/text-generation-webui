from modules.relative_imports import RelativeImport

with RelativeImport("repositories/gpt-fast"):
    from generate import _load_model
    from tp import maybe_init_dist

import os
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch
from torch.nn import CrossEntropyLoss
from transformers import GenerationConfig, PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast

from modules import shared
from modules.logging_colors import logger
from transformers import LogitsProcessorList, is_torch_xpu_available


class GptFastHF(PreTrainedModel):
    def __init__(self, model, device):
        super().__init__(PretrainedConfig())
        self.model = model
        self.torch_device = device
        self.generation_config = GenerationConfig()
        self.past_seq = None
        self.n_tokens = 0

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

        input_ids = kwargs['input_ids']
        past_seq = self.past_seq

        seq = input_ids[0]
        reset = True

        if labels is None:
            if past_seq is not None:
                min_length = min(past_seq.shape[0], seq.shape[0])
                indices = torch.nonzero(~torch.eq(past_seq[:min_length], seq[:min_length]))
                if len(indices) > 0:
                    longest_prefix = indices[0].item()
                else:
                    longest_prefix = min_length

                if longest_prefix > 0:
                    reset = False
                    self.n_tokens = longest_prefix
                    if len(seq) - longest_prefix > 0:
                        for token in seq[longest_prefix:-1]:
                            a = torch.tensor([[token]], device=self.torch_device, dtype=torch.int32)
                            b = torch.tensor([self.n_tokens], device=self.torch_device, dtype=torch.int32)
                            logits = self.model(a, b)
                            self.n_tokens += 1
                    elif len(seq) == longest_prefix:
                        # Very tricky: if the prefix we are reusing *is* the input_ids, then we have to back up the cache pointer by one,
                        # because we feed input_ids[-1] to forward() below, but that last token is already in the cache!
                        self.n_tokens -= 1

            if reset:
                self.n_tokens = 0
                for token in seq[:-1]:
                    a = torch.tensor([[token]], device=self.torch_device, dtype=torch.int32)
                    b = torch.tensor([self.n_tokens], device=self.torch_device, dtype=torch.int32)
                    logits = self.model(a, b)
                    self.n_tokens += 1

            a = torch.tensor([[seq[-1]]], device=self.torch_device, dtype=torch.int32)
            b = torch.tensor([self.n_tokens], device=self.torch_device, dtype=torch.int32)
            logits = self.model(a, b)
            self.n_tokens += 1
        # else:
        #     logits = self.model(input_ids).to(input_ids.device)

        self.past_seq = seq

        loss = None
        # if labels is not None:
        #     # Shift so that tokens < n predict n
        #     shift_logits = logits[..., :-1, :].contiguous()
        #     shift_labels = labels[..., 1:].contiguous()
        #     # Flatten the tokens
        #     loss_fct = CrossEntropyLoss()
        #     shift_logits = shift_logits.view(-1, logits.shape[-1])
        #     shift_labels = shift_labels.view(-1)
        #     # Enable model parallelism
        #     shift_labels = shift_labels.to(shift_logits.device)
        #     loss = loss_fct(shift_logits, shift_labels)

        return CausalLMOutputWithPast(logits=logits, past_key_values=None, loss=loss)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Optional[Union[str, os.PathLike]], *model_args, **kwargs):
        assert len(model_args) == 0 and len(kwargs) == 0, "extra args is currently not supported"

        if isinstance(pretrained_model_name_or_path, str):
            pretrained_model_name_or_path = Path(pretrained_model_name_or_path)

        path = Path(f'{shared.args.model_dir}') / Path(pretrained_model_name_or_path)
        model_file = list(path.glob('*.pth'))[0]
        logger.info(f"Model weights detected: {model_file}\n")

        # From here on the code has been copied from
        # https://github.com/pytorch-labs/gpt-fast/blob/main/generate.py
        rank = maybe_init_dist()
        use_tp = rank is not None
        precision = torch.bfloat16

        if torch.backends.mps.is_available():
            device = 'mps'
        elif is_torch_xpu_available():
            device = 'xpu:0'
        else:
            device = 'cuda'

        logger.info(f"Device: {device}")
        model = _load_model(model_file, device, precision, use_tp)
        # torch.cuda.synchronize()

        with torch.device(device):
            model.setup_caches(max_batch_size=1, max_seq_length=shared.args.max_seq_len)

        return GptFastHF(model, device)
