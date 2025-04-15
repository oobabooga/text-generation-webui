import os
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch
from torch.nn import CrossEntropyLoss
from transformers import GenerationConfig, PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast

from modules import shared
from modules.llama_cpp_server import LlamaServer
from modules.logging_colors import logger


class LlamacppHF(PreTrainedModel):
    def __init__(self, model):
        super().__init__(PretrainedConfig())
        self.model = model
        self.generation_config = GenerationConfig()

    def _validate_model_class(self):
        pass

    def _validate_model_kwargs(self, model_kwargs: Dict[str, Any]):
        pass

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        return {'input_ids': input_ids, **kwargs}

    def save_cache(self):
        self.llamacpp_cache.update({
            'n_tokens': self.model.n_tokens,
            'input_ids': self.model.input_ids,
            'scores': self.model.scores,
            'ctx': self.model._ctx.ctx
        })

    def load_cache(self):
        self.model.n_tokens = self.llamacpp_cache['n_tokens']
        self.model.input_ids = self.llamacpp_cache['input_ids']
        self.model.scores = self.llamacpp_cache['scores']
        self.model._ctx.ctx = self.llamacpp_cache['ctx']

    @property
    def device(self) -> torch.device:
        return torch.device(0)

    def __call__(self, *args, **kwargs):
        use_cache = kwargs.get('use_cache', True)
        labels = kwargs.get('labels', None)

        input_ids = args[0] if len(args) > 0 else kwargs['input_ids']
        seq = input_ids[0].tolist()
        seq_tensor = torch.tensor(seq)

        # Make the forward call. The prefix-match code has been adapted from
        # https://github.com/abetlen/llama-cpp-python/commit/f4090a0bb2a2a25acfe28d31c82cc1aa273bedee
        if labels is None:
            # logits = ...
            logits = torch.tensor(self.model.scores[self.model, :]).view(1, 1, -1).to(input_ids.device)
        else:
            # logits = ..., logits_all=True
            logits = logits.view(1, logits.shape[0], logits.shape[1]).to(input_ids.device)

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

        path = Path(f'{shared.args.model_dir}') / Path(pretrained_model_name_or_path)
        if path.is_file():
            model_file = path
        else:
            model_file = sorted(path.glob('*.gguf'))[0]

        logger.info(f"llama.cpp weights detected: {model_file}\n")

        try:
            model = LlamaServer(
                model_file,
            )
        except Exception as e:
            error_message = (
                f"Failed loading the model. **This usually happens due to lack of memory**. Try these steps:\n"
                f"1. Reduce the context length `n_ctx` (currently {shared.args.n_ctx})."
                f"{' Try a lower value like 4096.' if shared.args.n_ctx > 4096 else '.'}"
                "\n"
                f"2. Lower the `n-gpu-layers` value (currently {shared.args.n_gpu_layers})."
            )

            raise type(e)(error_message) from e

        return LlamacppHF(model, model_file)
