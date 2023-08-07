import os
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch
from torch.nn import CrossEntropyLoss
from transformers import GenerationConfig, PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast

from modules import shared
from modules.logging_colors import logger

import llama_cpp

if torch.cuda.is_available() and not torch.version.hip:
    try:
        import llama_cpp_cuda
    except:
        llama_cpp_cuda = None
else:
    llama_cpp_cuda = None


def llama_cpp_lib():
    if shared.args.cpu or llama_cpp_cuda is None:
        return llama_cpp
    else:
        return llama_cpp_cuda


class LlamacppHF(PreTrainedModel):
    def __init__(self, model):
        super().__init__(PretrainedConfig())
        self.model = model
        self.generation_config = GenerationConfig()
        self.cache = None

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
        input_ids = args[0] if len(args) > 0 else kwargs['input_ids']
        use_cache = kwargs.get('use_cache', True)
        labels = kwargs.get('labels', None)
        cache = kwargs.get('past_key_values', None)
        seq = input_ids[0].tolist()

        # Make the forward call
        seq_tensor = torch.tensor(seq)
        if labels is None:
            if self.cache is None or not torch.equal(self.cache, seq_tensor[:-1]):
                self.model.reset()
                self.model.eval(seq)
            else:
                self.model.eval([seq[-1]])

            logits = torch.tensor(self.model.scores[self.model.n_tokens - 1, :]).view(1, 1, -1).to(kwargs['input_ids'].device)
        else:
            self.model.reset()
            self.model.eval(seq)
            logits = torch.tensor(self.model.eval_logits)
            logits = logits.view(1, logits.shape[0], logits.shape[1]).to(input_ids.device)

        self.cache = seq_tensor

        # Based on transformers/models/llama/modeling_llama.py
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

        return CausalLMOutputWithPast(logits=logits, past_key_values=cache if use_cache else None, loss=loss)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Optional[Union[str, os.PathLike]], *model_args, **kwargs):
        assert len(model_args) == 0 and len(kwargs) == 0, "extra args is currently not supported"
        if isinstance(pretrained_model_name_or_path, str):
            pretrained_model_name_or_path = Path(pretrained_model_name_or_path)

        path = Path(f'{shared.args.model_dir}') / Path(pretrained_model_name_or_path)
        if path.is_file():
            model_file = path
        else:
            model_file = list(path.glob('*ggml*.bin'))[0]

        logger.info(f"llama.cpp weights detected: {model_file}\n")
        params = {
            'model_path': str(model_file),
            'n_ctx': shared.args.n_ctx,
            'seed': int(shared.args.llama_cpp_seed),
            'n_threads': shared.args.threads or None,
            'n_batch': shared.args.n_batch,
            'use_mmap': not shared.args.no_mmap,
            'use_mlock': shared.args.mlock,
            'low_vram': shared.args.low_vram,
            'n_gpu_layers': shared.args.n_gpu_layers,
            'rope_freq_base': 10000 * shared.args.alpha_value ** (64 / 63.),
            'rope_freq_scale': 1.0 / shared.args.compress_pos_emb,
            'n_gqa': shared.args.n_gqa or None,
            'rms_norm_eps': shared.args.rms_norm_eps or None,
            'logits_all': True,
        }

        Llama = llama_cpp_lib().Llama
        model = Llama(**params)

        return LlamacppHF(model)
