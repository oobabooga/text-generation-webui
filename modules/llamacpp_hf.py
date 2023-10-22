import os
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch
from torch.nn import CrossEntropyLoss
from transformers import GenerationConfig, PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast

from modules import RoPE, shared
from modules.logging_colors import logger

try:
    import llama_cpp
except:
    llama_cpp = None

try:
    import llama_cpp_cuda
except:
    llama_cpp_cuda = None


def llama_cpp_lib():
    if (shared.args.cpu and llama_cpp is not None) or llama_cpp_cuda is None:
        return llama_cpp
    else:
        return llama_cpp_cuda


class LlamacppHF(PreTrainedModel):
    def __init__(self, model, path):
        super().__init__(PretrainedConfig())
        self.model = model
        self.generation_config = GenerationConfig()

        self.past_seq = None
        self.llamacpp_cache = {
            'n_tokens': self.model.n_tokens,
            'input_ids': self.model.input_ids,
            'scores': self.model.scores,
            'ctx': self.model.ctx
        }

        if shared.args.cfg_cache:
            self.past_seq_negative = None
            self.llamacpp_cache_negative = {
                'n_tokens': self.model.n_tokens,
                'input_ids': self.model.input_ids.copy(),
                'scores': self.model.scores.copy(),
                'ctx': llama_cpp_lib().llama_new_context_with_model(model.model, model.context_params)
            }

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
            'ctx': self.model.ctx
        })

    def save_negative_cache(self):
        self.llamacpp_cache_negative.update({
            'n_tokens': self.model.n_tokens,
            'input_ids': self.model.input_ids,
            'scores': self.model.scores,
            'ctx': self.model.ctx
        })

    def load_cache(self):
        self.model.n_tokens = self.llamacpp_cache['n_tokens']
        self.model.input_ids = self.llamacpp_cache['input_ids']
        self.model.scores = self.llamacpp_cache['scores']
        self.model.ctx = self.llamacpp_cache['ctx']

    def load_negative_cache(self):
        self.model.n_tokens = self.llamacpp_cache_negative['n_tokens']
        self.model.input_ids = self.llamacpp_cache_negative['input_ids']
        self.model.scores = self.llamacpp_cache_negative['scores']
        self.model.ctx = self.llamacpp_cache_negative['ctx']

    @property
    def device(self) -> torch.device:
        return torch.device(0)

    def __call__(self, *args, **kwargs):
        use_cache = kwargs.get('use_cache', True)
        labels = kwargs.get('labels', None)
        past_key_values = kwargs.get('past_key_values', None)

        if len(args) > 0:
            if not shared.args.cfg_cache:
                logger.error("Please enable the cfg-cache option to use CFG with llamacpp_HF.")
                return

            input_ids = args[0]
            is_negative = True
            past_seq = self.past_seq_negative
            self.load_negative_cache()
        else:
            input_ids = kwargs['input_ids']
            is_negative = False
            past_seq = self.past_seq
            self.load_cache()

        seq = input_ids[0].tolist()
        if is_negative and past_key_values is not None:
            seq = past_key_values + seq

        seq_tensor = torch.tensor(seq)
        reset = True

        # Make the forward call. The prefix-match code has been adapted from
        # https://github.com/abetlen/llama-cpp-python/commit/f4090a0bb2a2a25acfe28d31c82cc1aa273bedee
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
                    self.model.n_tokens = longest_prefix
                    if len(seq_tensor) - longest_prefix > 0:
                        self.model.eval(seq[longest_prefix:])

            if reset:
                self.model.reset()
                self.model.eval(seq)

            logits = torch.tensor(self.model.scores[self.model.n_tokens - 1, :]).view(1, 1, -1).to(input_ids.device)
        else:
            self.model.reset()
            self.model.eval(seq)
            logits = torch.tensor(self.model.eval_logits)
            logits = logits.view(1, logits.shape[0], logits.shape[1]).to(input_ids.device)

        if is_negative:
            self.save_negative_cache()
            self.past_seq_negative = seq_tensor
        else:
            self.save_cache()
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

        path = Path(f'{shared.args.model_dir}') / Path(pretrained_model_name_or_path)
        if path.is_file():
            model_file = path
        else:
            model_file = list(path.glob('*.gguf'))[0]

        logger.info(f"llama.cpp weights detected: {model_file}\n")

        if shared.args.tensor_split is None or shared.args.tensor_split.strip() == '':
            tensor_split_list = None
        else:
            tensor_split_list = [float(x) for x in shared.args.tensor_split.strip().split(",")]

        params = {
            'model_path': str(model_file),
            'n_ctx': shared.args.n_ctx,
            'seed': int(shared.args.llama_cpp_seed),
            'n_threads': shared.args.threads or None,
            'n_threads_batch': shared.args.threads_batch or None,
            'n_batch': shared.args.n_batch,
            'use_mmap': not shared.args.no_mmap,
            'use_mlock': shared.args.mlock,
            'mul_mat_q': shared.args.mul_mat_q,
            'numa': shared.args.numa,
            'n_gpu_layers': shared.args.n_gpu_layers,
            'rope_freq_base': RoPE.get_rope_freq_base(shared.args.alpha_value, shared.args.rope_freq_base),
            'tensor_split': tensor_split_list,
            'rope_freq_scale': 1.0 / shared.args.compress_pos_emb,
            'logits_all': True,
        }

        Llama = llama_cpp_lib().Llama
        model = Llama(**params)

        return LlamacppHF(model, model_file)
