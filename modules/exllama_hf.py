import os
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch
from torch.nn import CrossEntropyLoss
from transformers import GenerationConfig, PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast

from modules import shared
from modules.logging_colors import logger

try:
    from exllama.model import ExLlama, ExLlamaCache, ExLlamaConfig
except:
    logger.warning('Exllama module failed to load. Will attempt to load from repositories.')
    try:
        from modules.relative_imports import RelativeImport

        with RelativeImport("repositories/exllama"):
            from model import ExLlama, ExLlamaCache, ExLlamaConfig
    except:
        logger.error("Could not find repositories/exllama/. Make sure that exllama is cloned inside repositories/ and is up to date.")
        raise


class ExllamaHF(PreTrainedModel):
    def __init__(self, config: ExLlamaConfig):
        super().__init__(PretrainedConfig())
        self.ex_config = config
        self.ex_model = ExLlama(self.ex_config)
        self.generation_config = GenerationConfig()
        self.lora = None

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
        # TODO: Some decoding methods (such as Contrastive Search) may not work at this time
        assert len(args) == 0, 'no *args should be passed to forward'
        use_cache = kwargs.get('use_cache', True)
        labels = kwargs.get('labels', None)
        seq = kwargs['input_ids'][0].tolist()
        cache = kwargs['past_key_values'] if 'past_key_values' in kwargs else None
        if cache is None:
            cache = ExLlamaCache(self.ex_model)
            self.ex_model.forward(torch.tensor([seq[:-1]], dtype=torch.long), cache, preprocess_only=True, lora=self.lora)

        logits = self.ex_model.forward(torch.tensor([seq[-1:]], dtype=torch.long), cache, lora=self.lora).to(kwargs['input_ids'].device)

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

        return CausalLMOutputWithPast(logits=logits, past_key_values=cache if use_cache else None)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Optional[Union[str, os.PathLike]], *model_args, **kwargs):
        assert len(model_args) == 0 and len(kwargs) == 0, "extra args is currently not supported"
        if isinstance(pretrained_model_name_or_path, str):
            pretrained_model_name_or_path = Path(pretrained_model_name_or_path)

        pretrained_model_name_or_path = Path(f'{shared.args.model_dir}') / Path(pretrained_model_name_or_path)
        config = ExLlamaConfig(pretrained_model_name_or_path / 'config.json')

        # from 'oobabooga/text-generation-webui/modules/exllama.py'
        weight_path = None
        for ext in ['.safetensors', '.pt', '.bin']:
            found = list(pretrained_model_name_or_path.glob(f"*{ext}"))
            if len(found) > 0:
                weight_path = found[-1]
                break
        assert weight_path is not None, f'could not find weight in "{pretrained_model_name_or_path}"'

        config.model_path = str(weight_path)
        config.max_seq_len = shared.args.max_seq_len
        config.compress_pos_emb = shared.args.compress_pos_emb
        if shared.args.gpu_split:
            config.set_auto_map(shared.args.gpu_split)
            config.gpu_peer_fix = True

        if shared.args.alpha_value:
            config.alpha_value = shared.args.alpha_value
            config.calculate_rotary_embedding_base()

        if torch.version.hip:
            config.rmsnorm_no_half2 = True
            config.rope_no_half2 = True
            config.matmul_no_half2 = True
            config.silu_no_half2 = True

        # This slowes down a bit but align better with autogptq generation.
        # TODO: Should give user choice to tune the exllama config
        # config.fused_attn = False
        # config.fused_mlp_thd = 0

        return ExllamaHF(config)
