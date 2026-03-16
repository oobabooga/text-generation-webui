import pprint
from pathlib import Path

import torch
import torch.nn.functional as F
import transformers
from accelerate import infer_auto_device_map, init_empty_weights
from accelerate.utils import is_xpu_available
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    LogitsProcessor
)

import modules.shared as shared
from modules.logging_colors import logger
from modules.text_generation import get_reply_from_output_ids
from modules.torch_utils import get_device

transformers.logging.set_verbosity_error()


class _StopEverythingStoppingCriteria(transformers.StoppingCriteria):
    def __init__(self):
        transformers.StoppingCriteria.__init__(self)

    def __call__(self, input_ids: torch.LongTensor, _scores: torch.FloatTensor) -> bool:
        return shared.stop_everything


class Stream(transformers.StoppingCriteria):
    def __init__(self, callback_func=None):
        self.callback_func = callback_func

    def __call__(self, input_ids, scores) -> bool:
        if self.callback_func is not None:
            self.callback_func(input_ids[0])

        return False


class LogitsBiasProcessor(LogitsProcessor):
    def __init__(self, logit_bias={}):
        self.logit_bias = logit_bias
        if self.logit_bias:
            self.keys = list([int(key) for key in self.logit_bias.keys()])
            values = [self.logit_bias[str(key)] for key in self.keys]
            self.values = torch.tensor(values, dtype=torch.float, device=shared.model.device)

    def __call__(self, input_ids: torch.LongTensor, logits: torch.FloatTensor) -> torch.FloatTensor:
        if self.logit_bias:
            logits[0, self.keys] += self.values

        return logits

    def __repr__(self):
        return f"<{self.__class__.__name__}(logit_bias={self.logit_bias})>"


class LogprobProcessor(LogitsProcessor):
    def __init__(self, logprobs=None):
        self.logprobs = logprobs
        self.token_alternatives = {}
        self.token_alternatives_history = []

    def __call__(self, input_ids: torch.LongTensor, logits: torch.FloatTensor) -> torch.FloatTensor:
        if self.logprobs is not None:  # 0-5
            log_e_probabilities = F.log_softmax(logits, dim=1)
            top_values, top_indices = torch.topk(log_e_probabilities, k=self.logprobs)
            top_tokens = [get_reply_from_output_ids([tok]) for tok in top_indices[0]]
            top_probs = [float(x) for x in top_values[0]]
            self.token_alternatives = dict(zip(top_tokens, top_probs))
            self.token_alternatives_history.append(self.token_alternatives)

        return logits

    def __repr__(self):
        return f"<{self.__class__.__name__}(logprobs={self.logprobs}, token_alternatives={self.token_alternatives})>"


def load_tokenizer(model_name, tokenizer_dir=None):
    if tokenizer_dir:
        path_to_model = Path(tokenizer_dir)
    else:
        path_to_model = Path(f"{shared.args.model_dir}/{model_name}/")

    tokenizer = None
    if path_to_model.exists():
        if shared.args.no_use_fast:
            logger.info('Loading the tokenizer with use_fast=False.')

        tokenizer = AutoTokenizer.from_pretrained(
            path_to_model,
            trust_remote_code=shared.original_args.trust_remote_code,
            use_fast=not shared.args.no_use_fast
        )

    return tokenizer


def load_model_HF(model_name):
    torch._dynamo.config.disable = True

    path_to_model = Path(f'{shared.args.model_dir}/{model_name}')
    params = {
        'low_cpu_mem_usage': True,
        'attn_implementation': shared.args.attn_implementation,
        'torch_dtype': torch.bfloat16 if shared.args.bf16 else torch.float16,
    }

    if shared.original_args.trust_remote_code:
        params['trust_remote_code'] = True

    if shared.args.force_safetensors:
        params['force_safetensors'] = True

    config = AutoConfig.from_pretrained(path_to_model, trust_remote_code=shared.original_args.trust_remote_code)

    if 'chatglm' in model_name.lower():
        LoaderClass = AutoModel
    else:
        if config.to_dict().get('is_encoder_decoder', False):
            LoaderClass = AutoModelForSeq2SeqLM
            shared.is_seq2seq = True
        else:
            LoaderClass = AutoModelForCausalLM

    # Determine if we should use default loading
    should_use_default_loading = not any([
        shared.args.cpu,
        shared.args.load_in_8bit,
        shared.args.load_in_4bit,
        shared.args.disk,
        shared.args.cpu_memory is not None,
    ])

    # Load the model without any special settings
    if should_use_default_loading:
        params['device_map'] = 'auto'

        logger.info("TRANSFORMERS_PARAMS=")
        pprint.PrettyPrinter(indent=4, sort_dicts=False).pprint(params)
        print()

        model = LoaderClass.from_pretrained(path_to_model, **params)
        if not (hasattr(model, 'is_loaded_in_4bit') and model.is_loaded_in_4bit):
            device = get_device()
            if device:
                model = model.to(device)

    # Load with quantization and/or offloading
    else:
        if not any((shared.args.cpu, torch.cuda.is_available(), is_xpu_available(), torch.backends.mps.is_available())):
            logger.warning('torch.cuda.is_available() and is_xpu_available() returned False. This means that no GPU has been detected. Falling back to CPU mode.')
            shared.args.cpu = True

        if shared.args.cpu:
            params['torch_dtype'] = torch.float32
        else:
            params['device_map'] = 'auto'
            if x := get_max_memory_dict():
                params['max_memory'] = x

            if shared.args.load_in_4bit:
                # See https://github.com/huggingface/transformers/pull/23479/files
                # and https://huggingface.co/blog/4bit-transformers-bitsandbytes
                quantization_config_params = {
                    'load_in_4bit': True,
                    'bnb_4bit_compute_dtype': eval(f"torch.{shared.args.compute_dtype}") if shared.args.compute_dtype in ["bfloat16", "float16", "float32"] else None,
                    'bnb_4bit_quant_type': shared.args.quant_type,
                    'bnb_4bit_use_double_quant': shared.args.use_double_quant,
                    'llm_int8_enable_fp32_cpu_offload': True
                }
                params['quantization_config'] = BitsAndBytesConfig(**quantization_config_params)

            elif shared.args.load_in_8bit:
                if shared.args.gpu_split:
                    params['quantization_config'] = BitsAndBytesConfig(load_in_8bit=True, llm_int8_enable_fp32_cpu_offload=True)
                else:
                    params['quantization_config'] = BitsAndBytesConfig(load_in_8bit=True)

                if params.get('max_memory') is not None:
                    with init_empty_weights():
                        model = LoaderClass.from_config(config, trust_remote_code=params.get('trust_remote_code'))

                    model.tie_weights()
                    params['device_map'] = infer_auto_device_map(
                        model,
                        dtype=torch.int8,
                        max_memory=params.get('max_memory'),
                        no_split_module_classes=model._no_split_modules
                    )

            if shared.args.disk:
                params['offload_folder'] = str(Path(shared.args.disk_cache_dir))

        logger.info("TRANSFORMERS_PARAMS=")
        pprint.PrettyPrinter(indent=4, sort_dicts=False).pprint(params)
        print()
        model = LoaderClass.from_pretrained(path_to_model, **params)

    return model


def get_max_memory_dict():
    max_memory = {}
    if shared.args.cpu_memory > 0:
        max_memory['cpu'] = f'{shared.args.cpu_memory}GiB'

    if shared.args.gpu_split:
        for i, memory in enumerate(shared.args.gpu_split.split(',')):
            max_memory[i] = f'{memory}GiB'

    return max_memory if len(max_memory) > 0 else None
