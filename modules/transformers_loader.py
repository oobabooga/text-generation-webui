import gc
import os
import pprint
import re
import time
from pathlib import Path

import torch
import transformers
from accelerate import infer_auto_device_map, init_empty_weights
from accelerate.utils import (
    is_ccl_available,
    is_npu_available,
    is_xpu_available
)
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    is_torch_npu_available,
    is_torch_xpu_available
)

import modules.shared as shared
from modules.logging_colors import logger
from modules.models_settings import get_model_metadata

transformers.logging.set_verbosity_error()

local_rank = None
if shared.args.deepspeed:
    import deepspeed
    from transformers.integrations.deepspeed import (
        HfDeepSpeedConfig,
        is_deepspeed_zero3_enabled
    )

    from modules.deepspeed_parameters import generate_ds_config

    # Distributed setup
    local_rank = shared.args.local_rank if shared.args.local_rank is not None else int(os.getenv("LOCAL_RANK", "0"))
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    if is_xpu_available() and is_ccl_available():
        torch.xpu.set_device(local_rank)
        deepspeed.init_distributed(backend="ccl")
    elif is_npu_available():
        torch.npu.set_device(local_rank)
        deepspeed.init_distributed(dist_backend="hccl")
    else:
        torch.cuda.set_device(local_rank)
        deepspeed.init_distributed()
    ds_config = generate_ds_config(shared.args.bf16, 1 * world_size, shared.args.nvme_offload_dir)
    dschf = HfDeepSpeedConfig(ds_config)  # Keep this object alive for the Transformers integration



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
            trust_remote_code=shared.args.trust_remote_code,
            use_fast=not shared.args.no_use_fast
        )

    return tokenizer


def load_model_HF(model_name):
    path_to_model = Path(f'{shared.args.model_dir}/{model_name}')
    params = {
        'low_cpu_mem_usage': True,
        'torch_dtype': torch.bfloat16 if shared.args.bf16 else torch.float16,
    }

    if shared.args.trust_remote_code:
        params['trust_remote_code'] = True

    if shared.args.use_flash_attention_2:
        params['use_flash_attention_2'] = True

    if shared.args.force_safetensors:
        params['force_safetensors'] = True

    if shared.args.use_eager_attention:
        params['attn_implementation'] = 'eager'

    config = AutoConfig.from_pretrained(path_to_model, trust_remote_code=shared.args.trust_remote_code)

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
        shared.args.auto_devices,
        shared.args.disk,
        shared.args.deepspeed,
        shared.args.gpu_memory is not None,
        shared.args.cpu_memory is not None,
        shared.args.compress_pos_emb > 1,
        shared.args.alpha_value > 1,
    ])

    # Load the model without any special settings
    if should_use_default_loading:
        logger.info("TRANSFORMERS_PARAMS=")
        pprint.PrettyPrinter(indent=4, sort_dicts=False).pprint(params)
        print()

        model = LoaderClass.from_pretrained(path_to_model, **params)
        if not (hasattr(model, 'is_loaded_in_4bit') and model.is_loaded_in_4bit):
            device = get_device()
            if device:
                model = model.to(device)

    # DeepSpeed ZeRO-3
    elif shared.args.deepspeed:
        model = LoaderClass.from_pretrained(
            path_to_model,
            torch_dtype=params['torch_dtype'],
            trust_remote_code=params.get('trust_remote_code')
        )

        model = deepspeed.initialize(
            model=model,
            config_params=ds_config,
            model_parameters=None,
            optimizer=None,
            lr_scheduler=None
        )[0]

        model.module.eval()  # Inference
        logger.info(f'DeepSpeed ZeRO-3 is enabled: {is_deepspeed_zero3_enabled()}')

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
                if shared.args.auto_devices or shared.args.gpu_memory:
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
                params['offload_folder'] = shared.args.disk_cache_dir

        if shared.args.compress_pos_emb > 1:
            params['rope_scaling'] = {'type': 'linear', 'factor': shared.args.compress_pos_emb}
        elif shared.args.alpha_value > 1:
            params['rope_scaling'] = {'type': 'dynamic', 'factor': shared.args.alpha_value}

        logger.info("TRANSFORMERS_PARAMS=")
        pprint.PrettyPrinter(indent=4, sort_dicts=False).pprint(params)
        print()
        model = LoaderClass.from_pretrained(path_to_model, **params)

    if shared.args.torch_compile:
        model = torch.compile(model)

    return model


def get_max_memory_dict():
    max_memory = {}
    max_cpu_memory = shared.args.cpu_memory.strip() if shared.args.cpu_memory is not None else '99GiB'
    if shared.args.gpu_memory:
        memory_map = list(map(lambda x: x.strip(), shared.args.gpu_memory))
        for i in range(len(memory_map)):
            max_memory[i] = f'{memory_map[i]}GiB' if not re.match('.*ib$', memory_map[i].lower()) else memory_map[i]

        max_memory['cpu'] = f'{max_cpu_memory}GiB' if not re.match('.*ib$', max_cpu_memory.lower()) else max_cpu_memory

    # If --auto-devices is provided standalone, try to get a reasonable value
    # for the maximum memory of device :0
    elif shared.args.auto_devices:
        if is_xpu_available():
            total_mem = (torch.xpu.get_device_properties(0).total_memory / (1024 * 1024))
        else:
            total_mem = (torch.cuda.get_device_properties(0).total_memory / (1024 * 1024))

        suggestion = round((total_mem - 1000) / 1000) * 1000
        if total_mem - suggestion < 800:
            suggestion -= 1000

        suggestion = int(round(suggestion / 1000))
        logger.warning(f"Auto-assiging --gpu-memory {suggestion} for your GPU to try to prevent out-of-memory errors. You can manually set other values.")
        max_memory[0] = f'{suggestion}GiB'
        max_memory['cpu'] = f'{max_cpu_memory}GiB' if not re.match('.*ib$', max_cpu_memory.lower()) else max_cpu_memory

    return max_memory if len(max_memory) > 0 else None


def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif shared.args.deepspeed:
        import deepspeed
        return deepspeed.get_accelerator().current_device_name()
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    elif is_torch_xpu_available():
        return torch.device('xpu:0')
    elif is_torch_npu_available():
        return torch.device('npu:0')
    else:
        return None


def clear_torch_cache():
    gc.collect()
    if not shared.args.cpu:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif is_xpu_available():
            torch.xpu.empty_cache()
        elif is_npu_available():
            torch.npu.empty_cache()
        elif torch.backends.mps.is_available():
            if hasattr(torch.backends.mps, 'empty_cache'):
                torch.backends.mps.empty_cache()
