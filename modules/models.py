import gc
import os
import re
import time
import traceback
from pathlib import Path

import torch
import transformers
from accelerate import infer_auto_device_map, init_empty_weights
from accelerate.utils import is_ccl_available, is_xpu_available
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    GPTQConfig
)

import modules.shared as shared
from modules import RoPE, llama_attn_hijack, sampler_hijack
from modules.logging_colors import logger
from modules.models_settings import get_model_metadata

transformers.logging.set_verbosity_error()

local_rank = None
if shared.args.deepspeed:
    import deepspeed
    from transformers.deepspeed import (
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
    else:
        torch.cuda.set_device(local_rank)
        deepspeed.init_distributed()
    ds_config = generate_ds_config(shared.args.bf16, 1 * world_size, shared.args.nvme_offload_dir)
    dschf = HfDeepSpeedConfig(ds_config)  # Keep this object alive for the Transformers integration

sampler_hijack.hijack_samplers()


def load_model(model_name, loader=None):
    logger.info(f"Loading {model_name}...")
    t0 = time.time()

    shared.is_seq2seq = False
    load_func_map = {
        'Transformers': huggingface_loader,
        'AutoGPTQ': AutoGPTQ_loader,
        'GPTQ-for-LLaMa': GPTQ_loader,
        'llama.cpp': llamacpp_loader,
        'llamacpp_HF': llamacpp_HF_loader,
        'RWKV': RWKV_loader,
        'ExLlama': ExLlama_loader,
        'ExLlama_HF': ExLlama_HF_loader,
        'ExLlamav2': ExLlamav2_loader,
        'ExLlamav2_HF': ExLlamav2_HF_loader,
        'ctransformers': ctransformers_loader,
        'AutoAWQ': AutoAWQ_loader,
    }

    metadata = get_model_metadata(model_name)
    if loader is None:
        if shared.args.loader is not None:
            loader = shared.args.loader
        else:
            loader = metadata['loader']
            if loader is None:
                logger.error('The path to the model does not exist. Exiting.')
                raise ValueError

    shared.args.loader = loader
    output = load_func_map[loader](model_name)
    if type(output) is tuple:
        model, tokenizer = output
    else:
        model = output
        if model is None:
            return None, None
        else:
            tokenizer = load_tokenizer(model_name, model)

    # Hijack attention with xformers
    if any((shared.args.xformers, shared.args.sdp_attention)):
        llama_attn_hijack.hijack_llama_attention()

    shared.settings.update({k: v for k, v in metadata.items() if k in shared.settings})
    if loader.lower().startswith('exllama'):
        shared.settings['truncation_length'] = shared.args.max_seq_len
    elif loader in ['llama.cpp', 'llamacpp_HF', 'ctransformers']:
        shared.settings['truncation_length'] = shared.args.n_ctx

    logger.info(f"TRUNCATION LENGTH: {shared.settings['truncation_length']}")
    logger.info(f"INSTRUCTION TEMPLATE: {shared.settings['instruction_template']}")
    logger.info(f"Loaded the model in {(time.time()-t0):.2f} seconds.")
    return model, tokenizer


def load_tokenizer(model_name, model):
    tokenizer = None
    path_to_model = Path(f"{shared.args.model_dir}/{model_name}/")
    if any(s in model_name.lower() for s in ['gpt-4chan', 'gpt4chan']) and Path(f"{shared.args.model_dir}/gpt-j-6B/").exists():
        tokenizer = AutoTokenizer.from_pretrained(Path(f"{shared.args.model_dir}/gpt-j-6B/"))
    elif path_to_model.exists():
        if shared.args.use_fast:
            logger.info('Loading the tokenizer with use_fast=True.')

        tokenizer = AutoTokenizer.from_pretrained(
            path_to_model,
            trust_remote_code=shared.args.trust_remote_code,
            use_fast=shared.args.use_fast
        )

    return tokenizer


def huggingface_loader(model_name):

    path_to_model = Path(f'{shared.args.model_dir}/{model_name}')
    params = {
        'low_cpu_mem_usage': True,
        'trust_remote_code': shared.args.trust_remote_code,
        'torch_dtype': torch.bfloat16 if shared.args.bf16 else torch.float16,
        'use_safetensors': True if shared.args.force_safetensors else None
    }

    if shared.args.use_flash_attention_2:
        params['use_flash_attention_2'] = True

    config = AutoConfig.from_pretrained(path_to_model, trust_remote_code=params['trust_remote_code'])

    if 'chatglm' in model_name.lower():
        LoaderClass = AutoModel
    else:
        if config.to_dict().get('is_encoder_decoder', False):
            LoaderClass = AutoModelForSeq2SeqLM
            shared.is_seq2seq = True
        else:
            LoaderClass = AutoModelForCausalLM

    # Load the model in simple 16-bit mode by default
    if not any([shared.args.cpu, shared.args.load_in_8bit, shared.args.load_in_4bit, shared.args.auto_devices, shared.args.disk, shared.args.deepspeed, shared.args.gpu_memory is not None, shared.args.cpu_memory is not None, shared.args.compress_pos_emb > 1, shared.args.alpha_value > 1, shared.args.disable_exllama]):
        model = LoaderClass.from_pretrained(path_to_model, **params)
        if torch.backends.mps.is_available():
            device = torch.device('mps')
            model = model.to(device)
        elif is_xpu_available():
            device = torch.device("xpu")
            model = model.to(device)
        else:
            model = model.cuda()

    # DeepSpeed ZeRO-3
    elif shared.args.deepspeed:
        model = LoaderClass.from_pretrained(path_to_model, torch_dtype=params['torch_dtype'])
        model = deepspeed.initialize(model=model, config_params=ds_config, model_parameters=None, optimizer=None, lr_scheduler=None)[0]
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
            params['max_memory'] = get_max_memory_dict()
            if shared.args.load_in_4bit:
                # See https://github.com/huggingface/transformers/pull/23479/files
                # and https://huggingface.co/blog/4bit-transformers-bitsandbytes
                quantization_config_params = {
                    'load_in_4bit': True,
                    'bnb_4bit_compute_dtype': eval("torch.{}".format(shared.args.compute_dtype)) if shared.args.compute_dtype in ["bfloat16", "float16", "float32"] else None,
                    'bnb_4bit_quant_type': shared.args.quant_type,
                    'bnb_4bit_use_double_quant': shared.args.use_double_quant,
                }

                logger.info('Using the following 4-bit params: ' + str(quantization_config_params))
                params['quantization_config'] = BitsAndBytesConfig(**quantization_config_params)

            elif shared.args.load_in_8bit:
                if any((shared.args.auto_devices, shared.args.gpu_memory)):
                    params['quantization_config'] = BitsAndBytesConfig(load_in_8bit=True, llm_int8_enable_fp32_cpu_offload=True)
                else:
                    params['quantization_config'] = BitsAndBytesConfig(load_in_8bit=True)

                if params['max_memory'] is not None:
                    with init_empty_weights():
                        model = LoaderClass.from_config(config, trust_remote_code=params['trust_remote_code'])

                    model.tie_weights()
                    params['device_map'] = infer_auto_device_map(
                        model,
                        dtype=torch.int8,
                        max_memory=params['max_memory'],
                        no_split_module_classes=model._no_split_modules
                    )

            if shared.args.disk:
                params['offload_folder'] = shared.args.disk_cache_dir

        if shared.args.disable_exllama:
            try:
                gptq_config = GPTQConfig(bits=config.quantization_config.get('bits', 4), disable_exllama=True)
                params['quantization_config'] = gptq_config
                logger.info('Loading with ExLlama kernel disabled.')
            except:
                exc = traceback.format_exc()
                logger.error('Failed to disable exllama. Does the config.json for this model contain the necessary quantization info?')
                print(exc)

        if shared.args.compress_pos_emb > 1:
            params['rope_scaling'] = {'type': 'linear', 'factor': shared.args.compress_pos_emb}
        elif shared.args.alpha_value > 1:
            params['rope_scaling'] = {'type': 'dynamic', 'factor': RoPE.get_alpha_value(shared.args.alpha_value, shared.args.rope_freq_base)}

        model = LoaderClass.from_pretrained(path_to_model, **params)

    return model


def llamacpp_loader(model_name):
    from modules.llamacpp_model import LlamaCppModel

    path = Path(f'{shared.args.model_dir}/{model_name}')
    if path.is_file():
        model_file = path
    else:
        model_file = list(Path(f'{shared.args.model_dir}/{model_name}').glob('*.gguf'))[0]

    logger.info(f"llama.cpp weights detected: {model_file}")
    model, tokenizer = LlamaCppModel.from_pretrained(model_file)
    return model, tokenizer


def llamacpp_HF_loader(model_name):
    from modules.llamacpp_hf import LlamacppHF

    for fname in [model_name, "oobabooga_llama-tokenizer", "llama-tokenizer"]:
        path = Path(f'{shared.args.model_dir}/{fname}')
        if all((path / file).exists() for file in ['tokenizer_config.json', 'special_tokens_map.json', 'tokenizer.model']):
            logger.info(f'Using tokenizer from: {path}')
            break
    else:
        logger.error("Could not load the model because a tokenizer in transformers format was not found. Please download oobabooga/llama-tokenizer.")
        return None, None

    if shared.args.use_fast:
        logger.info('Loading the tokenizer with use_fast=True.')

    tokenizer = AutoTokenizer.from_pretrained(
        path,
        trust_remote_code=shared.args.trust_remote_code,
        use_fast=shared.args.use_fast
    )

    model = LlamacppHF.from_pretrained(model_name)
    return model, tokenizer


def ctransformers_loader(model_name):
    from modules.ctransformers_model import CtransformersModel

    path = Path(f'{shared.args.model_dir}/{model_name}')
    ctrans = CtransformersModel()
    if ctrans.model_type_is_auto():
        model_file = path
    else:
        if path.is_file():
            model_file = path
        else:
            entries = Path(f'{shared.args.model_dir}/{model_name}')
            gguf = list(entries.glob('*.gguf'))
            bin = list(entries.glob('*.bin'))
            if len(gguf) > 0:
                model_file = gguf[0]
            elif len(bin) > 0:
                model_file = bin[0]
            else:
                logger.error("Could not find a model for ctransformers.")
                return None, None

    logger.info(f'ctransformers weights detected: {model_file}')
    model, tokenizer = ctrans.from_pretrained(model_file)
    return model, tokenizer


def AutoAWQ_loader(model_name):
    from awq import AutoAWQForCausalLM

    model_dir = Path(f'{shared.args.model_dir}/{model_name}')

    model = AutoAWQForCausalLM.from_quantized(
                quant_path=model_dir,
                max_new_tokens=shared.args.max_seq_len,
                trust_remote_code=shared.args.trust_remote_code,
                fuse_layers=not shared.args.no_inject_fused_attention,
                max_memory=get_max_memory_dict(),
                batch_size=1,
                safetensors=any(model_dir.glob('*.safetensors')),
            )

    return model


def GPTQ_loader(model_name):

    # Monkey patch
    if shared.args.monkey_patch:
        logger.warning("Applying the monkey patch for using LoRAs with GPTQ models. It may cause undefined behavior outside its intended scope.")
        from modules.monkey_patch_gptq_lora import load_model_llama

        model, _ = load_model_llama(model_name)

    # No monkey patch
    else:
        import modules.GPTQ_loader

        model = modules.GPTQ_loader.load_quantized(model_name)

    return model


def AutoGPTQ_loader(model_name):
    import modules.AutoGPTQ_loader

    return modules.AutoGPTQ_loader.load_quantized(model_name)


def ExLlama_loader(model_name):
    from modules.exllama import ExllamaModel

    model, tokenizer = ExllamaModel.from_pretrained(model_name)
    return model, tokenizer


def ExLlama_HF_loader(model_name):
    from modules.exllama_hf import ExllamaHF

    return ExllamaHF.from_pretrained(model_name)


def ExLlamav2_loader(model_name):
    from modules.exllamav2 import Exllamav2Model

    model, tokenizer = Exllamav2Model.from_pretrained(model_name)
    return model, tokenizer


def ExLlamav2_HF_loader(model_name):
    from modules.exllamav2_hf import Exllamav2HF

    return Exllamav2HF.from_pretrained(model_name)


def RWKV_loader(model_name):
    '''
    This loader is not currently maintained as RWKV can now be loaded
    through the transformers library.
    '''
    from modules.RWKV import RWKVModel, RWKVTokenizer

    model = RWKVModel.from_pretrained(
        Path(f'{shared.args.model_dir}/{model_name}'),
        dtype="fp32" if shared.args.cpu else "bf16" if shared.args.bf16 else "fp16",
        device="cpu" if shared.args.cpu else "xpu" if is_xpu_available() else "cuda"
    )

    tokenizer = RWKVTokenizer.from_pretrained(Path(shared.args.model_dir))
    return model, tokenizer


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


def clear_torch_cache():
    gc.collect()
    if not shared.args.cpu:
        if is_xpu_available():
            torch.xpu.empty_cache()
        else:
            torch.cuda.empty_cache()


def unload_model():
    shared.model = shared.tokenizer = None
    shared.lora_names = []
    shared.model_dirty_from_training = False
    clear_torch_cache()


def reload_model():
    unload_model()
    shared.model, shared.tokenizer = load_model(shared.model_name)
