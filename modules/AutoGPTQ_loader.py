from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
from pathlib import Path
import torch
import re
import json
import logging

import modules.shared as shared


# Used to locate the .bin/.safetensors quantized file
def find_quantized_model_file(model_name):
    if shared.args.checkpoint:
        return Path(shared.args.checkpoint)

    path_to_model = Path(f'{shared.args.model_dir}/{model_name}')
    bin_path = None
    priority_name_list = [
        Path(f'{shared.args.model_dir}/{model_name}{hyphen}{shared.args.wbits}bit{group}{ext}')
        for group in ([f'-{shared.args.groupsize}g', ''] if shared.args.groupsize > 0 else [''])
        for ext in ['.safetensors', '.bin']
        for hyphen in ['-', f'/{model_name}-', '/']
    ]
    for path in priority_name_list:
        if path.exists():
            bin_path = path
            break

    # If the model hasn't been found with a well-behaved name, pick the last .bin
    # or the last .safetensors found in its folder as a last resort
    if not bin_path:
        found_bins = [bin for bin in list(path_to_model.glob("*.bin")) if 'pytorch' not in bin.stem]  # ignore pytorch bins
        found_safetensors = list(path_to_model.glob("*.safetensors"))
        bin_path = None

        if len(found_bins) > 0:
            if len(found_bins) > 1:
                logging.info('Warning: more than one .bin model has been found. The last one will be selected. It could be wrong.')
            bin_path = found_bins[-1]
        elif len(found_safetensors) > 0:
            if len(found_bins) > 1:
                logging.info('Warning: more than one .safetensors model has been found. The last one will be selected. It could be wrong.')
            bin_path = found_safetensors[-1]

    return bin_path


def has_quantize_config(model_name):
    path_to_model = Path(f'{shared.args.model_dir}/{model_name}')
    return (path_to_model / 'quantize_config.json').exists()


def set_quantize_config(model_name):
    path_to_model = Path(f'{shared.args.model_dir}/{model_name}')
    try:
        with open(path_to_model / 'quantize_config.json') as f:
            conf = json.load(f)
            if conf:
                shared.args.wbits = conf.get('bits', shared.args.wbits)
                shared.args.groupsize = max(conf.get('group_size', shared.args.groupsize), 0)  # convert -1 to 0
                shared.args.autogptq_act_order = conf.get('desc_act', shared.args.autogptq_act_order)
                logging.info(f'Quantize config found for {model_name}. Setting wbits={shared.args.wbits} and groupsize={shared.args.groupsize}. act-order={shared.args.autogptq_act_order}')
    except FileNotFoundError:
        logging.info(f'No quantize config found for {model_name}.')
        return
    except JsonDecodeError:
        logging.error(f'quantize_config.json invalid for {model_name}.')
        return


def load_quantized(model_name):
    model_file = find_quantized_model_file(model_name)
    if model_file is None:
        raise FileNotFoundError(f'No quantized model found for {model_name}')

    safetensors = model_file.suffix == '.safetensors'

    path_to_model = Path(f'{shared.args.model_dir}/{model_name}')

    # check if model has quantize_config.json else use settings
    quantize_config = None
    if not has_quantize_config(model_name):
        quantize_config = BaseQuantizeConfig(
            bits=bits if (bits := shared.args.wbits) else 4,  # we shouldn't be here if wbits is not set, but default to 4 anyway
            group_size=gs if (gs := shared.args.groupsize) > 0 else -1,  # convert 0 to -1
            desc_act=shared.args.autogptq_act_order
        )

    max_memory = None

    if shared.args.gpu_memory:
        memory_map = list(map(lambda x: x.strip(), shared.args.gpu_memory))
        max_cpu_memory = shared.args.cpu_memory.strip() if shared.args.cpu_memory is not None else '99GiB'
        max_memory = {}
        for i in range(len(memory_map)):
            max_memory[i] = f'{memory_map[i]}GiB' if not re.match('.*ib$', memory_map[i].lower()) else memory_map[i]
        max_memory['cpu'] = max_cpu_memory
    elif shared.args.auto_devices:
        total_mem = (torch.cuda.get_device_properties(0).total_memory / (1024 * 1024))
        suggestion = round((total_mem - 1000) / 1000) * 1000
        if total_mem - suggestion < 800:
            suggestion -= 1000
        suggestion = int(round(suggestion / 1000))
        logging.info(
            f"\033[1;32;1mAuto-assigning --gpu-memory {suggestion} for your GPU to try to prevent out-of-memory errors.\nYou can manually set other values.\033[0;37;0m")

        max_memory = {0: f'{suggestion}GiB', 'cpu': f'{shared.args.cpu_memory or 99}GiB'}

    if max_memory:
        logging.info(f'max_memory: {max_memory}')

    dev = "cpu" if shared.args.cpu else "cuda:0"  # cpu is not supported for now

    #dev = "cuda"

    logging.info(f'Loading quantized model with AutoGPTQ from {model_file}')

    model = None

    try:
        model = AutoGPTQForCausalLM.from_quantized(path_to_model,
                                                   device=dev,
                                                   use_triton=shared.args.autogptq_triton,
                                                   use_safetensors=safetensors,
                                                   quantize_config=quantize_config,
                                                   model_basename=model_file.stem,
                                                   trust_remote_code=shared.args.trust_remote_code,
                                                   max_memory=max_memory,
                                                   device_map=shared.args.autogptq_device_map,
                                                   fused_attn=shared.args.quant_attn,
                                                   fused_mlp=shared.args.fused_mlp,
                                                   use_cuda_fp16=shared.args.autogptq_cuda_tweak,
                                                   strict=not shared.args.autogptq_compat)
    except ValueError:
        logging.error('Could not load model. The model might be using old quantization. Use the --autogptq-compat flag.')
        raise Exception('Could not load model. The model might be using old quantization. Use the --autogptq-compat flag.')

    return model
