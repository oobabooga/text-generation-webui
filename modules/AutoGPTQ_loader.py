from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
from pathlib import Path
import torch
import re
import json

import modules.shared as shared


# Used to locate the .bin/.safetensors quantized file
def find_quantized_model_file(model_name):
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
        found_bins = list(path_to_model.glob("*.bin"))
        found_safetensors = list(path_to_model.glob("*.safetensors"))
        bin_path = None

        if len(found_bins) > 0:
            if len(found_bins) > 1:
                print('Warning: more than one .bin model has been found. The last one will be selected. It could be wrong.')
            bin_path = found_bins[-1]
        elif len(found_safetensors) > 0:
            if len(found_bins) > 1:
                print('Warning: more than one .safetensors model has been found. The last one will be selected. It could be wrong.')
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
                shared.args.wbits = conf['bits']
                shared.args.groupsize = conf['group_size']
                print(f'Quantize config found for {model_name}. Setting wbits={shared.args.wbits} and groupsize={shared.args.groupsize}.')
    except FileNotFoundError:
        print(f'No quantize config found for {model_name}.')
        return
    except JsonDecodeError:
        print(f'quantize_config.json invalid for {model_name}.')


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
            bits=shared.args.wbits,
            group_size=shared.args.groupsize
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
        print(
            f"\033[1;32;1mAuto-assiging --gpu-memory {suggestion} for your GPU to try to prevent out-of-memory errors.\nYou can manually set other values.\033[0;37;0m")

        max_memory = {0: f'{suggestion}GiB', 'cpu': f'{shared.args.cpu_memory or 99}GiB'}

    if max_memory:
        print(f'max_memory: {max_memory}')

    # dev = "cpu" if shared.args.cpu else "cuda:0"  # cpu is not supported for now

    dev = "cuda"

    print(f'Loading quantized model with AutoGPTQ from {model_file}')
    model = AutoGPTQForCausalLM.from_quantized(path_to_model,
                                               device=dev,
                                               use_triton=shared.args.autogptq_triton,
                                               use_safetensors=safetensors,
                                               quantize_config=quantize_config,
                                               model_basename=model_file.stem,
                                               trust_remote_code=shared.args.trust_remote_code,
                                               max_memory=max_memory,
                                               device_map=shared.args.autogptq_device_map)
    return model
