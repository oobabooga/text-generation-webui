import re
import sys
from pathlib import Path

import accelerate
import torch

import modules.shared as shared

sys.path.insert(0, str(Path("repositories/GPTQ-for-LLaMa")))
import llama
import llama_inference_offload
import opt


def load_quantized(model_name):
    if not shared.args.model_type:
        # Try to determine model type from model name
        model_type = model_name.split('-')[0].lower()
        if model_type not in ('llama', 'opt'):
            print("Can't determine model type from model name. Please specify it manually using --gptq-model-type "
                  "argument")
            exit()
    else:
        model_type = shared.args.model_type.lower()

    if model_type == 'llama':
        if not shared.args.pre_layer:
            load_quant = llama.load_quant
        else:
            load_quant = llama_inference_offload.load_quant
    elif model_type == 'opt':
        load_quant = opt.load_quant
    else:
        print("Unknown pre-quantized model type specified. Only 'llama' and 'opt' are supported")
        exit()

    path_to_model = Path(f'models/{model_name}')
    if path_to_model.name.lower().startswith('llama-7b'):
        pt_model = f'llama-7b-{shared.args.wbits}bit.pt'
    elif path_to_model.name.lower().startswith('llama-13b'):
        pt_model = f'llama-13b-{shared.args.wbits}bit.pt'
    elif path_to_model.name.lower().startswith('llama-30b'):
        pt_model = f'llama-30b-{shared.args.wbits}bit.pt'
    elif path_to_model.name.lower().startswith('llama-65b'):
        pt_model = f'llama-65b-{shared.args.wbits}bit.pt'
    else:
        pt_model = f'{model_name}-{shared.args.wbits}bit.pt'

    # Try to find the .pt both in models/ and in the subfolder
    pt_path = None
    for path in [Path(p) for p in [f"models/{pt_model}", f"{path_to_model}/{pt_model}"]]:
        if path.exists():
            pt_path = path

    if not pt_path:
        print(f"Could not find {pt_model}, exiting...")
        exit()

    # qwopqwop200's offload
    if shared.args.pre_layer:
        model = load_quant(str(path_to_model), str(pt_path), shared.args.wbits, shared.args.pre_layer)
    else:
        model = load_quant(str(path_to_model), str(pt_path), shared.args.wbits)

        # accelerate offload (doesn't work properly)
        if shared.args.gpu_memory:
            memory_map = list(map(lambda x : x.strip(), shared.args.gpu_memory))
            max_cpu_memory = shared.args.cpu_memory.strip() if shared.args.cpu_memory is not None else '99GiB'
            max_memory = {}
            for i in range(len(memory_map)):
                max_memory[i] = f'{memory_map[i]}GiB' if not re.match('.*ib$', memory_map[i].lower()) else memory_map[i]
            max_memory['cpu'] = max_cpu_memory

            device_map = accelerate.infer_auto_device_map(model, max_memory=max_memory, no_split_module_classes=["LlamaDecoderLayer"])
            print("Using the following device map for the 4-bit model:", device_map)
            # https://huggingface.co/docs/accelerate/package_reference/big_modeling#accelerate.dispatch_model
            model = accelerate.dispatch_model(model, device_map=device_map, offload_buffers=True)

        # No offload
        elif not shared.args.cpu:
            model = model.to(torch.device('cuda:0'))

    return model
