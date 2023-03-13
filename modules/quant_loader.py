import sys
from pathlib import Path

import accelerate
import torch

import modules.shared as shared

sys.path.insert(0, str(Path("repositories/GPTQ-for-LLaMa")))


# 4-bit LLaMA
def load_quant(model_name, model_type):
    if model_type == 'llama':
        from llama import load_quant
    elif model_type == 'opt':
        from opt import load_quant
    else:
        print("Unknown pre-quantized model type specified. Only 'llama' and 'opt' are supported")
        exit()

    path_to_model = Path(f'models/{model_name}')
    pt_model = f'{model_name}-{shared.args.gptq_bits}bit.pt'

    # Try to find the .pt both in models/ and in the subfolder
    pt_path = None
    for path in [Path(p) for p in [f"models/{pt_model}", f"{path_to_model}/{pt_model}"]]:
        if path.exists():
            pt_path = path

    if not pt_path:
        print(f"Could not find {pt_model}, exiting...")
        exit()

    model = load_quant(path_to_model, str(pt_path), shared.args.gptq_bits)

    # Multiple GPUs or GPU+CPU
    if shared.args.gpu_memory:
        max_memory = {}
        for i in range(len(shared.args.gpu_memory)):
            max_memory[i] = f"{shared.args.gpu_memory[i]}GiB"
        max_memory['cpu'] = f"{shared.args.cpu_memory or '99'}GiB"

        device_map = accelerate.infer_auto_device_map(model, max_memory=max_memory, no_split_module_classes=["LLaMADecoderLayer"])
        model = accelerate.dispatch_model(model, device_map=device_map)

    # Single GPU
    else:
        model = model.to(torch.device('cuda:0'))

    return model
