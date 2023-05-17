import logging
from pathlib import Path

from auto_gptq import AutoGPTQForCausalLM

import modules.shared as shared
from modules.models import get_max_memory_dict


def load_quantized(model_name):
    path_to_model = Path(f'{shared.args.model_dir}/{model_name}')
    pt_path = None
    use_safetensors = False

    # Find the model checkpoint
    found_pts = list(path_to_model.glob("*.pt"))
    found_safetensors = list(path_to_model.glob("*.safetensors"))
    if len(found_safetensors) > 0:
        if len(found_safetensors) > 1:
            logging.warning('More than one .safetensors model has been found. The last one will be selected. It could be wrong.')

        use_safetensors = True
        pt_path = found_safetensors[-1]
    elif len(found_pts) > 0:
        if len(found_pts) > 1:
            logging.warning('More than one .pt model has been found. The last one will be selected. It could be wrong.')

        pt_path = found_pts[-1]

    # Define the params for AutoGPTQForCausalLM.from_quantized
    params = {
        'model_basename': pt_path.stem,
        'device': "cuda:0" if not shared.args.cpu else "cpu",
        'use_triton': shared.args.triton,
        'use_safetensors': use_safetensors,
        'max_memory': get_max_memory_dict()
    }

    logging.warning(f"The AutoGPTQ params are: {params}")
    model = AutoGPTQForCausalLM.from_quantized(path_to_model, **params)
    return model
