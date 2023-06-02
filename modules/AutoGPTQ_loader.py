from pathlib import Path

from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig

import modules.shared as shared
from modules.logging_colors import logger
from modules.models import get_max_memory_dict


def load_quantized(model_name):
    path_to_model = Path(f'{shared.args.model_dir}/{model_name}')
    pt_path = None

    # Find the model checkpoint
    if shared.args.checkpoint:
        pt_path = Path(shared.args.checkpoint)
    else:
        for ext in ['.safetensors', '.pt', '.bin']:
            found = list(path_to_model.glob(f"*{ext}"))
            if len(found) > 0:
                if len(found) > 1:
                    logger.warning(f'More than one {ext} model has been found. The last one will be selected. It could be wrong.')

                pt_path = found[-1]
                break

    if pt_path is None:
        logger.error("The model could not be loaded because its checkpoint file in .bin/.pt/.safetensors format could not be located.")
        return

    use_safetensors = pt_path.suffix == '.safetensors'
    if not (path_to_model / "quantize_config.json").exists():
        quantize_config = BaseQuantizeConfig(
            bits=bits if (bits := shared.args.wbits) > 0 else 4,
            group_size=gs if (gs := shared.args.groupsize) > 0 else -1,
            desc_act=shared.args.desc_act
        )
    else:
        quantize_config = None

    # Define the params for AutoGPTQForCausalLM.from_quantized
    params = {
        'model_basename': pt_path.stem,
        'device': "cuda:0" if not shared.args.cpu else "cpu",
        'use_triton': shared.args.triton,
        'use_safetensors': use_safetensors,
        'trust_remote_code': shared.args.trust_remote_code,
        'max_memory': get_max_memory_dict(),
        'quantize_config': quantize_config
    }

    logger.info(f"The AutoGPTQ params are: {params}")
    model = AutoGPTQForCausalLM.from_quantized(path_to_model, **params)
    return model
