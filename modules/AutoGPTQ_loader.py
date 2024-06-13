from pathlib import Path

from accelerate.utils import is_xpu_available
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
        'device': "xpu:0" if is_xpu_available() else "cuda:0" if not shared.args.cpu else "cpu",
        'use_triton': shared.args.triton,
        'inject_fused_attention': False,
        'inject_fused_mlp': not shared.args.no_inject_fused_mlp,
        'use_safetensors': use_safetensors,
        'trust_remote_code': shared.args.trust_remote_code,
        'max_memory': get_max_memory_dict(),
        'quantize_config': quantize_config,
        'use_cuda_fp16': not shared.args.no_use_cuda_fp16,
        'disable_exllama': shared.args.disable_exllama,
        'disable_exllamav2': shared.args.disable_exllamav2,
    }

    logger.info(f"The AutoGPTQ params are: {params}")
    model = AutoGPTQForCausalLM.from_quantized(path_to_model, **params)

    # These lines fix the multimodal extension when used with AutoGPTQ
    if hasattr(model, 'model'):
        if not hasattr(model, 'dtype'):
            if hasattr(model.model, 'dtype'):
                model.dtype = model.model.dtype

        if hasattr(model.model, 'model') and hasattr(model.model.model, 'embed_tokens'):
            if not hasattr(model, 'embed_tokens'):
                model.embed_tokens = model.model.model.embed_tokens

            if not hasattr(model.model, 'embed_tokens'):
                model.model.embed_tokens = model.model.model.embed_tokens

    return model
