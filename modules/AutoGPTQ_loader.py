from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
from pathlib import Path

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


def load_quantized(model_name):
    model_file = find_quantized_model_file(model_name)
    if model_file is None:
        raise FileNotFoundError(f'No quantized model found for {model_name}')

    safetensors = model_file.suffix == '.safetensors'

    path_to_model = Path(f'{shared.args.model_dir}/{model_name}')
    quantize_config = None
    if not (path_to_model / 'quantize_config.json').exists():
        quantize_config = BaseQuantizeConfig(
            bits=shared.args.wbits,
            group_size=shared.args.groupsize
        )

    dev = "cpu" if shared.args.cpu else "cuda:0"

    print(f'Loading quantized model with AutoGPTQ from {model_file}')
    model = AutoGPTQForCausalLM.from_quantized(path_to_model,
                                               device=dev,
                                               use_triton=shared.args.autogptq_triton,
                                               use_safetensors=safetensors,
                                               quantize_config=quantize_config,
                                               model_basename=model_file.stem,
                                               trust_remote_code=shared.args.trust_remote_code)
    return model
