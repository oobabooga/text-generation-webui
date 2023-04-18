import modules.shared as shared
from pathlib import Path

# Used to locate the .pt/.safetensors quantized file
def find_quantized_model_file(model_name):
    path_to_model = Path(f'{shared.args.model_dir}/{model_name}')
    pt_path = None
    priority_name_list = [
        Path(f'{shared.args.model_dir}/{model_name}{hyphen}{shared.args.wbits}bit{group}{ext}')
        for group in ([f'-{shared.args.groupsize}g', ''] if shared.args.groupsize > 0 else [''])
        for ext in ['.safetensors', '.pt']
        for hyphen in ['-', f'/{model_name}-', '/']
    ]
    for path in priority_name_list:
        if path.exists():
            pt_path = path
            break

    # If the model hasn't been found with a well-behaved name, pick the last .pt
    # or the last .safetensors found in its folder as a last resort
    if not pt_path:
        found_pts = list(path_to_model.glob("*.pt"))
        found_safetensors = list(path_to_model.glob("*.safetensors"))
        pt_path = None

        if len(found_pts) > 0:
            if len(found_pts) > 1:
                print('Warning: more than one .pt model has been found. The last one will be selected. It could be wrong.')
            pt_path = found_pts[-1]
        elif len(found_safetensors) > 0:
            if len(found_pts) > 1:
                print('Warning: more than one .safetensors model has been found. The last one will be selected. It could be wrong.')
            pt_path = found_safetensors[-1]

    return pt_path
