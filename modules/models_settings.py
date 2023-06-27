import re
from pathlib import Path

import yaml

from modules import shared, ui


def get_model_settings_from_yamls(model):
    settings = shared.model_config
    model_settings = {}
    for pat in settings:
        if re.match(pat.lower(), model.lower()):
            for k in settings[pat]:
                model_settings[k] = settings[pat][k]

    return model_settings


def infer_loader(model_name):
    path_to_model = Path(f'{shared.args.model_dir}/{model_name}')
    model_settings = get_model_settings_from_yamls(model_name)
    if not path_to_model.exists():
        loader = None
    elif Path(f'{shared.args.model_dir}/{model_name}/quantize_config.json').exists() or ('wbits' in model_settings and type(model_settings['wbits']) is int and model_settings['wbits'] > 0):
        loader = 'AutoGPTQ'
    elif len(list(path_to_model.glob('*ggml*.bin'))) > 0:
        loader = 'llama.cpp'
    elif re.match('.*ggml.*\.bin', model_name.lower()):
        loader = 'llama.cpp'
    elif re.match('.*rwkv.*\.pth', model_name.lower()):
        loader = 'RWKV'
    elif shared.args.flexgen:
        loader = 'FlexGen'
    else:
        loader = 'Transformers'

    return loader


# UI: update the command-line arguments based on the interface values
def update_model_parameters(state, initial=False):
    elements = ui.list_model_elements()  # the names of the parameters
    gpu_memories = []

    for i, element in enumerate(elements):
        if element not in state:
            continue

        value = state[element]
        if element.startswith('gpu_memory'):
            gpu_memories.append(value)
            continue

        if initial and vars(shared.args)[element] != vars(shared.args_defaults)[element]:
            continue

        # Setting null defaults
        if element in ['wbits', 'groupsize', 'model_type'] and value == 'None':
            value = vars(shared.args_defaults)[element]
        elif element in ['cpu_memory'] and value == 0:
            value = vars(shared.args_defaults)[element]

        # Making some simple conversions
        if element in ['wbits', 'groupsize', 'pre_layer']:
            value = int(value)
        elif element == 'cpu_memory' and value is not None:
            value = f"{value}MiB"

        if element in ['pre_layer']:
            value = [value] if value > 0 else None

        setattr(shared.args, element, value)

    found_positive = False
    for i in gpu_memories:
        if i > 0:
            found_positive = True
            break

    if not (initial and vars(shared.args)['gpu_memory'] != vars(shared.args_defaults)['gpu_memory']):
        if found_positive:
            shared.args.gpu_memory = [f"{i}MiB" for i in gpu_memories]
        else:
            shared.args.gpu_memory = None


# UI: update the state variable with the model settings
def apply_model_settings_to_state(model, state):
    model_settings = get_model_settings_from_yamls(model)
    if 'loader' not in model_settings:
        loader = infer_loader(model)
        if 'wbits' in model_settings and type(model_settings['wbits']) is int and model_settings['wbits'] > 0:
            loader = 'AutoGPTQ'

        # If the user is using an alternative GPTQ loader, let them keep using it
        if not (loader == 'AutoGPTQ' and state['loader'] in ['GPTQ-for-LLaMa', 'ExLlama', 'ExLlama_HF']):
            state['loader'] = loader

    for k in model_settings:
        if k in state:
            state[k] = model_settings[k]

    return state


# Save the settings for this model to models/config-user.yaml
def save_model_settings(model, state):
    if model == 'None':
        yield ("Not saving the settings because no model is loaded.")
        return

    with Path(f'{shared.args.model_dir}/config-user.yaml') as p:
        if p.exists():
            user_config = yaml.safe_load(open(p, 'r').read())
        else:
            user_config = {}

        model_regex = model + '$'  # For exact matches
        for _dict in [user_config, shared.model_config]:
            if model_regex not in _dict:
                _dict[model_regex] = {}

        if model_regex not in user_config:
            user_config[model_regex] = {}

        for k in ui.list_model_elements():
            user_config[model_regex][k] = state[k]
            shared.model_config[model_regex][k] = state[k]

        with open(p, 'w') as f:
            f.write(yaml.dump(user_config, sort_keys=False))

        yield (f"Settings for {model} saved to {p}")
