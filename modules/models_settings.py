import json
import re
from pathlib import Path

import yaml

from modules import loaders, metadata_gguf, shared, ui


def get_fallback_settings():
    return {
        'wbits': 'None',
        'groupsize': 'None',
        'desc_act': False,
        'model_type': 'None',
        'max_seq_len': 2048,
        'n_ctx': 2048,
        'rope_freq_base': 0,
        'compress_pos_emb': 1,
        'truncation_length': shared.settings['truncation_length'],
        'skip_special_tokens': shared.settings['skip_special_tokens'],
        'custom_stopping_strings': shared.settings['custom_stopping_strings'],
    }


def get_model_metadata(model):
    model_settings = {}

    # Get settings from models/config.yaml and models/config-user.yaml
    settings = shared.model_config
    for pat in settings:
        if re.match(pat.lower(), model.lower()):
            for k in settings[pat]:
                model_settings[k] = settings[pat][k]

    if 'loader' not in model_settings:
        loader = infer_loader(model, model_settings)
        if 'wbits' in model_settings and type(model_settings['wbits']) is int and model_settings['wbits'] > 0:
            loader = 'AutoGPTQ'

        model_settings['loader'] = loader

    # Read GGUF metadata
    if model_settings['loader'] in ['llama.cpp', 'llamacpp_HF', 'ctransformers']:
        path = Path(f'{shared.args.model_dir}/{model}')
        if path.is_file():
            model_file = path
        else:
            model_file = list(path.glob('*.gguf'))[0]

        metadata = metadata_gguf.load_metadata(model_file)
        if 'llama.context_length' in metadata:
            model_settings['n_ctx'] = metadata['llama.context_length']
        if 'llama.rope.scale_linear' in metadata:
            model_settings['compress_pos_emb'] = metadata['llama.rope.scale_linear']
        if 'llama.rope.freq_base' in metadata:
            model_settings['rope_freq_base'] = metadata['llama.rope.freq_base']

    else:
        # Read transformers metadata
        path = Path(f'{shared.args.model_dir}/{model}/config.json')
        if path.exists():
            metadata = json.loads(open(path, 'r').read())
            if 'max_position_embeddings' in metadata:
                model_settings['truncation_length'] = metadata['max_position_embeddings']
                model_settings['max_seq_len'] = metadata['max_position_embeddings']

            if 'rope_theta' in metadata:
                model_settings['rope_freq_base'] = metadata['rope_theta']

            if 'rope_scaling' in metadata and type(metadata['rope_scaling']) is dict and all(key in metadata['rope_scaling'] for key in ('type', 'factor')):
                if metadata['rope_scaling']['type'] == 'linear':
                    model_settings['compress_pos_emb'] = metadata['rope_scaling']['factor']

            if 'quantization_config' in metadata:
                if 'bits' in metadata['quantization_config']:
                    model_settings['wbits'] = metadata['quantization_config']['bits']
                if 'group_size' in metadata['quantization_config']:
                    model_settings['groupsize'] = metadata['quantization_config']['group_size']
                if 'desc_act' in metadata['quantization_config']:
                    model_settings['desc_act'] = metadata['quantization_config']['desc_act']

        # Read AutoGPTQ metadata
        path = Path(f'{shared.args.model_dir}/{model}/quantize_config.json')
        if path.exists():
            metadata = json.loads(open(path, 'r').read())
            if 'bits' in metadata:
                model_settings['wbits'] = metadata['bits']
            if 'group_size' in metadata:
                model_settings['groupsize'] = metadata['group_size']
            if 'desc_act' in metadata:
                model_settings['desc_act'] = metadata['desc_act']

    # Ignore rope_freq_base if set to the default value
    if 'rope_freq_base' in model_settings and model_settings['rope_freq_base'] == 10000:
        model_settings.pop('rope_freq_base')

    # Apply user settings from models/config-user.yaml
    settings = shared.user_config
    for pat in settings:
        if re.match(pat.lower(), model.lower()):
            for k in settings[pat]:
                model_settings[k] = settings[pat][k]

    return model_settings


def infer_loader(model_name, model_settings):
    path_to_model = Path(f'{shared.args.model_dir}/{model_name}')
    if not path_to_model.exists():
        loader = None
    elif (path_to_model / 'quantize_config.json').exists() or ('wbits' in model_settings and type(model_settings['wbits']) is int and model_settings['wbits'] > 0):
        loader = 'ExLlama_HF'
    elif (path_to_model / 'quant_config.json').exists() or re.match(r'.*-awq', model_name.lower()):
        loader = 'AutoAWQ'
    elif len(list(path_to_model.glob('*.gguf'))) > 0:
        loader = 'llama.cpp'
    elif re.match(r'.*\.gguf', model_name.lower()):
        loader = 'llama.cpp'
    elif re.match(r'.*rwkv.*\.pth', model_name.lower()):
        loader = 'RWKV'
    elif re.match(r'.*exl2', model_name.lower()):
        loader = 'ExLlamav2_HF'
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

        if initial and element in shared.provided_arguments:
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
    model_settings = get_model_metadata(model)
    if 'loader' in model_settings:
        loader = model_settings.pop('loader')

        # If the user is using an alternative loader for the same model type, let them keep using it
        if not (loader == 'AutoGPTQ' and state['loader'] in ['GPTQ-for-LLaMa', 'ExLlama', 'ExLlama_HF', 'ExLlamav2', 'ExLlamav2_HF']) and not (loader == 'llama.cpp' and state['loader'] in ['llamacpp_HF', 'ctransformers']):
            state['loader'] = loader

    for k in model_settings:
        if k in state:
            if k in ['wbits', 'groupsize']:
                state[k] = str(model_settings[k])
            else:
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
        if model_regex not in user_config:
            user_config[model_regex] = {}

        for k in ui.list_model_elements():
            if k == 'loader' or k in loaders.loaders_and_params[state['loader']]:
                user_config[model_regex][k] = state[k]

        shared.user_config = user_config

        output = yaml.dump(user_config, sort_keys=False)
        with open(p, 'w') as f:
            f.write(output)

        yield (f"Settings for `{model}` saved to `{p}`.")
