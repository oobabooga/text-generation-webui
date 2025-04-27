import json
import re
from pathlib import Path

import yaml

from modules import chat, loaders, metadata_gguf, shared, ui


def get_fallback_settings():
    return {
        'bf16': False,
        'use_eager_attention': False,
        'ctx_size': 2048,
        'rope_freq_base': 0,
        'compress_pos_emb': 1,
        'alpha_value': 1,
        'truncation_length': shared.settings['truncation_length'],
        'truncation_length_info': shared.settings['truncation_length'],
        'skip_special_tokens': shared.settings['skip_special_tokens'],
        'custom_stopping_strings': shared.settings['custom_stopping_strings'],
    }


def get_model_metadata(model):
    model_settings = {}

    # Get settings from user_data/models/config.yaml and user_data/models/config-user.yaml
    settings = shared.model_config
    for pat in settings:
        if re.match(pat.lower(), Path(model).name.lower()):
            for k in settings[pat]:
                model_settings[k] = settings[pat][k]

    path = Path(f'{shared.args.model_dir}/{model}/config.json')
    if path.exists():
        hf_metadata = json.loads(open(path, 'r', encoding='utf-8').read())
    else:
        hf_metadata = None

    if 'loader' not in model_settings:
        quant_method = None if hf_metadata is None else hf_metadata.get("quantization_config", {}).get("quant_method", None)
        model_settings['loader'] = infer_loader(
            model,
            model_settings,
            hf_quant_method=quant_method
        )

    # GGUF metadata
    if model_settings['loader'] == 'llama.cpp':
        path = Path(f'{shared.args.model_dir}/{model}')
        if path.is_file():
            model_file = path
        else:
            model_file = list(path.glob('*.gguf'))[0]

        metadata = metadata_gguf.load_metadata(model_file)

        for k in metadata:
            if k.endswith('context_length'):
                model_settings['ctx_size'] = min(metadata[k], 8192)
                model_settings['truncation_length_info'] = metadata[k]
            elif k.endswith('rope.freq_base'):
                model_settings['rope_freq_base'] = metadata[k]
            elif k.endswith('rope.scale_linear'):
                model_settings['compress_pos_emb'] = metadata[k]
            elif k.endswith('rope.scaling.factor'):
                model_settings['compress_pos_emb'] = metadata[k]
            elif k.endswith('block_count'):
                model_settings['n_gpu_layers'] = metadata[k] + 1

        if 'tokenizer.chat_template' in metadata:
            template = metadata['tokenizer.chat_template']
            eos_token = metadata['tokenizer.ggml.tokens'][metadata['tokenizer.ggml.eos_token_id']]
            if 'tokenizer.ggml.bos_token_id' in metadata:
                bos_token = metadata['tokenizer.ggml.tokens'][metadata['tokenizer.ggml.bos_token_id']]
            else:
                bos_token = ""

            template = template.replace('eos_token', "'{}'".format(eos_token))
            template = template.replace('bos_token', "'{}'".format(bos_token))

            template = re.sub(r'raise_exception\([^)]*\)', "''", template)
            template = re.sub(r'{% if add_generation_prompt %}.*', '', template, flags=re.DOTALL)
            model_settings['instruction_template'] = 'Custom (obtained from model metadata)'
            model_settings['instruction_template_str'] = template

    else:
        # Transformers metadata
        if hf_metadata is not None:
            metadata = json.loads(open(path, 'r', encoding='utf-8').read())
            if 'pretrained_config' in metadata:
                metadata = metadata['pretrained_config']

            for k in ['max_position_embeddings', 'model_max_length', 'max_seq_len']:
                if k in metadata:
                    model_settings['truncation_length'] = metadata[k]
                    model_settings['truncation_length_info'] = metadata[k]
                    model_settings['ctx_size'] = min(metadata[k], 8192)

            if 'rope_theta' in metadata:
                model_settings['rope_freq_base'] = metadata['rope_theta']
            elif 'attn_config' in metadata and 'rope_theta' in metadata['attn_config']:
                model_settings['rope_freq_base'] = metadata['attn_config']['rope_theta']

            if 'rope_scaling' in metadata and isinstance(metadata['rope_scaling'], dict) and all(key in metadata['rope_scaling'] for key in ('type', 'factor')):
                if metadata['rope_scaling']['type'] == 'linear':
                    model_settings['compress_pos_emb'] = metadata['rope_scaling']['factor']

            # For Gemma-2
            if 'torch_dtype' in metadata and metadata['torch_dtype'] == 'bfloat16':
                model_settings['bf16'] = True

            # For Gemma-2
            if 'architectures' in metadata and isinstance(metadata['architectures'], list) and 'Gemma2ForCausalLM' in metadata['architectures']:
                model_settings['use_eager_attention'] = True

    # Try to find the Jinja instruct template
    path = Path(f'{shared.args.model_dir}/{model}') / 'tokenizer_config.json'
    if path.exists():
        metadata = json.loads(open(path, 'r', encoding='utf-8').read())
        if 'chat_template' in metadata:
            template = metadata['chat_template']
            if isinstance(template, list):
                template = template[0]['template']

            for k in ['eos_token', 'bos_token']:
                if k in metadata:
                    value = metadata[k]
                    if isinstance(value, dict):
                        value = value['content']

                    template = template.replace(k, "'{}'".format(value))

            template = re.sub(r'raise_exception\([^)]*\)', "''", template)
            template = re.sub(r'{% if add_generation_prompt %}.*', '', template, flags=re.DOTALL)
            model_settings['instruction_template'] = 'Custom (obtained from model metadata)'
            model_settings['instruction_template_str'] = template

    if 'instruction_template' not in model_settings:
        model_settings['instruction_template'] = 'Alpaca'

    # Ignore rope_freq_base if set to the default value
    if 'rope_freq_base' in model_settings and model_settings['rope_freq_base'] == 10000:
        model_settings.pop('rope_freq_base')

    # Apply user settings from user_data/models/config-user.yaml
    settings = shared.user_config
    for pat in settings:
        if re.match(pat.lower(), Path(model).name.lower()):
            for k in settings[pat]:
                model_settings[k] = settings[pat][k]

    # Load instruction template if defined by name rather than by value
    if model_settings['instruction_template'] != 'Custom (obtained from model metadata)':
        model_settings['instruction_template_str'] = chat.load_instruction_template(model_settings['instruction_template'])

    return model_settings


def infer_loader(model_name, model_settings, hf_quant_method=None):
    path_to_model = Path(f'{shared.args.model_dir}/{model_name}')
    if not path_to_model.exists():
        loader = None
    elif len(list(path_to_model.glob('*.gguf'))) > 0:
        loader = 'llama.cpp'
    elif re.match(r'.*\.gguf', model_name.lower()):
        loader = 'llama.cpp'
    elif hf_quant_method == 'exl3':
        loader = 'ExLlamav3_HF'
    elif hf_quant_method in ['exl2', 'gptq']:
        loader = 'ExLlamav2_HF'
    elif re.match(r'.*exl3', model_name.lower()):
        loader = 'ExLlamav3_HF'
    elif re.match(r'.*exl2', model_name.lower()):
        loader = 'ExLlamav2_HF'
    elif re.match(r'.*-hqq', model_name.lower()):
        return 'HQQ'
    else:
        loader = 'Transformers'

    return loader


def update_model_parameters(state, initial=False):
    '''
    UI: update the command-line arguments based on the interface values
    '''
    elements = ui.list_model_elements()  # the names of the parameters

    for i, element in enumerate(elements):
        if element not in state:
            continue

        value = state[element]
        if initial and element in shared.provided_arguments:
            continue

        if element == 'cpu_memory' and value == 0:
            value = vars(shared.args_defaults)[element]

        setattr(shared.args, element, value)


def apply_model_settings_to_state(model, state):
    '''
    UI: update the state variable with the model settings
    '''
    model_settings = get_model_metadata(model)
    if 'loader' in model_settings:
        loader = model_settings.pop('loader')

        # If the user is using an alternative loader for the same model type, let them keep using it
        if not (loader == 'ExLlamav2_HF' and state['loader'] in ['ExLlamav2']):
            state['loader'] = loader

    for k in model_settings:
        if k in state:
            state[k] = model_settings[k]

    return state


def save_model_settings(model, state):
    '''
    Save the settings for this model to user_data/models/config-user.yaml
    '''
    if model == 'None':
        yield ("Not saving the settings because no model is selected in the menu.")
        return

    user_config = shared.load_user_config()
    model_regex = Path(model).name + '$'  # For exact matches
    if model_regex not in user_config:
        user_config[model_regex] = {}

    for k in ui.list_model_elements():
        if k == 'loader' or k in loaders.loaders_and_params[state['loader']]:
            user_config[model_regex][k] = state[k]

    shared.user_config = user_config

    output = yaml.dump(user_config, sort_keys=False)
    p = Path(f'{shared.args.model_dir}/config-user.yaml')
    with open(p, 'w') as f:
        f.write(output)

    yield (f"Settings for `{model}` saved to `{p}`.")


def save_instruction_template(model, template):
    '''
    Similar to the function above, but it saves only the instruction template.
    '''
    if model == 'None':
        yield ("Not saving the template because no model is selected in the menu.")
        return

    user_config = shared.load_user_config()
    model_regex = Path(model).name + '$'  # For exact matches
    if model_regex not in user_config:
        user_config[model_regex] = {}

    if template == 'None':
        user_config[model_regex].pop('instruction_template', None)
    else:
        user_config[model_regex]['instruction_template'] = template

    shared.user_config = user_config

    output = yaml.dump(user_config, sort_keys=False)
    p = Path(f'{shared.args.model_dir}/config-user.yaml')
    with open(p, 'w') as f:
        f.write(output)

    if template == 'None':
        yield (f"Instruction template for `{model}` unset in `{p}`, as the value for template was `{template}`.")
    else:
        yield (f"Instruction template for `{model}` saved to `{p}` as `{template}`.")
