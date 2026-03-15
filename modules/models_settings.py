import functools
import json
import re
from math import floor
from pathlib import Path

import yaml

from modules import loaders, metadata_gguf, shared
from modules.logging_colors import logger
from modules.utils import resolve_model_path


def get_fallback_settings():
    return {
        'bf16': False,
        'ctx_size': 8192,
        'truncation_length': shared.settings['truncation_length'],
        'truncation_length_info': shared.settings['truncation_length'],
        'skip_special_tokens': shared.settings['skip_special_tokens'],
    }


def get_model_metadata(model):
    model_path = resolve_model_path(model)
    model_settings = {}

    # Get settings from user_data/models/config.yaml and user_data/models/config-user.yaml
    settings = shared.model_config
    for pat in settings:
        if re.match(pat.lower(), Path(model).name.lower()):
            for k in settings[pat]:
                model_settings[k] = settings[pat][k]

    path = model_path / 'config.json'
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
        path = model_path
        if path.is_file():
            model_file = path
        else:
            gguf_files = list(path.glob('*.gguf'))
            if not gguf_files:
                error_msg = f"No .gguf models found in directory: {path}"
                logger.error(error_msg)
                raise FileNotFoundError(error_msg)

            model_file = gguf_files[0]

        metadata = load_gguf_metadata_with_cache(model_file)

        for k in metadata:
            if k.endswith('.context_length'):
                model_settings['ctx_size'] = 0
                model_settings['truncation_length_info'] = metadata[k]
            elif k.endswith('.block_count'):
                model_settings['gpu_layers'] = -1
                model_settings['max_gpu_layers'] = metadata[k] + 1

        if 'tokenizer.chat_template' in metadata:
            template = metadata['tokenizer.chat_template']
            if 'tokenizer.ggml.eos_token_id' in metadata:
                eos_token = metadata['tokenizer.ggml.tokens'][metadata['tokenizer.ggml.eos_token_id']]
            else:
                eos_token = ""

            if 'tokenizer.ggml.bos_token_id' in metadata:
                bos_token = metadata['tokenizer.ggml.tokens'][metadata['tokenizer.ggml.bos_token_id']]
            else:
                bos_token = ""

            shared.bos_token = bos_token
            shared.eos_token = eos_token

            template = re.sub(r"\{\{-?\s*raise_exception\(.*?\)\s*-?\}\}", "", template, flags=re.DOTALL)
            template = re.sub(r'raise_exception\([^)]*\)', "''", template)
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
                    value = metadata[k]
                elif k in metadata.get('text_config', {}):
                    value = metadata['text_config'][k]
                else:
                    continue

                model_settings['truncation_length'] = value
                model_settings['truncation_length_info'] = value
                model_settings['ctx_size'] = min(value, 8192)
                break

            if 'torch_dtype' in metadata and metadata['torch_dtype'] == 'bfloat16':
                model_settings['bf16'] = True

    # Try to find the Jinja instruct template
    path = model_path / 'tokenizer_config.json'
    template = None

    # 1. Prioritize reading from chat_template.jinja if it exists
    jinja_path = model_path / 'chat_template.jinja'
    if jinja_path.exists():
        with open(jinja_path, 'r', encoding='utf-8') as f:
            template = f.read()

    # 2. If no .jinja file, try chat_template.json
    if template is None:
        json_template_path = model_path / 'chat_template.json'
        if json_template_path.exists():
            with open(json_template_path, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
                if 'chat_template' in json_data:
                    template = json_data['chat_template']

    # 3. Fall back to tokenizer_config.json metadata
    if path.exists():
        metadata = json.loads(open(path, 'r', encoding='utf-8').read())

        # Only read from metadata if we haven't already loaded from .jinja or .json
        if template is None and 'chat_template' in metadata:
            template = metadata['chat_template']
            if isinstance(template, list):
                template = template[0]['template']

        # 4. If a template was found from any source, process it
        if template:
            shared.bos_token = '<s>'
            shared.eos_token = '</s>'

            for k in ['eos_token', 'bos_token']:
                if k in metadata:
                    value = metadata[k]
                    if isinstance(value, dict):
                        value = value['content']

                    setattr(shared, k, value)

            template = re.sub(r"\{\{-?\s*raise_exception\(.*?\)\s*-?\}\}", "", template, flags=re.DOTALL)
            template = re.sub(r'raise_exception\([^)]*\)', "''", template)
            model_settings['instruction_template'] = 'Custom (obtained from model metadata)'
            model_settings['instruction_template_str'] = template

    if 'instruction_template' not in model_settings:
        model_settings['instruction_template'] = 'Alpaca'

    # Apply user settings from user_data/models/config-user.yaml
    settings = shared.user_config
    for pat in settings:
        if re.match(pat.lower(), Path(model).name.lower()):
            for k in settings[pat]:
                new_k = k
                if k == 'n_gpu_layers':
                    new_k = 'gpu_layers'

                model_settings[new_k] = settings[pat][k]

    # Load instruction template if defined by name rather than by value
    if model_settings['instruction_template'] != 'Custom (obtained from model metadata)':
        model_settings['instruction_template_str'] = load_instruction_template(model_settings['instruction_template'])

    return model_settings


def infer_loader(model_name, model_settings, hf_quant_method=None):
    path_to_model = resolve_model_path(model_name)
    if not path_to_model.exists():
        loader = None
    elif shared.args.portable:
        loader = 'llama.cpp'
    elif len(list(path_to_model.glob('*.gguf'))) > 0:
        loader = 'llama.cpp'
    elif re.match(r'.*\.gguf', model_name.lower()):
        loader = 'llama.cpp'
    elif hf_quant_method == 'exl3':
        loader = 'ExLlamav3'
    elif re.match(r'.*exl3', model_name.lower()):
        loader = 'ExLlamav3'
    else:
        loader = 'Transformers'

    return loader


def update_model_parameters(state, initial=False):
    '''
    UI: update the command-line arguments based on the interface values
    '''
    elements = loaders.list_model_elements()  # the names of the parameters

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
    import gradio as gr
    model_settings = get_model_metadata(model)
    if 'loader' in model_settings:
        loader = model_settings.pop('loader')
        if not (loader == 'ExLlamav3_HF' and state['loader'] == 'ExLlamav3'):
            state['loader'] = loader

    for k in model_settings:
        if k in state and k != 'gpu_layers':  # Skip gpu_layers, handle separately
            state[k] = model_settings[k]

    # Handle GPU layers and VRAM update for llama.cpp
    if state['loader'] == 'llama.cpp' and 'gpu_layers' in model_settings:
        gpu_layers = model_settings['gpu_layers']  # -1 (auto) by default, or user-saved value
        max_layers = model_settings.get('max_gpu_layers', 256)
        state['gpu_layers'] = gr.update(value=gpu_layers, maximum=max_layers)

        vram_info = update_gpu_layers_and_vram(
            state['loader'],
            model,
            gpu_layers,
            state['ctx_size'],
            state['cache_type'],
        )

        state['vram_info'] = vram_info

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

    for k in loaders.list_model_elements():
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


@functools.lru_cache(maxsize=1)
def load_gguf_metadata_with_cache(model_file):
    return metadata_gguf.load_metadata(model_file)


def get_model_size_mb(model_file: Path) -> float:
    filename = model_file.name

    # Check for multipart pattern
    match = re.match(r'(.+)-\d+-of-\d+\.gguf$', filename)

    if match:
        # It's a multipart file, find all matching parts
        base_pattern = match.group(1)
        part_files = sorted(model_file.parent.glob(f'{base_pattern}-*-of-*.gguf'))
        total_size = sum(p.stat().st_size for p in part_files)
    else:
        # Single part
        total_size = model_file.stat().st_size

    return total_size / (1024 ** 2)  # Return size in MB


def estimate_vram(gguf_file, gpu_layers, ctx_size, cache_type):
    model_file = resolve_model_path(gguf_file)
    metadata = load_gguf_metadata_with_cache(model_file)
    size_in_mb = get_model_size_mb(model_file)

    # Extract values from metadata
    n_layers = None
    n_kv_heads = None
    n_attention_heads = None  # Fallback for models without separate KV heads
    embedding_dim = None

    for key, value in metadata.items():
        if key.endswith('.block_count'):
            n_layers = value
        elif key.endswith('.attention.head_count_kv'):
            n_kv_heads = max(value) if isinstance(value, list) else value
        elif key.endswith('.attention.head_count'):
            n_attention_heads = max(value) if isinstance(value, list) else value
        elif key.endswith('.embedding_length'):
            embedding_dim = value

    if n_kv_heads is None:
        n_kv_heads = n_attention_heads

    if gpu_layers > n_layers:
        gpu_layers = n_layers

    # Convert cache_type to numeric
    if cache_type == 'q4_0':
        cache_type = 4
    elif cache_type == 'q8_0':
        cache_type = 8
    else:
        cache_type = 16

    # Derived features
    size_per_layer = size_in_mb / max(n_layers, 1e-6)
    kv_cache_factor = n_kv_heads * cache_type * ctx_size
    embedding_per_context = embedding_dim / ctx_size

    # Calculate VRAM using the model
    # Details: https://oobabooga.github.io/blog/posts/gguf-vram-formula/
    vram = (
        (size_per_layer - 17.99552795246051 + 3.148552680382576e-05 * kv_cache_factor)
        * (gpu_layers + max(0.9690636483914102, cache_type - (floor(50.77817218646521 * embedding_per_context) + 9.987899908205632)))
        + 1516.522943869404
    )

    return vram


def update_gpu_layers_and_vram(loader, model, gpu_layers, ctx_size, cache_type):
    """
    Compute the estimated VRAM usage for the given GPU layers and return
    an HTML string for the UI display.
    """
    if loader != 'llama.cpp' or model in ["None", None] or not model.endswith(".gguf") or gpu_layers < 0 or ctx_size == 0:
        return f"<div id=\"vram-info\"'>Estimated VRAM to load the model: <span class=\"value\">auto</span></div>"

    vram_usage = estimate_vram(model, gpu_layers, ctx_size, cache_type)
    return f"<div id=\"vram-info\"'>Estimated VRAM to load the model: <span class=\"value\">{vram_usage:.0f} MiB</span></div>"


def load_instruction_template(template):
    if template == 'None':
        return ''

    for filepath in [shared.user_data_dir / 'instruction-templates' / f'{template}.yaml', shared.user_data_dir / 'instruction-templates' / 'Alpaca.yaml']:
        if filepath.exists():
            break
    else:
        return ''

    with open(filepath, 'r', encoding='utf-8') as f:
        file_contents = f.read()
    data = yaml.safe_load(file_contents)
    if 'instruction_template' in data:
        return data['instruction_template']
    else:
        return _jinja_template_from_old_format(data)


def _jinja_template_from_old_format(params, verbose=False):
    MASTER_TEMPLATE = """
{%- set ns = namespace(found=false) -%}
{%- for message in messages -%}
    {%- if message['role'] == 'system' -%}
        {%- set ns.found = true -%}
    {%- endif -%}
{%- endfor -%}
{%- if not ns.found -%}
    {{- '<|PRE-SYSTEM|>' + '<|SYSTEM-MESSAGE|>' + '<|POST-SYSTEM|>' -}}
{%- endif %}
{%- for message in messages %}
    {%- if message['role'] == 'system' -%}
        {{- '<|PRE-SYSTEM|>' + message['content'] + '<|POST-SYSTEM|>' -}}
    {%- else -%}
        {%- if message['role'] == 'user' -%}
            {{-'<|PRE-USER|>' + message['content'] + '<|POST-USER|>'-}}
        {%- else -%}
            {{-'<|PRE-ASSISTANT|>' + message['content'] + '<|POST-ASSISTANT|>' -}}
        {%- endif -%}
    {%- endif -%}
{%- endfor -%}
{%- if add_generation_prompt -%}
    {{-'<|PRE-ASSISTANT-GENERATE|>'-}}
{%- endif -%}
"""

    if 'context' in params and '<|system-message|>' in params['context']:
        pre_system = params['context'].split('<|system-message|>')[0]
        post_system = params['context'].split('<|system-message|>')[1]
    else:
        pre_system = ''
        post_system = ''

    pre_user = params['turn_template'].split('<|user-message|>')[0].replace('<|user|>', params['user'])
    post_user = params['turn_template'].split('<|user-message|>')[1].split('<|bot|>')[0]

    pre_assistant = '<|bot|>' + params['turn_template'].split('<|bot-message|>')[0].split('<|bot|>')[1]
    pre_assistant = pre_assistant.replace('<|bot|>', params['bot'])
    post_assistant = params['turn_template'].split('<|bot-message|>')[1]

    def preprocess(string):
        return string.replace('\n', '\\n').replace('\'', '\\\'')

    pre_system = preprocess(pre_system)
    post_system = preprocess(post_system)
    pre_user = preprocess(pre_user)
    post_user = preprocess(post_user)
    pre_assistant = preprocess(pre_assistant)
    post_assistant = preprocess(post_assistant)

    if verbose:
        print(
            '\n',
            repr(pre_system) + '\n',
            repr(post_system) + '\n',
            repr(pre_user) + '\n',
            repr(post_user) + '\n',
            repr(pre_assistant) + '\n',
            repr(post_assistant) + '\n',
        )

    result = MASTER_TEMPLATE
    if 'system_message' in params:
        result = result.replace('<|SYSTEM-MESSAGE|>', preprocess(params['system_message']))
    else:
        result = result.replace('<|SYSTEM-MESSAGE|>', '')

    result = result.replace('<|PRE-SYSTEM|>', pre_system)
    result = result.replace('<|POST-SYSTEM|>', post_system)
    result = result.replace('<|PRE-USER|>', pre_user)
    result = result.replace('<|POST-USER|>', post_user)
    result = result.replace('<|PRE-ASSISTANT|>', pre_assistant)
    result = result.replace('<|PRE-ASSISTANT-GENERATE|>', pre_assistant.rstrip(' '))
    result = result.replace('<|POST-ASSISTANT|>', post_assistant)

    result = result.strip()

    return result
