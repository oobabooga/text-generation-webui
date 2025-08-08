import functools
import json
import re
import subprocess
from math import floor
from pathlib import Path

import gradio as gr
import yaml

from modules import chat, loaders, metadata_gguf, shared, ui
from modules.logging_colors import logger


def get_fallback_settings():
    return {
        'bf16': False,
        'ctx_size': 2048,
        'rope_freq_base': 0,
        'compress_pos_emb': 1,
        'alpha_value': 1,
        'truncation_length': shared.settings['truncation_length'],
        'truncation_length_info': shared.settings['truncation_length'],
        'skip_special_tokens': shared.settings['skip_special_tokens'],
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
            gguf_files = list(path.glob('*.gguf'))
            if not gguf_files:
                error_msg = f"No .gguf models found in directory: {path}"
                logger.error(error_msg)
                raise FileNotFoundError(error_msg)

            model_file = gguf_files[0]

        metadata = load_gguf_metadata_with_cache(model_file)

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
            elif k.endswith('.block_count'):
                model_settings['gpu_layers'] = metadata[k] + 1
                model_settings['max_gpu_layers'] = metadata[k] + 1

        if 'tokenizer.chat_template' in metadata:
            template = metadata['tokenizer.chat_template']
            eos_token = metadata['tokenizer.ggml.tokens'][metadata['tokenizer.ggml.eos_token_id']]
            if 'tokenizer.ggml.bos_token_id' in metadata:
                bos_token = metadata['tokenizer.ggml.tokens'][metadata['tokenizer.ggml.bos_token_id']]
            else:
                bos_token = ""

            template = template.replace('eos_token', "'{}'".format(eos_token))
            template = template.replace('bos_token', "'{}'".format(bos_token))

            template = re.sub(r"\{\{-?\s*raise_exception\(.*?\)\s*-?\}\}", "", template, flags=re.DOTALL)
            template = re.sub(r'raise_exception\([^)]*\)', "''", template)
            template = re.sub(r'{% if add_generation_prompt %}.*', '', template, flags=re.DOTALL)
            template = re.sub(r'elif loop\.last and not add_generation_prompt', 'elif False', template)  # Handle GPT-OSS
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

            if 'torch_dtype' in metadata and metadata['torch_dtype'] == 'bfloat16':
                model_settings['bf16'] = True

    # Try to find the Jinja instruct template
    path = Path(f'{shared.args.model_dir}/{model}') / 'tokenizer_config.json'
    template = None

    # 1. Prioritize reading from chat_template.jinja if it exists
    jinja_path = Path(f'{shared.args.model_dir}/{model}') / 'chat_template.jinja'
    if jinja_path.exists():
        with open(jinja_path, 'r', encoding='utf-8') as f:
            template = f.read()

    if path.exists():
        metadata = json.loads(open(path, 'r', encoding='utf-8').read())

        # 2. Only read from metadata if we haven't already loaded from .jinja
        if template is None and 'chat_template' in metadata:
            template = metadata['chat_template']
            if isinstance(template, list):
                template = template[0]['template']

        # 3. If a template was found from either source, process it
        if template:
            for k in ['eos_token', 'bos_token']:
                if k in metadata:
                    value = metadata[k]
                    if isinstance(value, dict):
                        value = value['content']

                    template = template.replace(k, "'{}'".format(value))

            template = re.sub(r"\{\{-?\s*raise_exception\(.*?\)\s*-?\}\}", "", template, flags=re.DOTALL)
            template = re.sub(r'raise_exception\([^)]*\)', "''", template)
            template = re.sub(r'{% if add_generation_prompt %}.*', '', template, flags=re.DOTALL)
            template = re.sub(r'elif loop\.last and not add_generation_prompt', 'elif False', template)  # Handle GPT-OSS
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
                new_k = k
                if k == 'n_gpu_layers':
                    new_k = 'gpu_layers'

                model_settings[new_k] = settings[pat][k]

    # Load instruction template if defined by name rather than by value
    if model_settings['instruction_template'] != 'Custom (obtained from model metadata)':
        model_settings['instruction_template_str'] = chat.load_instruction_template(model_settings['instruction_template'])

    return model_settings


def infer_loader(model_name, model_settings, hf_quant_method=None):
    path_to_model = Path(f'{shared.args.model_dir}/{model_name}')
    if not path_to_model.exists():
        loader = None
    elif shared.args.portable:
        loader = 'llama.cpp'
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
        if not (loader == 'ExLlamav2_HF' and state['loader'] in ['ExLlamav2']):
            state['loader'] = loader

    for k in model_settings:
        if k in state and k != 'gpu_layers':  # Skip gpu_layers, handle separately
            state[k] = model_settings[k]

    # Handle GPU layers and VRAM update for llama.cpp
    if state['loader'] == 'llama.cpp' and 'gpu_layers' in model_settings:
        vram_info, gpu_layers_update = update_gpu_layers_and_vram(
            state['loader'],
            model,
            model_settings['gpu_layers'],
            state['ctx_size'],
            state['cache_type'],
            auto_adjust=True
        )

        state['gpu_layers'] = gpu_layers_update
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
    model_file = Path(f'{shared.args.model_dir}/{gguf_file}')
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


def get_nvidia_vram(return_free=True):
    """
    Calculates VRAM statistics across all NVIDIA GPUs by parsing nvidia-smi output.

    Args:
        return_free (bool): If True, returns free VRAM. If False, returns total VRAM.

    Returns:
        int: Either the total free VRAM or total VRAM in MiB summed across all detected NVIDIA GPUs.
             Returns -1 if nvidia-smi command fails (not found, error, etc.).
             Returns 0 if nvidia-smi succeeds but no GPU memory info found.
    """
    try:
        # Execute nvidia-smi command
        result = subprocess.run(
            ['nvidia-smi'],
            capture_output=True,
            text=True,
            check=False
        )

        # Check if nvidia-smi returned an error
        if result.returncode != 0:
            return -1

        # Parse the output for memory usage patterns
        output = result.stdout

        # Find memory usage like "XXXXMiB / YYYYMiB"
        # Captures used and total memory for each GPU
        matches = re.findall(r"(\d+)\s*MiB\s*/\s*(\d+)\s*MiB", output)

        if not matches:
            # No GPUs found in expected format
            return 0

        total_vram_mib = 0
        total_free_vram_mib = 0

        for used_mem_str, total_mem_str in matches:
            try:
                used_mib = int(used_mem_str)
                total_mib = int(total_mem_str)
                total_vram_mib += total_mib
                total_free_vram_mib += (total_mib - used_mib)
            except ValueError:
                # Skip malformed entries
                pass

        # Return either free or total VRAM based on the flag
        return total_free_vram_mib if return_free else total_vram_mib

    except FileNotFoundError:
        # nvidia-smi not found (likely no NVIDIA drivers installed)
        return -1
    except Exception:
        # Handle any other unexpected exceptions
        return -1


def update_gpu_layers_and_vram(loader, model, gpu_layers, ctx_size, cache_type, auto_adjust=False, for_ui=True):
    """
    Unified function to handle GPU layers and VRAM updates.

    Args:
        for_ui: If True, returns Gradio updates. If False, returns raw values.

    Returns:
        - If for_ui=True: (vram_info_update, gpu_layers_update) or just vram_info_update
        - If for_ui=False: (vram_usage, adjusted_layers) or just vram_usage
    """
    if loader != 'llama.cpp' or model in ["None", None] or not model.endswith(".gguf"):
        vram_info = "<div id=\"vram-info\"'>Estimated VRAM to load the model:</div>"
        if for_ui:
            return (vram_info, gr.update()) if auto_adjust else vram_info
        else:
            return (0, gpu_layers) if auto_adjust else 0

    # Get model settings including user preferences
    model_settings = get_model_metadata(model)

    current_layers = gpu_layers
    max_layers = model_settings.get('max_gpu_layers', 256)

    if auto_adjust:
        # Check if this is a user-saved setting
        user_config = shared.user_config
        model_regex = Path(model).name + '$'
        has_user_setting = model_regex in user_config and 'gpu_layers' in user_config[model_regex]

        if not has_user_setting:
            # No user setting, auto-adjust from the maximum
            current_layers = max_layers  # Start from max

            # Auto-adjust based on available/total VRAM
            # If a model is loaded and it's for the UI, use the total VRAM to avoid confusion
            return_free = False if (for_ui and shared.model_name not in [None, 'None']) else True
            available_vram = get_nvidia_vram(return_free=return_free)
            if available_vram > 0:
                tolerance = 577
                while current_layers > 0 and estimate_vram(model, current_layers, ctx_size, cache_type) > available_vram - tolerance:
                    current_layers -= 1

    # Calculate VRAM with current layers
    vram_usage = estimate_vram(model, current_layers, ctx_size, cache_type)

    if for_ui:
        vram_info = f"<div id=\"vram-info\"'>Estimated VRAM to load the model: <span class=\"value\">{vram_usage:.0f} MiB</span></div>"
        if auto_adjust:
            return vram_info, gr.update(value=current_layers, maximum=max_layers)
        else:
            return vram_info
    else:
        if auto_adjust:
            return vram_usage, current_layers
        else:
            return vram_usage
