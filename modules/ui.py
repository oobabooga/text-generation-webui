import copy
from pathlib import Path

import gradio as gr
import torch
import yaml
from transformers import is_torch_xpu_available

import extensions
from modules import shared

with open(Path(__file__).resolve().parent / '../css/NotoSans/stylesheet.css', 'r') as f:
    css = f.read()
with open(Path(__file__).resolve().parent / '../css/main.css', 'r') as f:
    css += f.read()
with open(Path(__file__).resolve().parent / '../css/katex/katex.min.css', 'r') as f:
    css += f.read()
with open(Path(__file__).resolve().parent / '../css/highlightjs/highlightjs-copy.min.css', 'r') as f:
    css += f.read()
with open(Path(__file__).resolve().parent / '../js/main.js', 'r') as f:
    js = f.read()
with open(Path(__file__).resolve().parent / '../js/save_files.js', 'r') as f:
    save_files_js = f.read()
with open(Path(__file__).resolve().parent / '../js/switch_tabs.js', 'r') as f:
    switch_tabs_js = f.read()
with open(Path(__file__).resolve().parent / '../js/show_controls.js', 'r') as f:
    show_controls_js = f.read()
with open(Path(__file__).resolve().parent / '../js/update_big_picture.js', 'r') as f:
    update_big_picture_js = f.read()
with open(Path(__file__).resolve().parent / '../js/dark_theme.js', 'r') as f:
    dark_theme_js = f.read()

refresh_symbol = '🔄'
delete_symbol = '🗑️'
save_symbol = '💾'

theme = gr.themes.Default(
    font=['Noto Sans', 'Helvetica', 'ui-sans-serif', 'system-ui', 'sans-serif'],
    font_mono=['IBM Plex Mono', 'ui-monospace', 'Consolas', 'monospace'],
).set(
    border_color_primary='#c5c5d2',
    button_large_padding='6px 12px',
    body_text_color_subdued='#484848',
    background_fill_secondary='#eaeaea',
    background_fill_primary='var(--neutral-50)',
    body_background_fill="white",
    block_background_fill="#f4f4f4",
    body_text_color="#333",
    button_secondary_background_fill="#f4f4f4",
    button_secondary_border_color="var(--border-color-primary)"
)

if Path("notification.mp3").exists():
    audio_notification_js = "document.querySelector('#audio_notification audio')?.play();"
else:
    audio_notification_js = ""


def list_model_elements():
    elements = [
        'loader',
        'filter_by_loader',
        'cpu_memory',
        'auto_devices',
        'disk',
        'cpu',
        'bf16',
        'load_in_8bit',
        'trust_remote_code',
        'no_use_fast',
        'use_flash_attention_2',
        'use_eager_attention',
        'load_in_4bit',
        'compute_dtype',
        'quant_type',
        'use_double_quant',
        'wbits',
        'groupsize',
        'triton',
        'desc_act',
        'no_inject_fused_mlp',
        'no_use_cuda_fp16',
        'disable_exllama',
        'disable_exllamav2',
        'cfg_cache',
        'no_flash_attn',
        'no_xformers',
        'no_sdpa',
        'num_experts_per_token',
        'cache_8bit',
        'cache_4bit',
        'autosplit',
        'enable_tp',
        'threads',
        'threads_batch',
        'n_batch',
        'no_mmap',
        'mlock',
        'no_mul_mat_q',
        'n_gpu_layers',
        'tensor_split',
        'n_ctx',
        'gpu_split',
        'max_seq_len',
        'compress_pos_emb',
        'alpha_value',
        'rope_freq_base',
        'numa',
        'logits_all',
        'no_offload_kqv',
        'row_split',
        'tensorcores',
        'flash_attn',
        'streaming_llm',
        'attention_sink_size',
        'hqq_backend',
        'cpp_runner',
    ]

    if is_torch_xpu_available():
        for i in range(torch.xpu.device_count()):
            elements.append(f'gpu_memory_{i}')
    else:
        for i in range(torch.cuda.device_count()):
            elements.append(f'gpu_memory_{i}')

    return elements


def list_interface_input_elements():
    elements = [
        'max_new_tokens',
        'auto_max_new_tokens',
        'max_tokens_second',
        'max_updates_second',
        'prompt_lookup_num_tokens',
        'seed',
        'temperature',
        'temperature_last',
        'dynamic_temperature',
        'dynatemp_low',
        'dynatemp_high',
        'dynatemp_exponent',
        'smoothing_factor',
        'smoothing_curve',
        'top_p',
        'min_p',
        'top_k',
        'typical_p',
        'epsilon_cutoff',
        'eta_cutoff',
        'repetition_penalty',
        'presence_penalty',
        'frequency_penalty',
        'repetition_penalty_range',
        'encoder_repetition_penalty',
        'no_repeat_ngram_size',
        'dry_multiplier',
        'dry_base',
        'dry_allowed_length',
        'dry_sequence_breakers',
        'xtc_threshold',
        'xtc_probability',
        'do_sample',
        'penalty_alpha',
        'mirostat_mode',
        'mirostat_tau',
        'mirostat_eta',
        'grammar_string',
        'negative_prompt',
        'guidance_scale',
        'add_bos_token',
        'ban_eos_token',
        'custom_token_bans',
        'sampler_priority',
        'truncation_length',
        'custom_stopping_strings',
        'skip_special_tokens',
        'stream',
        'tfs',
        'top_a',
    ]

    # Chat elements
    elements += [
        'textbox',
        'start_with',
        'character_menu',
        'history',
        'unique_id',
        'name1',
        'user_bio',
        'name2',
        'greeting',
        'context',
        'mode',
        'custom_system_message',
        'instruction_template_str',
        'chat_template_str',
        'chat_style',
        'chat-instruct_command',
    ]

    # Notebook/default elements
    elements += [
        'textbox-notebook',
        'textbox-default',
        'output_textbox',
        'prompt_menu-default',
        'prompt_menu-notebook',
    ]

    # Model elements
    elements += list_model_elements()

    return elements


def gather_interface_values(*args):
    interface_elements = list_interface_input_elements()

    output = {}
    for element, value in zip(interface_elements, args):
        output[element] = value

    if not shared.args.multi_user:
        shared.persistent_interface_state = output

    return output


def apply_interface_values(state, use_persistent=False):
    if use_persistent:
        state = shared.persistent_interface_state
        if 'textbox-default' in state:
            state.pop('prompt_menu-default')

        if 'textbox-notebook' in state:
            state.pop('prompt_menu-notebook')

    elements = list_interface_input_elements()

    if len(state) == 0:
        return [gr.update() for k in elements]  # Dummy, do nothing
    else:
        return [state[k] if k in state else gr.update() for k in elements]


def save_settings(state, preset, extensions_list, show_controls, theme_state):
    output = copy.deepcopy(shared.settings)
    exclude = ['name2', 'greeting', 'context', 'truncation_length', 'instruction_template_str']
    for k in state:
        if k in shared.settings and k not in exclude:
            output[k] = state[k]

    output['preset'] = preset
    output['prompt-default'] = state['prompt_menu-default']
    output['prompt-notebook'] = state['prompt_menu-notebook']
    output['character'] = state['character_menu']
    output['default_extensions'] = extensions_list
    output['seed'] = int(output['seed'])
    output['show_controls'] = show_controls
    output['dark_theme'] = True if theme_state == 'dark' else False

    # Save extension values in the UI
    for extension_name in extensions_list:
        extension = getattr(extensions, extension_name, None)
        if extension:
            extension = extension.script
            if hasattr(extension, 'params'):
                params = getattr(extension, 'params')
                for param in params:
                    _id = f"{extension_name}-{param}"
                    # Only save if different from default value
                    if param not in shared.default_settings or params[param] != shared.default_settings[param]:
                        output[_id] = params[param]

    # Do not save unchanged settings
    for key in list(output.keys()):
        if key in shared.default_settings and output[key] == shared.default_settings[key]:
            output.pop(key)

    return yaml.dump(output, sort_keys=False, width=float("inf"), allow_unicode=True)


def create_refresh_button(refresh_component, refresh_method, refreshed_args, elem_class, interactive=True):
    """
    Copied from https://github.com/AUTOMATIC1111/stable-diffusion-webui
    """
    def refresh():
        refresh_method()
        args = refreshed_args() if callable(refreshed_args) else refreshed_args

        return gr.update(**(args or {}))

    refresh_button = gr.Button(refresh_symbol, elem_classes=elem_class, interactive=interactive)
    refresh_button.click(
        fn=lambda: {k: tuple(v) if type(k) is list else v for k, v in refresh().items()},
        inputs=[],
        outputs=[refresh_component]
    )

    return refresh_button
