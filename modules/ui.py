import copy
from pathlib import Path

import gradio as gr
import yaml

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
with open(Path(__file__).resolve().parent / '../js/global_scope_js.js', 'r') as f:
    global_scope_js = f.read()
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

if not shared.args.old_colors:
    theme = theme.set(
        # General Colors
        border_color_primary='#c5c5d2',
        body_text_color_subdued='#484848',
        background_fill_secondary='#eaeaea',
        background_fill_secondary_dark='var(--selected-item-color-dark)',
        background_fill_primary='var(--neutral-50)',
        background_fill_primary_dark='var(--darker-gray)',
        body_background_fill="white",
        block_background_fill="transparent",
        body_text_color="#333",
        button_secondary_background_fill="#f4f4f4",
        button_secondary_border_color="var(--border-color-primary)",

        # Dark Mode Colors
        input_background_fill_dark='var(--darker-gray)',
        checkbox_background_color_dark='var(--darker-gray)',
        block_background_fill_dark='transparent',
        block_border_color_dark='transparent',
        input_border_color_dark='var(--border-color-dark)',
        checkbox_border_color_dark='var(--border-color-dark)',
        border_color_primary_dark='var(--border-color-dark)',
        button_secondary_border_color_dark='var(--border-color-dark)',
        body_background_fill_dark='var(--dark-gray)',
        button_primary_background_fill_dark='transparent',
        button_secondary_background_fill_dark='transparent',
        checkbox_label_background_fill_dark='transparent',
        button_cancel_background_fill_dark='transparent',
        button_secondary_background_fill_hover_dark='var(--selected-item-color-dark)',
        checkbox_label_background_fill_hover_dark='var(--selected-item-color-dark)',
        table_even_background_fill_dark='var(--darker-gray)',
        table_odd_background_fill_dark='var(--selected-item-color-dark)',
        code_background_fill_dark='var(--darker-gray)',

        # Shadows and Radius
        checkbox_label_shadow='none',
        block_shadow='none',
        block_shadow_dark='none',
        button_large_radius='0.375rem',
        button_large_padding='6px 12px',
        input_radius='0.375rem',
    )

if Path("user_data/notification.mp3").exists():
    audio_notification_js = "document.querySelector('#audio_notification audio')?.play();"
else:
    audio_notification_js = ""


def list_model_elements():
    elements = [
        'filter_by_loader',
        'loader',
        'cpu_memory',
        'n_gpu_layers',
        'threads',
        'threads_batch',
        'batch_size',
        'hqq_backend',
        'ctx_size',
        'cache_type',
        'tensor_split',
        'extra_flags',
        'gpu_split',
        'alpha_value',
        'rope_freq_base',
        'compress_pos_emb',
        'compute_dtype',
        'quant_type',
        'num_experts_per_token',
        'load_in_8bit',
        'load_in_4bit',
        'torch_compile',
        'flash_attn',
        'use_flash_attention_2',
        'cpu',
        'disk',
        'row_split',
        'no_kv_offload',
        'no_mmap',
        'mlock',
        'numa',
        'use_double_quant',
        'use_eager_attention',
        'bf16',
        'autosplit',
        'enable_tp',
        'no_flash_attn',
        'no_xformers',
        'no_sdpa',
        'cfg_cache',
        'cpp_runner',
        'trust_remote_code',
        'no_use_fast',
        'model_draft',
        'draft_max',
        'gpu_layers_draft',
        'device_draft',
        'ctx_size_draft',
    ]

    return elements


def list_interface_input_elements():
    elements = [
        'temperature',
        'dynatemp_low',
        'dynatemp_high',
        'dynatemp_exponent',
        'smoothing_factor',
        'smoothing_curve',
        'min_p',
        'top_p',
        'top_k',
        'typical_p',
        'xtc_threshold',
        'xtc_probability',
        'epsilon_cutoff',
        'eta_cutoff',
        'tfs',
        'top_a',
        'top_n_sigma',
        'dry_multiplier',
        'dry_allowed_length',
        'dry_base',
        'repetition_penalty',
        'frequency_penalty',
        'presence_penalty',
        'encoder_repetition_penalty',
        'no_repeat_ngram_size',
        'repetition_penalty_range',
        'penalty_alpha',
        'guidance_scale',
        'mirostat_mode',
        'mirostat_tau',
        'mirostat_eta',
        'max_new_tokens',
        'prompt_lookup_num_tokens',
        'max_tokens_second',
        'max_updates_second',
        'do_sample',
        'dynamic_temperature',
        'temperature_last',
        'auto_max_new_tokens',
        'ban_eos_token',
        'add_bos_token',
        'skip_special_tokens',
        'stream',
        'static_cache',
        'truncation_length',
        'seed',
        'sampler_priority',
        'custom_stopping_strings',
        'custom_token_bans',
        'negative_prompt',
        'dry_sequence_breakers',
        'grammar_string',
    ]

    # Chat elements
    elements += [
        'history',
        'search_chat',
        'unique_id',
        'textbox',
        'start_with',
        'mode',
        'chat_style',
        'chat-instruct_command',
        'character_menu',
        'name2',
        'context',
        'greeting',
        'name1',
        'user_bio',
        'custom_system_message',
        'instruction_template_str',
        'chat_template_str',
    ]

    # Notebook/default elements
    elements += [
        'textbox-default',
        'textbox-notebook',
        'prompt_menu-default',
        'prompt_menu-notebook',
        'output_textbox',
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
        if 'textbox-default' in state and 'prompt_menu-default' in state:
            state.pop('prompt_menu-default')

        if 'textbox-notebook' in state and 'prompt_menu-notebook' in state:
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
