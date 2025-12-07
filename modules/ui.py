import copy
import threading
from pathlib import Path

import gradio as gr
import yaml

import extensions
import modules.extensions as extensions_module
from modules import shared
from modules.chat import load_history
from modules.utils import gradio

# Global state for auto-saving UI settings with debouncing
_auto_save_timer = None
_auto_save_lock = threading.Lock()
_last_interface_state = None
_last_preset = None
_last_extensions = None
_last_show_controls = None
_last_theme_state = None

with open(Path(__file__).resolve().parent / '../css/NotoSans/stylesheet.css', 'r', encoding='utf-8') as f:
    css = f.read()
with open(Path(__file__).resolve().parent / '../css/main.css', 'r', encoding='utf-8') as f:
    css += f.read()
with open(Path(__file__).resolve().parent / '../css/katex/katex.min.css', 'r', encoding='utf-8') as f:
    css += f.read()
with open(Path(__file__).resolve().parent / '../css/highlightjs/highlightjs-copy.min.css', 'r', encoding='utf-8') as f:
    css += f.read()
with open(Path(__file__).resolve().parent / '../js/main.js', 'r', encoding='utf-8') as f:
    js = f.read()
with open(Path(__file__).resolve().parent / '../js/global_scope_js.js', 'r', encoding='utf-8') as f:
    global_scope_js = f.read()
with open(Path(__file__).resolve().parent / '../js/save_files.js', 'r', encoding='utf-8') as f:
    save_files_js = f.read()
with open(Path(__file__).resolve().parent / '../js/switch_tabs.js', 'r', encoding='utf-8') as f:
    switch_tabs_js = f.read()
with open(Path(__file__).resolve().parent / '../js/show_controls.js', 'r', encoding='utf-8') as f:
    show_controls_js = f.read()
with open(Path(__file__).resolve().parent / '../js/update_big_picture.js', 'r', encoding='utf-8') as f:
    update_big_picture_js = f.read()
with open(Path(__file__).resolve().parent / '../js/dark_theme.js', 'r', encoding='utf-8') as f:
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
        background_fill_secondary_dark='var(--selected-item-color-dark, #282930)',
        background_fill_primary='var(--neutral-50)',
        background_fill_primary_dark='var(--darker-gray, #1C1C1D)',
        body_background_fill="white",
        block_background_fill="transparent",
        body_text_color='rgb(64, 64, 64)',
        button_secondary_background_fill="white",
        button_secondary_border_color="var(--border-color-primary)",
        input_shadow="none",
        button_shadow_hover="none",

        # Dark Mode Colors
        input_background_fill_dark='var(--darker-gray, #1C1C1D)',
        checkbox_background_color_dark='var(--darker-gray, #1C1C1D)',
        block_background_fill_dark='transparent',
        block_border_color_dark='transparent',
        input_border_color_dark='var(--border-color-dark, #525252)',
        input_border_color_focus_dark='var(--border-color-dark, #525252)',
        checkbox_border_color_dark='var(--border-color-dark, #525252)',
        border_color_primary_dark='var(--border-color-dark, #525252)',
        button_secondary_border_color_dark='var(--border-color-dark, #525252)',
        body_background_fill_dark='var(--dark-gray, #212125)',
        button_primary_background_fill_dark='transparent',
        button_secondary_background_fill_dark='transparent',
        checkbox_label_background_fill_dark='transparent',
        button_cancel_background_fill_dark='transparent',
        button_secondary_background_fill_hover_dark='var(--selected-item-color-dark, #282930)',
        checkbox_label_background_fill_hover_dark='var(--selected-item-color-dark, #282930)',
        table_even_background_fill_dark='var(--darker-gray, #1C1C1D)',
        table_odd_background_fill_dark='var(--selected-item-color-dark, #282930)',
        code_background_fill_dark='var(--darker-gray, #1C1C1D)',

        # Shadows and Radius
        checkbox_label_shadow='none',
        block_shadow='none',
        block_shadow_dark='none',
        input_shadow_focus='none',
        input_shadow_focus_dark='none',
        button_large_radius='0.375rem',
        button_large_padding='6px 12px',
        input_radius='0.375rem',
        block_radius='0',
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
        'gpu_layers',
        'cpu_moe',
        'threads',
        'threads_batch',
        'batch_size',
        'ubatch_size',
        'ctx_size',
        'cache_type',
        'tensor_split',
        'extra_flags',
        'streaming_llm',
        'gpu_split',
        'alpha_value',
        'rope_freq_base',
        'compress_pos_emb',
        'compute_dtype',
        'quant_type',
        'num_experts_per_token',
        'load_in_8bit',
        'load_in_4bit',
        'attn_implementation',
        'cpu',
        'disk',
        'row_split',
        'no_kv_offload',
        'no_mmap',
        'mlock',
        'numa',
        'use_double_quant',
        'bf16',
        'autosplit',
        'enable_tp',
        'tp_backend',
        'no_flash_attn',
        'no_xformers',
        'no_sdpa',
        'cfg_cache',
        'cpp_runner',
        'no_use_fast',
        'model_draft',
        'draft_max',
        'gpu_layers_draft',
        'device_draft',
        'ctx_size_draft',
        'mmproj',
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
        'do_sample',
        'dynamic_temperature',
        'temperature_last',
        'auto_max_new_tokens',
        'ban_eos_token',
        'add_bos_token',
        'enable_thinking',
        'reasoning_effort',
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
        'navigate_message_index',
        'navigate_direction',
        'navigate_message_role',
        'edit_message_index',
        'edit_message_text',
        'edit_message_role',
        'branch_index',
        'enable_web_search',
        'web_search_pages',
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

    # Other elements
    elements += [
        'show_two_notebook_columns',
        'paste_to_attachment',
        'include_past_attachments',
    ]

    if not shared.args.portable:
        # Image generation elements
        elements += [
            'image_prompt',
            'image_neg_prompt',
            'image_width',
            'image_height',
            'image_aspect_ratio',
            'image_steps',
            'image_cfg_scale',
            'image_seed',
            'image_batch_size',
            'image_batch_count',
            'image_llm_variations',
            'image_llm_variations_prompt',
            'image_model_menu',
            'image_dtype',
            'image_attn_backend',
            'image_compile',
            'image_cpu_offload',
            'image_quant',
        ]

    return elements


def gather_interface_values(*args):
    interface_elements = list_interface_input_elements()

    output = {}
    for element, value in zip(interface_elements, args):
        output[element] = value

    if not shared.args.multi_user:
        shared.persistent_interface_state = output

        # Remove the chat input, as it gets cleared after this function call
        shared.persistent_interface_state.pop('textbox')

    # Prevent history loss if backend is restarted but UI is not refreshed
    if (output['history'] is None or (len(output['history'].get('visible', [])) == 0 and len(output['history'].get('internal', [])) == 0)) and output['unique_id'] is not None:
        output['history'] = load_history(output['unique_id'], output['character_menu'], output['mode'])

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


def save_settings(state, preset, extensions_list, show_controls, theme_state, manual_save=False):
    output = copy.deepcopy(shared.settings)
    exclude = []
    for k in state:
        if k in shared.settings and k not in exclude:
            output[k] = state[k]

    output['preset'] = preset
    output['prompt-notebook'] = state['prompt_menu-default'] if state['show_two_notebook_columns'] else state['prompt_menu-notebook']
    output['character'] = state['character_menu']
    output['seed'] = int(output['seed'])
    output['show_controls'] = show_controls
    output['dark_theme'] = True if theme_state == 'dark' else False
    output.pop('instruction_template_str')
    output.pop('truncation_length')

    # Handle extensions and extension parameters
    if manual_save:
        # Save current extensions and their parameter values
        output['default_extensions'] = extensions_list

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
    else:
        # Preserve existing extensions and extension parameters during autosave
        settings_path = Path('user_data') / 'settings.yaml'
        if settings_path.exists():
            try:
                with open(settings_path, 'r', encoding='utf-8') as f:
                    existing_settings = yaml.safe_load(f.read()) or {}

                # Preserve default_extensions
                if 'default_extensions' in existing_settings:
                    output['default_extensions'] = existing_settings['default_extensions']

                # Preserve extension parameter values
                for key, value in existing_settings.items():
                    if any(key.startswith(f"{ext_name}-") for ext_name in extensions_module.available_extensions):
                        output[key] = value
            except Exception:
                pass  # If we can't read the file, just don't modify extensions

    # Do not save unchanged settings
    for key in list(output.keys()):
        if key in shared.default_settings and output[key] == shared.default_settings[key]:
            output.pop(key)

    return yaml.dump(output, sort_keys=False, width=float("inf"), allow_unicode=True)


def store_current_state_and_debounce(interface_state, preset, extensions, show_controls, theme_state):
    """Store current state and trigger debounced save"""
    global _auto_save_timer, _last_interface_state, _last_preset, _last_extensions, _last_show_controls, _last_theme_state

    if shared.args.multi_user:
        return

    # Store the current state in global variables
    _last_interface_state = interface_state
    _last_preset = preset
    _last_extensions = extensions
    _last_show_controls = show_controls
    _last_theme_state = theme_state

    # Reset the debounce timer
    with _auto_save_lock:
        if _auto_save_timer is not None:
            _auto_save_timer.cancel()

        _auto_save_timer = threading.Timer(1.0, _perform_debounced_save)
        _auto_save_timer.start()


def _perform_debounced_save():
    """Actually perform the save using the stored state"""
    global _auto_save_timer

    try:
        if _last_interface_state is not None:
            contents = save_settings(_last_interface_state, _last_preset, _last_extensions, _last_show_controls, _last_theme_state, manual_save=False)
            settings_path = Path('user_data') / 'settings.yaml'
            settings_path.parent.mkdir(exist_ok=True)
            with open(settings_path, 'w', encoding='utf-8') as f:
                f.write(contents)
    except Exception as e:
        print(f"Auto-save failed: {e}")
    finally:
        with _auto_save_lock:
            _auto_save_timer = None


def setup_auto_save():
    """Attach auto-save to key UI elements"""
    if shared.args.multi_user:
        return

    change_elements = [
        # Chat tab (ui_chat.py)
        'start_with',
        'enable_web_search',
        'web_search_pages',
        'mode',
        'chat_style',
        'chat-instruct_command',
        'character_menu',
        'name1',
        'name2',
        'context',
        'greeting',
        'user_bio',
        'custom_system_message',
        'chat_template_str',

        # Parameters tab (ui_parameters.py) - Generation parameters
        'preset_menu',
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
        'do_sample',
        'dynamic_temperature',
        'temperature_last',
        'auto_max_new_tokens',
        'ban_eos_token',
        'add_bos_token',
        'enable_thinking',
        'reasoning_effort',
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

        # Default tab (ui_default.py)
        'prompt_menu-default',

        # Notebook tab (ui_notebook.py)
        'prompt_menu-notebook',

        # Session tab (ui_session.py)
        'show_controls',
        'theme_state',
        'show_two_notebook_columns',
        'paste_to_attachment',
        'include_past_attachments',

    ]

    if not shared.args.portable:
        # Image generation tab (ui_image_generation.py)
        change_elements += [
            'image_prompt',
            'image_neg_prompt',
            'image_width',
            'image_height',
            'image_aspect_ratio',
            'image_steps',
            'image_cfg_scale',
            'image_seed',
            'image_batch_size',
            'image_batch_count',
            'image_llm_variations',
            'image_llm_variations_prompt',
            'image_model_menu',
            'image_dtype',
            'image_attn_backend',
            'image_compile',
            'image_cpu_offload',
            'image_quant',
        ]

    for element_name in change_elements:
        if element_name in shared.gradio:
            shared.gradio[element_name].change(
                gather_interface_values, gradio(shared.input_elements), gradio('interface_state')).then(
                store_current_state_and_debounce, gradio('interface_state', 'preset_menu', 'extensions_menu', 'show_controls', 'theme_state'), None, show_progress=False)


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
