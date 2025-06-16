import os
import shutil
import warnings
from pathlib import Path

from modules import shared
from modules.block_requests import OpenMonkeyPatch, RequestBlocker
from modules.logging_colors import logger

# Set up Gradio temp directory path
gradio_temp_path = Path('user_data') / 'cache' / 'gradio'
shutil.rmtree(gradio_temp_path, ignore_errors=True)
gradio_temp_path.mkdir(parents=True, exist_ok=True)

# Set environment variables
os.environ.update({
    'GRADIO_ANALYTICS_ENABLED': 'False',
    'BITSANDBYTES_NOWELCOME': '1',
    'GRADIO_TEMP_DIR': str(gradio_temp_path)
})

warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')
warnings.filterwarnings('ignore', category=UserWarning, message='Using the update method is deprecated')
warnings.filterwarnings('ignore', category=UserWarning, message='Field "model_name" has conflict')
warnings.filterwarnings('ignore', category=UserWarning, message='The value passed into gr.Dropdown()')
warnings.filterwarnings('ignore', category=UserWarning, message='Field "model_names" has conflict')

with RequestBlocker():
    from modules import gradio_hijack
    import gradio as gr

import matplotlib

matplotlib.use('Agg')  # This fixes LaTeX rendering on some systems

import os
import signal
import sys
import time
from functools import partial
from threading import Lock, Thread

import yaml

import modules.extensions as extensions_module
from modules import (
    training,
    ui,
    ui_chat,
    ui_default,
    ui_file_saving,
    ui_model_menu,
    ui_notebook,
    ui_parameters,
    ui_session,
    utils
)
from modules.chat import generate_pfp_cache
from modules.extensions import apply_extensions
from modules.LoRA import add_lora_to_model
from modules.models import load_model, unload_model_if_idle
from modules.models_settings import (
    get_fallback_settings,
    get_model_metadata,
    update_gpu_layers_and_vram,
    update_model_parameters
)
from modules.shared import do_cmd_flags_warnings
from modules.utils import gradio


def signal_handler(sig, frame):
    logger.info("Received Ctrl+C. Shutting down Text generation web UI gracefully.")

    # Explicitly stop LlamaServer to avoid __del__ cleanup issues during shutdown
    if shared.model and shared.model.__class__.__name__ == 'LlamaServer':
        try:
            shared.model.stop()
        except:
            pass

    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)


def create_interface():

    title = 'Text generation web UI'

    # Password authentication
    auth = []
    if shared.args.gradio_auth:
        auth.extend(x.strip() for x in shared.args.gradio_auth.strip('"').replace('\n', '').split(',') if x.strip())
    if shared.args.gradio_auth_path:
        with open(shared.args.gradio_auth_path, 'r', encoding="utf8") as file:
            auth.extend(x.strip() for line in file for x in line.split(',') if x.strip())
    auth = [tuple(cred.split(':')) for cred in auth]

    # Import the extensions and execute their setup() functions
    if shared.args.extensions is not None and len(shared.args.extensions) > 0:
        extensions_module.load_extensions()

    # Force some events to be triggered on page load
    shared.persistent_interface_state.update({
        'mode': shared.settings['mode'],
        'loader': shared.args.loader or 'llama.cpp',
        'filter_by_loader': (shared.args.loader or 'All') if not shared.args.portable else 'llama.cpp'
    })

    # Clear existing cache files
    for cache_file in ['pfp_character.png', 'pfp_character_thumb.png']:
        cache_path = Path(f"user_data/cache/{cache_file}")
        if cache_path.exists():
            cache_path.unlink()

    # Regenerate for default character
    if shared.settings['mode'] != 'instruct':
        generate_pfp_cache(shared.settings['character'])

    # css/js strings
    css = ui.css
    js = ui.js
    css += apply_extensions('css')
    js += apply_extensions('js')

    # Interface state elements
    shared.input_elements = ui.list_interface_input_elements()

    with gr.Blocks(css=css, analytics_enabled=False, title=title, theme=ui.theme) as shared.gradio['interface']:

        # Interface state
        shared.gradio['interface_state'] = gr.State({k: None for k in shared.input_elements})

        # Audio notification
        if Path("user_data/notification.mp3").exists():
            shared.gradio['audio_notification'] = gr.Audio(interactive=False, value="user_data/notification.mp3", elem_id="audio_notification", visible=False)

        # Floating menus for saving/deleting files
        ui_file_saving.create_ui()

        # Temporary clipboard for saving files
        shared.gradio['temporary_text'] = gr.Textbox(visible=False)

        # Chat tab
        ui_chat.create_ui()

        # Notebook tab
        with gr.Tab("Notebook", elem_id='notebook-parent-tab'):
            ui_default.create_ui()
            ui_notebook.create_ui()

        ui_parameters.create_ui()  # Parameters tab
        ui_chat.create_character_settings_ui()  # Character tab
        ui_model_menu.create_ui()  # Model tab
        if not shared.args.portable:
            training.create_ui()  # Training tab
        ui_session.create_ui()  # Session tab

        # Generation events
        ui_chat.create_event_handlers()
        ui_default.create_event_handlers()
        ui_notebook.create_event_handlers()

        # Other events
        ui_file_saving.create_event_handlers()
        ui_parameters.create_event_handlers()
        ui_model_menu.create_event_handlers()

        # UI persistence events
        ui.setup_auto_save()

        # Interface launch events
        shared.gradio['interface'].load(
            None,
            gradio('show_controls'),
            None,
            js=f"""(x) => {{
                // Check if this is first visit or if localStorage is out of sync
                const savedTheme = localStorage.getItem('theme');
                const serverTheme = {str(shared.settings['dark_theme']).lower()} ? 'dark' : 'light';

                // If no saved theme or mismatch with server on first load, use server setting
                if (!savedTheme || !sessionStorage.getItem('theme_synced')) {{
                    localStorage.setItem('theme', serverTheme);
                    sessionStorage.setItem('theme_synced', 'true');
                    if (serverTheme === 'dark') {{
                        document.getElementsByTagName('body')[0].classList.add('dark');
                    }} else {{
                        document.getElementsByTagName('body')[0].classList.remove('dark');
                    }}
                }} else {{
                    // Use localStorage for subsequent reloads
                    if (savedTheme === 'dark') {{
                        document.getElementsByTagName('body')[0].classList.add('dark');
                    }} else {{
                        document.getElementsByTagName('body')[0].classList.remove('dark');
                    }}
                }}
                {js}
                {ui.show_controls_js}
                toggle_controls(x);
            }}"""
        )

        shared.gradio['interface'].load(partial(ui.apply_interface_values, {}, use_persistent=True), None, gradio(ui.list_interface_input_elements()), show_progress=False)

        extensions_module.create_extensions_tabs()  # Extensions tabs
        extensions_module.create_extensions_block()  # Extensions block

    # Launch the interface
    shared.gradio['interface'].queue()
    with OpenMonkeyPatch():
        shared.gradio['interface'].launch(
            max_threads=64,
            prevent_thread_lock=True,
            share=shared.args.share,
            server_name=None if not shared.args.listen else (shared.args.listen_host or '0.0.0.0'),
            server_port=shared.args.listen_port,
            inbrowser=shared.args.auto_launch,
            auth=auth or None,
            ssl_verify=False if (shared.args.ssl_keyfile or shared.args.ssl_certfile) else True,
            ssl_keyfile=shared.args.ssl_keyfile,
            ssl_certfile=shared.args.ssl_certfile,
            root_path=shared.args.subpath,
            allowed_paths=["css", "js", "extensions", "user_data/cache"]
        )


if __name__ == "__main__":

    logger.info("Starting Text generation web UI")
    do_cmd_flags_warnings()

    # Load custom settings
    settings_file = None
    if shared.args.settings is not None and Path(shared.args.settings).exists():
        settings_file = Path(shared.args.settings)
    elif Path('user_data/settings.yaml').exists():
        settings_file = Path('user_data/settings.yaml')

    if settings_file is not None:
        logger.info(f"Loading settings from \"{settings_file}\"")
        with open(settings_file, 'r', encoding='utf-8') as f:
            new_settings = yaml.safe_load(f.read())

        if new_settings:
            shared.settings.update(new_settings)

    # Fallback settings for models
    shared.model_config['.*'] = get_fallback_settings()
    shared.model_config.move_to_end('.*', last=False)  # Move to the beginning

    # Activate the extensions listed on settings.yaml
    extensions_module.available_extensions = utils.get_available_extensions()
    for extension in shared.settings['default_extensions']:
        shared.args.extensions = shared.args.extensions or []
        if extension not in shared.args.extensions:
            shared.args.extensions.append(extension)

    available_models = utils.get_available_models()

    # Model defined through --model
    if shared.args.model is not None:
        shared.model_name = shared.args.model

    # Select the model from a command-line menu
    elif shared.args.model_menu:
        if len(available_models) == 0:
            logger.error('No models are available! Please download at least one.')
            sys.exit(0)
        else:
            print('The following models are available:\n')
            for i, model in enumerate(available_models):
                print(f'{i+1}. {model}')

            print(f'\nWhich one do you want to load? 1-{len(available_models)}\n')
            i = int(input()) - 1
            print()

        shared.model_name = available_models[i]

    # If any model has been selected, load it
    if shared.model_name != 'None':
        p = Path(shared.model_name)
        if p.exists():
            model_name = p.parts[-1]
            shared.model_name = model_name
        else:
            model_name = shared.model_name

        model_settings = get_model_metadata(model_name)
        update_model_parameters(model_settings, initial=True)  # hijack the command-line arguments

        # Auto-adjust GPU layers if not provided by user and it's a llama.cpp model
        if 'gpu_layers' not in shared.provided_arguments and shared.args.loader == 'llama.cpp' and 'gpu_layers' in model_settings:
            vram_usage, adjusted_layers = update_gpu_layers_and_vram(
                shared.args.loader,
                model_name,
                model_settings['gpu_layers'],
                shared.args.ctx_size,
                shared.args.cache_type,
                auto_adjust=True,
                for_ui=False
            )

            shared.args.gpu_layers = adjusted_layers

        # Load the model
        shared.model, shared.tokenizer = load_model(model_name)
        if shared.args.lora:
            add_lora_to_model(shared.args.lora)

    shared.generation_lock = Lock()

    if shared.args.idle_timeout > 0:
        timer_thread = Thread(target=unload_model_if_idle)
        timer_thread.daemon = True
        timer_thread.start()

    if shared.args.nowebui:
        # Start the API in standalone mode
        shared.args.extensions = [x for x in (shared.args.extensions or []) if x != 'gallery']
        if shared.args.extensions:
            extensions_module.load_extensions()
    else:
        # Launch the web UI
        create_interface()
        while True:
            time.sleep(0.5)
            if shared.need_restart:
                shared.need_restart = False
                time.sleep(0.5)
                shared.gradio['interface'].close()
                time.sleep(0.5)
                create_interface()
