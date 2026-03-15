import os
import signal
import sys
import time
import warnings
from functools import partial
from pathlib import Path
from threading import Lock, Thread

import yaml

from modules import shared, utils
from modules.image_models import load_image_model
from modules.logging_colors import logger
from modules.prompts import load_prompt

import modules.extensions as extensions_module
from modules.LoRA import add_lora_to_model
from modules.models import load_model, unload_model_if_idle
from modules.models_settings import (
    get_fallback_settings,
    get_model_metadata,
    update_model_parameters
)
from modules.shared import do_cmd_flags_warnings

os.environ['BITSANDBYTES_NOWELCOME'] = '1'

warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')
warnings.filterwarnings('ignore', category=UserWarning, message='Using the update method is deprecated')
warnings.filterwarnings('ignore', category=UserWarning, message='Field "model_name" has conflict')
warnings.filterwarnings('ignore', category=UserWarning, message='Field "model_names" has conflict')


def signal_handler(sig, frame):
    # On second Ctrl+C, force an immediate exit
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    signal.signal(signal.SIGTERM, signal.SIG_DFL)

    logger.info("Received Ctrl+C. Shutting down Text Generation Web UI gracefully.")

    # Explicitly stop LlamaServer to avoid __del__ cleanup issues during shutdown
    if shared.model and shared.model.__class__.__name__ == 'LlamaServer':
        try:
            shared.model.stop()
        except Exception:
            pass

    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


def create_interface():

    import shutil

    import gradio as gr

    from modules import (
        training,
        ui,
        ui_chat,
        ui_default,
        ui_file_saving,
        ui_image_generation,
        ui_model_menu,
        ui_notebook,
        ui_parameters,
        ui_session,
    )
    from modules.chat import generate_pfp_cache
    from modules.extensions import apply_extensions
    from modules.utils import gradio

    warnings.filterwarnings('ignore', category=UserWarning, message='The value passed into gr.Dropdown()')

    # Set up Gradio temp directory path
    gradio_temp_path = shared.user_data_dir / 'cache' / 'gradio'
    shutil.rmtree(gradio_temp_path, ignore_errors=True)
    gradio_temp_path.mkdir(parents=True, exist_ok=True)
    os.environ.update({
        'GRADIO_ANALYTICS_ENABLED': 'False',
        'GRADIO_TEMP_DIR': str(gradio_temp_path)
    })

    title = 'Text Generation Web UI'

    # Password authentication
    auth = []
    if shared.args.gradio_auth:
        auth.extend(x.strip() for x in shared.args.gradio_auth.strip('"').replace('\n', '').split(',') if x.strip())
    if shared.args.gradio_auth_path:
        with open(shared.args.gradio_auth_path, 'r', encoding="utf8") as file:
            auth.extend(x.strip() for line in file for x in line.split(',') if x.strip())
    auth = [tuple(cred.split(':')) for cred in auth]

    # Allowed paths
    allowed_paths = ["css", "js", "extensions", str(shared.user_data_dir / "cache")]
    if not shared.args.multi_user:
        allowed_paths.append(str(shared.user_data_dir / "image_outputs"))

    # Import the extensions and execute their setup() functions
    if shared.args.extensions is not None and len(shared.args.extensions) > 0:
        extensions_module.load_extensions()

    # Force some events to be triggered on page load
    shared.persistent_interface_state.update({
        'mode': shared.settings['mode'],
        'loader': shared.args.loader or 'llama.cpp',
        'filter_by_loader': (shared.args.loader or 'All') if not shared.args.portable else 'llama.cpp'
    })

    if not shared.settings['prompt-notebook']:
        shared.settings['prompt-notebook'] = utils.get_available_prompts()[0]

    prompt = load_prompt(shared.settings['prompt-notebook'])
    shared.persistent_interface_state.update({
        'textbox-default': prompt,
        'textbox-notebook': prompt
    })

    # Clear existing cache files
    for cache_file in ['pfp_character.png', 'pfp_character_thumb.png']:
        cache_path = shared.user_data_dir / "cache" / cache_file
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

    # Head HTML for font preloads, KaTeX, highlight.js, morphdom, and global JS
    head_html = '\n'.join([
        '<link rel="preload" href="file/css/Inter/Inter-VariableFont_opsz,wght.ttf" as="font" type="font/ttf" crossorigin>',
        '<link rel="preload" href="file/css/Inter/Inter-Italic-VariableFont_opsz,wght.ttf" as="font" type="font/ttf" crossorigin>',
        '<link rel="preload" href="file/css/NotoSans/NotoSans-Medium.woff2" as="font" type="font/woff2" crossorigin>',
        '<link rel="preload" href="file/css/NotoSans/NotoSans-MediumItalic.woff2" as="font" type="font/woff2" crossorigin>',
        '<link rel="preload" href="file/css/NotoSans/NotoSans-Bold.woff2" as="font" type="font/woff2" crossorigin>',
        '<script src="file/js/katex/katex.min.js"></script>',
        '<script src="file/js/katex/auto-render.js"></script>',
        '<script src="file/js/highlightjs/highlight.min.js"></script>',
        '<script src="file/js/highlightjs/highlightjs-copy.min.js"></script>',
        '<script src="file/js/morphdom/morphdom-umd.min.js"></script>',
        f'<link id="highlight-css" rel="stylesheet" href="file/css/highlightjs/{"github-dark" if shared.settings["dark_theme"] else "github"}.min.css">',
        '<script>hljs.addPlugin(new CopyButtonPlugin());</script>',
        f'<script>{ui.global_scope_js}</script>',
    ])

    with gr.Blocks(css=css, analytics_enabled=False, title=title, theme=ui.theme, head=head_html, dark_theme=shared.settings['dark_theme']) as shared.gradio['interface']:

        # Interface state
        shared.gradio['interface_state'] = gr.State({k: None for k in shared.input_elements})

        # Audio notification
        if (shared.user_data_dir / "notification.mp3").exists():
            shared.gradio['audio_notification'] = gr.Audio(interactive=False, value=str(shared.user_data_dir / "notification.mp3"), elem_id="audio_notification", visible=False)

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
            ui_image_generation.create_ui()  # Image generation tab
            training.create_ui()  # Training tab
        ui_session.create_ui()  # Session tab

        # Generation events
        ui_chat.create_event_handlers()
        ui_default.create_event_handlers()
        ui_notebook.create_event_handlers()
        if not shared.args.portable:
            ui_image_generation.create_event_handlers()

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
                {js}
                {ui.show_controls_js}
                toggle_controls(x);
            }}"""
        )

        shared.gradio['interface'].load(partial(ui.apply_interface_values, {}, use_persistent=True), None, gradio(ui.list_interface_input_elements()), show_progress=False)

        # Sync theme_state with the actual client-side theme so that
        # autosave always writes the correct dark_theme value.
        shared.gradio['interface'].load(None, None, gradio('theme_state'), js='() => document.body.classList.contains("dark") ? "dark" : "light"')

        extensions_module.create_extensions_tabs()  # Extensions tabs
        extensions_module.create_extensions_block()  # Extensions block

    # Launch the interface
    shared.gradio['interface'].queue()
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
        allowed_paths=allowed_paths,
    )


if __name__ == "__main__":

    logger.info("Starting Text Generation Web UI")
    do_cmd_flags_warnings()

    # Load custom settings
    settings_file = None
    if shared.args.settings is not None and Path(shared.args.settings).exists():
        settings_file = Path(shared.args.settings)
    elif (shared.user_data_dir / 'settings.yaml').exists():
        settings_file = shared.user_data_dir / 'settings.yaml'

    if settings_file is not None:
        logger.info(f"Loading settings from \"{settings_file}\"")
        with open(settings_file, 'r', encoding='utf-8') as f:
            new_settings = yaml.safe_load(f.read())

        if new_settings:
            shared.settings.update(new_settings)

    # Apply CLI overrides for image model settings (CLI flags take precedence over saved settings)
    shared.apply_image_model_cli_overrides()

    # Fallback settings for models
    shared.model_config['.*'] = get_fallback_settings()
    shared.model_config.move_to_end('.*', last=False)  # Move to the beginning

    # Activate the extensions listed on settings.yaml
    extensions_module.available_extensions = utils.get_available_extensions()
    for extension in shared.settings['default_extensions']:
        shared.args.extensions = shared.args.extensions or []
        if extension not in shared.args.extensions:
            shared.args.extensions.append(extension)

    # Load image model if specified via CLI
    if shared.args.image_model:
        logger.info(f"Loading image model: {shared.args.image_model}")
        result = load_image_model(
            shared.args.image_model,
            dtype=shared.settings.get('image_dtype', 'bfloat16'),
            attn_backend=shared.settings.get('image_attn_backend', 'sdpa'),
            cpu_offload=shared.settings.get('image_cpu_offload', False),
            compile_model=shared.settings.get('image_compile', False),
            quant_method=shared.settings.get('image_quant', 'none')
        )
        if result is not None:
            shared.image_model_name = shared.args.image_model
        else:
            logger.error(f"Failed to load image model: {shared.args.image_model}")

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
        model_settings = get_model_metadata(shared.model_name)
        update_model_parameters(model_settings, initial=True)  # hijack the command-line arguments

        # Load the model
        shared.model, shared.tokenizer = load_model(shared.model_name)
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
