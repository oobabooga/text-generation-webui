import json
import re
import urllib.request
from pathlib import Path

import gradio as gr

from modules import shared, ui, utils
from modules.utils import gradio

PORTABLE_FOLDER_RE = re.compile(r'^textgen(?:-ik)?-(\d+\.\d+(?:\.\d+)?)$')


def detect_portable_install():
    """Return the local version string if running from a portable build, else None."""
    grandparent = Path(__file__).resolve().parent.parent
    if grandparent.name != 'app':
        return None

    match = PORTABLE_FOLDER_RE.match(grandparent.parent.name)
    if not match:
        return None

    return match.group(1)


def check_for_updates(local_version):
    try:
        req = urllib.request.Request(
            'https://api.github.com/repos/oobabooga/textgen/releases/latest',
            headers={'Accept': 'application/vnd.github+json'},
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode('utf-8'))
    except Exception as e:
        return f'<div class="update-status update-error">Failed to check for updates: {e}</div>'

    latest = (data.get('tag_name') or '').lstrip('v')
    published = (data.get('published_at') or '')[:10]
    url = data.get('html_url') or 'https://github.com/oobabooga/textgen/releases/latest'

    if latest and latest != local_version:
        return (
            f'<div class="update-status update-available">'
            f'<h3>Update available</h3>'
            f'<ul>'
            f'<li>Current version: {local_version}</li>'
            f'<li>Latest version: {latest} (released {published})</li>'
            f'</ul>'
            f'<p><a href="{url}" target="_blank" rel="noopener">Download here</a></p>'
            f'</div>'
        )

    return f'<div class="update-status update-current">Already up to date (version {local_version}).</div>'


def create_ui():
    mu = shared.args.multi_user
    portable_version = detect_portable_install() if shared.args.portable else None
    with gr.Tab("Session", elem_id="session-tab"):
        with gr.Row():
            with gr.Column():
                gr.Markdown("## Settings")
                if shared.is_electron:
                    with gr.Row():
                        shared.gradio['model_dir'] = gr.Textbox(label='Models directory', value=shared.settings['model_dir'], scale=4, elem_classes='slim-textbox')
                        shared.gradio['model_dir_browse'] = gr.Button('Browse', elem_classes=['refresh-button', 'refresh-button-medium'])

                shared.gradio['toggle_dark_mode'] = gr.Button('Toggle light/dark theme 💡', elem_classes=['refresh-button', 'settings-button'])
                shared.gradio['show_two_notebook_columns'] = gr.Checkbox(label='Show two columns in the Notebook tab', value=shared.settings['show_two_notebook_columns'])
                shared.gradio['paste_to_attachment'] = gr.Checkbox(label='Turn long pasted text into attachments in the Chat tab', value=shared.settings['paste_to_attachment'], elem_id='paste_to_attachment')
                shared.gradio['include_past_attachments'] = gr.Checkbox(label='Include attachments/search results from previous messages in the chat prompt', value=shared.settings['include_past_attachments'])
                if shared.is_electron:
                    shared.gradio['spellcheck'] = gr.Checkbox(label='Enable spellcheck in text inputs', value=shared.settings['spellcheck'], elem_id='spellcheck')

                if portable_version:
                    gr.Markdown("## Updates")
                    shared.gradio['check_updates'] = gr.Button('Check for updates 🔄', elem_classes=['refresh-button', 'settings-button'])
                    shared.gradio['update_status'] = gr.HTML(value='', elem_id='update-status')

            with gr.Column():
                gr.Markdown("## Extensions & flags")
                with gr.Row():
                    shared.gradio['save_settings'] = gr.Button(f'Save extensions settings to {shared.user_data_dir}/settings.yaml', elem_classes=['refresh-button', 'settings-button'], interactive=not mu)
                    shared.gradio['reset_interface'] = gr.Button("Apply flags/extensions and restart", elem_classes=['refresh-button', 'settings-button'], interactive=not mu)
                with gr.Row():
                    with gr.Column():
                        shared.gradio['extensions_menu'] = gr.CheckboxGroup(choices=utils.get_available_extensions(), value=shared.args.extensions, label="Available extensions", info='Note that some of these extensions may require manually installing Python requirements through the command: pip install -r extensions/extension_name/requirements.txt', elem_classes='checkboxgroup-table')

                    with gr.Column():
                        shared.gradio['bool_menu'] = gr.CheckboxGroup(choices=get_boolean_arguments(), value=get_boolean_arguments(active=True), label="Boolean command-line flags", elem_classes='checkboxgroup-table')

        shared.gradio['theme_state'] = gr.Textbox(visible=False, value='dark' if shared.settings['dark_theme'] else 'light')
        if not mu:
            shared.gradio['save_settings'].click(
                ui.gather_interface_values, gradio(shared.input_elements), gradio('interface_state')).then(
                handle_save_settings, gradio('interface_state', 'preset_menu', 'extensions_menu', 'show_controls', 'theme_state'), gradio('save_contents', 'save_filename', 'save_root', 'save_root_state', 'file_saver'), show_progress=False)

        shared.gradio['toggle_dark_mode'].click(
            lambda x: 'dark' if x == 'light' else 'light', gradio('theme_state'), gradio('theme_state')).then(
            None, None, None, js=f'() => {{{ui.dark_theme_js}; toggleDarkMode(); localStorage.setItem("theme", document.body.classList.contains("dark") ? "dark" : "light")}}')

        if portable_version:
            shared.gradio['check_updates'].click(
                lambda: check_for_updates(portable_version), None, gradio('update_status'), show_progress=False)

        if shared.is_electron:
            shared.gradio['model_dir_browse'].click(
                None, gradio('model_dir'), gradio('model_dir'),
                js='async (current) => { const p = await window.electronAPI.pickDirectory(); return p === null ? current : p; }')

            shared.gradio['model_dir'].change(apply_model_dir, gradio('model_dir'), gradio('model_menu', 'model_draft'), show_progress=False)

        shared.gradio['show_two_notebook_columns'].change(
            handle_default_to_notebook_change,
            gradio('show_two_notebook_columns', 'textbox-default', 'output_textbox', 'prompt_menu-default', 'textbox-notebook', 'prompt_menu-notebook'),
            gradio('default-tab', 'notebook-tab', 'textbox-default', 'output_textbox', 'prompt_menu-default', 'textbox-notebook', 'prompt_menu-notebook')
        )

        # Reset interface event
        if not mu:
            shared.gradio['reset_interface'].click(
                set_interface_arguments, gradio('extensions_menu', 'bool_menu'), None).then(
                None, None, None, js='() => {document.body.innerHTML=\'<h1 style="font-family:monospace;padding-top:20%;margin:0;height:100vh;color:lightgray;text-align:center;background:var(--body-background-fill)">Reloading...</h1>\'; setTimeout(function(){location.reload()},2500); return []}')


def handle_save_settings(state, preset, extensions, show_controls, theme):
    contents = ui.save_settings(state, preset, extensions, show_controls, theme, manual_save=True)
    root = str(shared.user_data_dir) + "/"
    return [
        contents,
        "settings.yaml",
        root,
        root,
        gr.update(visible=True)
    ]


def handle_default_to_notebook_change(show_two_columns, default_input, default_output, default_prompt, notebook_input, notebook_prompt):
    if show_two_columns:
        # Notebook to default
        return [
            gr.update(visible=True),
            gr.update(visible=False),
            notebook_input,
            "",
            gr.update(value=notebook_prompt, choices=utils.get_available_prompts()),
            gr.update(),
            gr.update(),
        ]
    else:
        # Default to notebook
        return [
            gr.update(visible=False),
            gr.update(visible=True),
            gr.update(),
            gr.update(),
            gr.update(),
            default_input,
            gr.update(value=default_prompt, choices=utils.get_available_prompts())
        ]


def apply_model_dir(value):
    if Path(value).is_dir():
        shared.args.model_dir = value
        shared.user_config = shared.load_user_config()
        models = utils.get_available_models()
        return gr.update(choices=models), gr.update(choices=['None'] + models)

    return gr.update(), gr.update()


def set_interface_arguments(extensions, bool_active):
    shared.args.extensions = extensions

    bool_list = get_boolean_arguments()

    for k in bool_list:
        setattr(shared.args, k, False)
    for k in bool_active:
        setattr(shared.args, k, True)

    shared.need_restart = True


def get_boolean_arguments(active=False):
    cmd_list = vars(shared.args)
    bool_list = sorted([k for k in cmd_list if type(cmd_list[k]) is bool and k not in ui.list_model_elements()])
    bool_active = [k for k in bool_list if vars(shared.args)[k]]

    if active:
        return bool_active
    else:
        return bool_list
