import gradio as gr

from modules import shared, ui, utils
from modules.github import clone_or_pull_repository
from modules.utils import gradio


def create_ui():
    with gr.Tab("Session", elem_id="session-tab"):
        modes = ["default", "notebook", "chat"]
        current_mode = "default"
        for mode in modes[1:]:
            if getattr(shared.args, mode):
                current_mode = mode
                break

        cmd_list = vars(shared.args)
        bool_list = sorted([k for k in cmd_list if type(cmd_list[k]) is bool and k not in modes + ui.list_model_elements()])
        bool_active = [k for k in bool_list if vars(shared.args)[k]]

        with gr.Row():

            with gr.Column():
                with gr.Row():
                    shared.gradio['interface_modes_menu'] = gr.Dropdown(choices=modes, value=current_mode, label="Mode", elem_classes='slim-dropdown')
                    shared.gradio['reset_interface'] = gr.Button("Apply and restart", elem_classes="small-button", variant="primary")
                    shared.gradio['toggle_dark_mode'] = gr.Button('Toggle 💡', elem_classes="small-button")

                with gr.Row():
                    with gr.Column():
                        shared.gradio['extensions_menu'] = gr.CheckboxGroup(choices=utils.get_available_extensions(), value=shared.args.extensions, label="Available extensions", info='Note that some of these extensions may require manually installing Python requirements through the command: pip install -r extensions/extension_name/requirements.txt', elem_classes='checkboxgroup-table')

                    with gr.Column():
                        shared.gradio['bool_menu'] = gr.CheckboxGroup(choices=bool_list, value=bool_active, label="Boolean command-line flags", elem_classes='checkboxgroup-table')

            with gr.Column():
                if not shared.args.multi_user:
                    shared.gradio['save_session'] = gr.Button('Save session', elem_id="save_session")
                    shared.gradio['load_session'] = gr.File(type='binary', file_types=['.json'], label="Upload Session JSON")

                extension_name = gr.Textbox(lines=1, label='Install or update an extension', info='Enter the GitHub URL below and press Enter. For a list of extensions, see: https://github.com/oobabooga/text-generation-webui-extensions ⚠️  WARNING ⚠️ : extensions can execute arbitrary code. Make sure to inspect their source code before activating them.')
                extension_status = gr.Markdown()

        extension_name.submit(
            clone_or_pull_repository, extension_name, extension_status, show_progress=False).then(
            lambda: gr.update(choices=utils.get_available_extensions(), value=shared.args.extensions), None, gradio('extensions_menu'))

        # Reset interface event
        shared.gradio['reset_interface'].click(
            set_interface_arguments, gradio('interface_modes_menu', 'extensions_menu', 'bool_menu'), None).then(
            lambda: None, None, None, _js='() => {document.body.innerHTML=\'<h1 style="font-family:monospace;padding-top:20%;margin:0;height:100vh;color:lightgray;text-align:center;background:var(--body-background-fill)">Reloading...</h1>\'; setTimeout(function(){location.reload()},2500); return []}')

        shared.gradio['toggle_dark_mode'].click(lambda: None, None, None, _js='() => {document.getElementsByTagName("body")[0].classList.toggle("dark")}')


def set_interface_arguments(interface_mode, extensions, bool_active):
    modes = ["default", "notebook", "chat", "cai_chat"]
    cmd_list = vars(shared.args)
    bool_list = [k for k in cmd_list if type(cmd_list[k]) is bool and k not in modes]

    shared.args.extensions = extensions
    for k in modes[1:]:
        setattr(shared.args, k, False)
    if interface_mode != "default":
        setattr(shared.args, interface_mode, True)
    for k in bool_list:
        setattr(shared.args, k, False)
    for k in bool_active:
        setattr(shared.args, k, True)

    shared.need_restart = True
