import copy
import json

import gradio as gr

from modules import chat, presets, shared, ui, ui_chat, utils
from modules.utils import gradio


def create_ui():

    # Text file saver
    with gr.Box(visible=False, elem_classes='file-saver') as shared.gradio['file_saver']:
        shared.gradio['save_filename'] = gr.Textbox(lines=1, label='File name')
        shared.gradio['save_root'] = gr.Textbox(lines=1, label='File folder', info='For reference. Unchangeable.', interactive=False)
        shared.gradio['save_contents'] = gr.Textbox(lines=10, label='File contents')
        with gr.Row():
            shared.gradio['save_confirm'] = gr.Button('Save', elem_classes="small-button")
            shared.gradio['save_cancel'] = gr.Button('Cancel', elem_classes="small-button")

    # Text file deleter
    with gr.Box(visible=False, elem_classes='file-saver') as shared.gradio['file_deleter']:
        shared.gradio['delete_filename'] = gr.Textbox(lines=1, label='File name')
        shared.gradio['delete_root'] = gr.Textbox(lines=1, label='File folder', info='For reference. Unchangeable.', interactive=False)
        with gr.Row():
            shared.gradio['delete_confirm'] = gr.Button('Delete', elem_classes="small-button", variant='stop')
            shared.gradio['delete_cancel'] = gr.Button('Cancel', elem_classes="small-button")

    # Character saver/deleter
    with gr.Box(visible=False, elem_classes='file-saver') as shared.gradio['character_saver']:
        shared.gradio['save_character_filename'] = gr.Textbox(lines=1, label='File name', info='The character will be saved to your characters/ folder with this base filename.')
        with gr.Row():
            shared.gradio['save_character_confirm'] = gr.Button('Save', elem_classes="small-button")
            shared.gradio['save_character_cancel'] = gr.Button('Cancel', elem_classes="small-button")

    with gr.Box(visible=False, elem_classes='file-saver') as shared.gradio['character_deleter']:
        gr.Markdown('Confirm the character deletion?')
        with gr.Row():
            shared.gradio['delete_character_confirm'] = gr.Button('Delete', elem_classes="small-button", variant='stop')
            shared.gradio['delete_character_cancel'] = gr.Button('Cancel', elem_classes="small-button")


def create_event_handlers():
    shared.gradio['save_confirm'].click(
        lambda x, y, z: utils.save_file(x + y, z), gradio('save_root', 'save_filename', 'save_contents'), None).then(
        lambda: gr.update(visible=False), None, gradio('file_saver'))

    shared.gradio['delete_confirm'].click(
        lambda x, y: utils.delete_file(x + y), gradio('delete_root', 'delete_filename'), None).then(
        lambda: gr.update(visible=False), None, gradio('file_deleter'))

    shared.gradio['delete_cancel'].click(lambda: gr.update(visible=False), None, gradio('file_deleter'))
    shared.gradio['save_cancel'].click(lambda: gr.update(visible=False), None, gradio('file_saver'))

    shared.gradio['save_character_confirm'].click(
        chat.save_character, gradio('name2', 'greeting', 'context', 'character_picture', 'save_character_filename'), None).then(
        lambda: gr.update(visible=False), None, gradio('character_saver')).then(
        lambda x: gr.update(choices=utils.get_available_characters(), value=x), gradio('save_character_filename'), gradio('character_menu'))

    shared.gradio['delete_character_confirm'].click(
        chat.delete_character, gradio('character_menu'), None).then(
        lambda: gr.update(visible=False), None, gradio('character_deleter')).then(
        lambda: gr.update(choices=utils.get_available_characters(), value="None"), None, gradio('character_menu'))

    shared.gradio['save_character_cancel'].click(lambda: gr.update(visible=False), None, gradio('character_saver'))
    shared.gradio['delete_character_cancel'].click(lambda: gr.update(visible=False), None, gradio('character_deleter'))

    shared.gradio['save_preset'].click(
        ui.gather_interface_values, gradio(shared.input_elements), gradio('interface_state')).then(
        presets.generate_preset_yaml, gradio('interface_state'), gradio('save_contents')).then(
        lambda: 'presets/', None, gradio('save_root')).then(
        lambda: 'My Preset.yaml', None, gradio('save_filename')).then(
        lambda: gr.update(visible=True), None, gradio('file_saver'))

    shared.gradio['delete_preset'].click(
        lambda x: f'{x}.yaml', gradio('preset_menu'), gradio('delete_filename')).then(
        lambda: 'presets/', None, gradio('delete_root')).then(
        lambda: gr.update(visible=True), None, gradio('file_deleter'))

    if not shared.args.multi_user:
        shared.gradio['save_session'].click(
            ui.gather_interface_values, gradio(shared.input_elements), gradio('interface_state')).then(
            save_session, gradio('interface_state'), gradio('temporary_text')).then(
            None, gradio('temporary_text'), None, _js=f"(contents) => {{{ui.save_files_js}; saveSession(contents)}}")

        shared.gradio['load_session'].upload(
            ui.gather_interface_values, gradio(shared.input_elements), gradio('interface_state')).then(
            load_session, gradio('load_session', 'interface_state'), gradio('interface_state')).then(
            ui.apply_interface_values, gradio('interface_state'), gradio(ui.list_interface_input_elements()), show_progress=False).then(
            chat.redraw_html, gradio(ui_chat.reload_arr), gradio('display')).then(
            None, None, None, _js='() => {alert("The session has been loaded.")}')


def load_session(file, state):
    decoded_file = file if isinstance(file, str) else file.decode('utf-8')
    data = json.loads(decoded_file)

    if 'character_menu' in data and state.get('character_menu') != data.get('character_menu'):
        shared.session_is_loading = True

    state.update(data)
    return state


def save_session(state):
    output = copy.deepcopy(state)
    for key in ['prompt_menu-default', 'prompt_menu-notebook']:
        del output[key]

    return json.dumps(output, indent=4)
