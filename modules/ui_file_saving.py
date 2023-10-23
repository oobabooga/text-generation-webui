import gradio as gr

from modules import chat, presets, shared, ui, utils
from modules.utils import gradio


def create_ui():
    mu = shared.args.multi_user

    # Text file saver
    with gr.Group(visible=False, elem_classes='file-saver') as shared.gradio['file_saver']:
        shared.gradio['save_filename'] = gr.Textbox(lines=1, label='File name')
        shared.gradio['save_root'] = gr.Textbox(lines=1, label='File folder', info='For reference. Unchangeable.', interactive=False)
        shared.gradio['save_contents'] = gr.Textbox(lines=10, label='File contents')
        with gr.Row():
            shared.gradio['save_confirm'] = gr.Button('Save', elem_classes="small-button", variant='primary', interactive=not mu)
            shared.gradio['save_cancel'] = gr.Button('Cancel', elem_classes="small-button")

    # Text file deleter
    with gr.Group(visible=False, elem_classes='file-saver') as shared.gradio['file_deleter']:
        shared.gradio['delete_filename'] = gr.Textbox(lines=1, label='File name')
        shared.gradio['delete_root'] = gr.Textbox(lines=1, label='File folder', info='For reference. Unchangeable.', interactive=False)
        with gr.Row():
            shared.gradio['delete_confirm'] = gr.Button('Delete', elem_classes="small-button", variant='stop', interactive=not mu)
            shared.gradio['delete_cancel'] = gr.Button('Cancel', elem_classes="small-button")

    # Character saver/deleter
    with gr.Group(visible=False, elem_classes='file-saver') as shared.gradio['character_saver']:
        shared.gradio['save_character_filename'] = gr.Textbox(lines=1, label='File name', info='The character will be saved to your characters/ folder with this base filename.')
        with gr.Row():
            shared.gradio['save_character_confirm'] = gr.Button('Save', elem_classes="small-button", variant='primary', interactive=not mu)
            shared.gradio['save_character_cancel'] = gr.Button('Cancel', elem_classes="small-button")

    with gr.Group(visible=False, elem_classes='file-saver') as shared.gradio['character_deleter']:
        gr.Markdown('Confirm the character deletion?')
        with gr.Row():
            shared.gradio['delete_character_confirm'] = gr.Button('Delete', elem_classes="small-button", variant='stop', interactive=not mu)
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
        lambda: gr.update(choices=(characters := utils.get_available_characters()), value=characters[0]), None, gradio('character_menu'))

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

    shared.gradio['save_grammar'].click(
        ui.gather_interface_values, gradio(shared.input_elements), gradio('interface_state')).then(
        lambda x: x, gradio('grammar_string'), gradio('save_contents')).then(
        lambda: 'grammars/', None, gradio('save_root')).then(
        lambda: 'My Fancy Grammar.gbnf', None, gradio('save_filename')).then(
        lambda: gr.update(visible=True), None, gradio('file_saver'))

    shared.gradio['delete_grammar'].click(
        lambda x: x, gradio('grammar_file'), gradio('delete_filename')).then(
        lambda: 'grammars/', None, gradio('delete_root')).then(
        lambda: gr.update(visible=True), None, gradio('file_deleter'))
