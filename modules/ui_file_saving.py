import traceback

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
            shared.gradio['save_cancel'] = gr.Button('Cancel', elem_classes="small-button")
            shared.gradio['save_confirm'] = gr.Button('Save', elem_classes="small-button", variant='primary', interactive=not mu)

    # Text file deleter
    with gr.Group(visible=False, elem_classes='file-saver') as shared.gradio['file_deleter']:
        shared.gradio['delete_filename'] = gr.Textbox(lines=1, label='File name')
        shared.gradio['delete_root'] = gr.Textbox(lines=1, label='File folder', info='For reference. Unchangeable.', interactive=False)
        with gr.Row():
            shared.gradio['delete_cancel'] = gr.Button('Cancel', elem_classes="small-button")
            shared.gradio['delete_confirm'] = gr.Button('Delete', elem_classes="small-button", variant='stop', interactive=not mu)

    # Character saver/deleter
    with gr.Group(visible=False, elem_classes='file-saver') as shared.gradio['character_saver']:
        shared.gradio['save_character_filename'] = gr.Textbox(lines=1, label='File name', info='The character will be saved to your characters/ folder with this base filename.')
        with gr.Row():
            shared.gradio['save_character_cancel'] = gr.Button('Cancel', elem_classes="small-button")
            shared.gradio['save_character_confirm'] = gr.Button('Save', elem_classes="small-button", variant='primary', interactive=not mu)

    with gr.Group(visible=False, elem_classes='file-saver') as shared.gradio['character_deleter']:
        gr.Markdown('Confirm the character deletion?')
        with gr.Row():
            shared.gradio['delete_character_cancel'] = gr.Button('Cancel', elem_classes="small-button")
            shared.gradio['delete_character_confirm'] = gr.Button('Delete', elem_classes="small-button", variant='stop', interactive=not mu)

    # Preset saver
    with gr.Group(visible=False, elem_classes='file-saver') as shared.gradio['preset_saver']:
        shared.gradio['save_preset_filename'] = gr.Textbox(lines=1, label='File name', info='The preset will be saved to your presets/ folder with this base filename.')
        shared.gradio['save_preset_contents'] = gr.Textbox(lines=10, label='File contents')
        with gr.Row():
            shared.gradio['save_preset_cancel'] = gr.Button('Cancel', elem_classes="small-button")
            shared.gradio['save_preset_confirm'] = gr.Button('Save', elem_classes="small-button", variant='primary', interactive=not mu)


def create_event_handlers():
    shared.gradio['save_preset'].click(
        ui.gather_interface_values, gradio(shared.input_elements), gradio('interface_state')).then(
        handle_save_preset_click, gradio('interface_state'), gradio('save_preset_contents', 'save_preset_filename', 'preset_saver'), show_progress=False)

    shared.gradio['delete_preset'].click(handle_delete_preset_click, gradio('preset_menu'), gradio('delete_filename', 'delete_root', 'file_deleter'), show_progress=False)
    shared.gradio['save_grammar'].click(handle_save_grammar_click, gradio('grammar_string'), gradio('save_contents', 'save_filename', 'save_root', 'file_saver'), show_progress=False)
    shared.gradio['delete_grammar'].click(handle_delete_grammar_click, gradio('grammar_file'), gradio('delete_filename', 'delete_root', 'file_deleter'), show_progress=False)

    shared.gradio['save_preset_confirm'].click(handle_save_preset_confirm_click, gradio('save_preset_filename', 'save_preset_contents'), gradio('preset_menu', 'preset_saver'), show_progress=False)
    shared.gradio['save_confirm'].click(handle_save_confirm_click, gradio('save_root', 'save_filename', 'save_contents'), gradio('file_saver'), show_progress=False)
    shared.gradio['delete_confirm'].click(handle_delete_confirm_click, gradio('delete_root', 'delete_filename'), gradio('file_deleter'), show_progress=False)
    shared.gradio['save_character_confirm'].click(handle_save_character_confirm_click, gradio('name2', 'greeting', 'context', 'character_picture', 'save_character_filename'), gradio('character_menu', 'character_saver'), show_progress=False)
    shared.gradio['delete_character_confirm'].click(handle_delete_character_confirm_click, gradio('character_menu'), gradio('character_menu', 'character_deleter'), show_progress=False)

    shared.gradio['save_preset_cancel'].click(lambda: gr.update(visible=False), None, gradio('preset_saver'), show_progress=False)
    shared.gradio['save_cancel'].click(lambda: gr.update(visible=False), None, gradio('file_saver'))
    shared.gradio['delete_cancel'].click(lambda: gr.update(visible=False), None, gradio('file_deleter'))
    shared.gradio['save_character_cancel'].click(lambda: gr.update(visible=False), None, gradio('character_saver'), show_progress=False)
    shared.gradio['delete_character_cancel'].click(lambda: gr.update(visible=False), None, gradio('character_deleter'), show_progress=False)


def handle_save_preset_confirm_click(filename, contents):
    try:
        utils.save_file(f"presets/{filename}.yaml", contents)
        available_presets = utils.get_available_presets()
        output = gr.update(choices=available_presets, value=filename),
    except Exception:
        output = gr.update()
        traceback.print_exc()

    return [
        output,
        gr.update(visible=False)
    ]


def handle_save_confirm_click(root, filename, contents):
    try:
        utils.save_file(root + filename, contents)
    except Exception:
        traceback.print_exc()

    return gr.update(visible=False)


def handle_delete_confirm_click(root, filename):
    try:
        utils.delete_file(root + filename)
    except Exception:
        traceback.print_exc()

    return gr.update(visible=False)


def handle_save_character_confirm_click(name2, greeting, context, character_picture, filename):
    try:
        chat.save_character(name2, greeting, context, character_picture, filename)
        available_characters = utils.get_available_characters()
        output = gr.update(choices=available_characters, value=filename)
    except Exception:
        output = gr.update()
        traceback.print_exc()

    return [
        output,
        gr.update(visible=False)
    ]


def handle_delete_character_confirm_click(character):
    try:
        index = str(utils.get_available_characters().index(character))
        chat.delete_character(character)
        output = chat.update_character_menu_after_deletion(index)
    except Exception:
        output = gr.update()
        traceback.print_exc()

    return [
        output,
        gr.update(visible=False)
    ]


def handle_save_preset_click(state):
    contents = presets.generate_preset_yaml(state)
    return [
        contents,
        "My Preset",
        gr.update(visible=True)
    ]


def handle_delete_preset_click(preset):
    return [
        f"{preset}.yaml",
        "presets/",
        gr.update(visible=True)
    ]


def handle_save_grammar_click(grammar_string):
    return [
        grammar_string,
        "My Fancy Grammar.gbnf",
        "grammars/",
        gr.update(visible=True)
    ]


def handle_delete_grammar_click(grammar_file):
    return [
        grammar_file,
        "grammars/",
        gr.update(visible=True)
    ]
