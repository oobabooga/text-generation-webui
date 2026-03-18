import traceback

import gradio as gr

from modules import chat, presets, shared, ui, utils
from modules.utils import gradio, sanitize_filename


def create_ui():
    mu = shared.args.multi_user

    # Server-side per-session root paths for the generic file saver/deleter.
    # Set by the handler that opens the dialog, read by the confirm handler.
    # Using gr.State so they are session-scoped and safe for multi-user.
    shared.gradio['save_root_state'] = gr.State(None)
    shared.gradio['delete_root_state'] = gr.State(None)

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
        shared.gradio['save_character_filename'] = gr.Textbox(lines=1, label='File name', info=f'The character will be saved to your {shared.user_data_dir}/characters folder with this base filename.')
        with gr.Row():
            shared.gradio['save_character_cancel'] = gr.Button('Cancel', elem_classes="small-button")
            shared.gradio['save_character_confirm'] = gr.Button('Save', elem_classes="small-button", variant='primary', interactive=not mu)

    with gr.Group(visible=False, elem_classes='file-saver') as shared.gradio['character_deleter']:
        gr.Markdown('Confirm the character deletion?')
        with gr.Row():
            shared.gradio['delete_character_cancel'] = gr.Button('Cancel', elem_classes="small-button")
            shared.gradio['delete_character_confirm'] = gr.Button('Delete', elem_classes="small-button", variant='stop', interactive=not mu)

    # User saver/deleter
    with gr.Group(visible=False, elem_classes='file-saver') as shared.gradio['user_saver']:
        shared.gradio['save_user_filename'] = gr.Textbox(lines=1, label='File name', info=f'The user profile will be saved to your {shared.user_data_dir}/users folder with this base filename.')
        with gr.Row():
            shared.gradio['save_user_cancel'] = gr.Button('Cancel', elem_classes="small-button")
            shared.gradio['save_user_confirm'] = gr.Button('Save', elem_classes="small-button", variant='primary', interactive=not mu)

    with gr.Group(visible=False, elem_classes='file-saver') as shared.gradio['user_deleter']:
        gr.Markdown('Confirm the user deletion?')
        with gr.Row():
            shared.gradio['delete_user_cancel'] = gr.Button('Cancel', elem_classes="small-button")
            shared.gradio['delete_user_confirm'] = gr.Button('Delete', elem_classes="small-button", variant='stop', interactive=not mu)

    # Preset saver
    with gr.Group(visible=False, elem_classes='file-saver') as shared.gradio['preset_saver']:
        shared.gradio['save_preset_filename'] = gr.Textbox(lines=1, label='File name', info=f'The preset will be saved to your {shared.user_data_dir}/presets folder with this base filename.')
        shared.gradio['save_preset_contents'] = gr.Textbox(lines=10, label='File contents')
        with gr.Row():
            shared.gradio['save_preset_cancel'] = gr.Button('Cancel', elem_classes="small-button")
            shared.gradio['save_preset_confirm'] = gr.Button('Save', elem_classes="small-button", variant='primary', interactive=not mu)


def create_event_handlers():
    shared.gradio['save_preset'].click(
        ui.gather_interface_values, gradio(shared.input_elements), gradio('interface_state')).then(
        handle_save_preset_click, gradio('interface_state'), gradio('save_preset_contents', 'save_preset_filename', 'preset_saver'), show_progress=False)

    shared.gradio['delete_preset'].click(handle_delete_preset_click, gradio('preset_menu'), gradio('delete_filename', 'delete_root', 'delete_root_state', 'file_deleter'), show_progress=False)
    shared.gradio['save_grammar'].click(handle_save_grammar_click, gradio('grammar_string'), gradio('save_contents', 'save_filename', 'save_root', 'save_root_state', 'file_saver'), show_progress=False)
    shared.gradio['delete_grammar'].click(handle_delete_grammar_click, gradio('grammar_file'), gradio('delete_filename', 'delete_root', 'delete_root_state', 'file_deleter'), show_progress=False)

    shared.gradio['save_preset_confirm'].click(handle_save_preset_confirm_click, gradio('save_preset_filename', 'save_preset_contents'), gradio('preset_menu', 'preset_saver'), show_progress=False)
    shared.gradio['save_confirm'].click(handle_save_confirm_click, gradio('save_root_state', 'save_filename', 'save_contents'), gradio('save_root_state', 'file_saver'), show_progress=False)
    shared.gradio['delete_confirm'].click(handle_delete_confirm_click, gradio('delete_root_state', 'delete_filename'), gradio('delete_root_state', 'file_deleter'), show_progress=False)
    shared.gradio['save_character_confirm'].click(handle_save_character_confirm_click, gradio('name2', 'greeting', 'context', 'character_picture', 'save_character_filename'), gradio('character_menu', 'character_saver'), show_progress=False)
    shared.gradio['delete_character_confirm'].click(handle_delete_character_confirm_click, gradio('character_menu'), gradio('character_menu', 'character_deleter'), show_progress=False)

    shared.gradio['save_preset_cancel'].click(lambda: gr.update(visible=False), None, gradio('preset_saver'), show_progress=False)
    shared.gradio['save_cancel'].click(lambda: gr.update(visible=False), None, gradio('file_saver'))
    shared.gradio['delete_cancel'].click(lambda: gr.update(visible=False), None, gradio('file_deleter'))
    shared.gradio['save_character_cancel'].click(lambda: gr.update(visible=False), None, gradio('character_saver'), show_progress=False)
    shared.gradio['delete_character_cancel'].click(lambda: gr.update(visible=False), None, gradio('character_deleter'), show_progress=False)

    # User save/delete event handlers
    shared.gradio['save_user_confirm'].click(handle_save_user_confirm_click, gradio('name1', 'user_bio', 'your_picture', 'save_user_filename'), gradio('user_menu', 'user_saver'), show_progress=False)
    shared.gradio['delete_user_confirm'].click(handle_delete_user_confirm_click, gradio('user_menu'), gradio('user_menu', 'user_deleter'), show_progress=False)
    shared.gradio['save_user_cancel'].click(lambda: gr.update(visible=False), None, gradio('user_saver'), show_progress=False)
    shared.gradio['delete_user_cancel'].click(lambda: gr.update(visible=False), None, gradio('user_deleter'), show_progress=False)


def handle_save_preset_confirm_click(filename, contents):
    try:
        filename = sanitize_filename(filename)
        utils.save_file(str(shared.user_data_dir / "presets" / f"{filename}.yaml"), contents)
        available_presets = utils.get_available_presets()
        output = gr.update(choices=available_presets, value=filename)
    except Exception:
        output = gr.update()
        traceback.print_exc()

    return [
        output,
        gr.update(visible=False)
    ]


def handle_save_confirm_click(root_state, filename, contents):
    try:
        if root_state is None:
            return None, gr.update(visible=False)

        filename = sanitize_filename(filename)
        utils.save_file(root_state + filename, contents)
    except Exception:
        traceback.print_exc()

    return None, gr.update(visible=False)


def handle_delete_confirm_click(root_state, filename):
    try:
        if root_state is None:
            return None, gr.update(visible=False)

        filename = sanitize_filename(filename)
        utils.delete_file(root_state + filename)
    except Exception:
        traceback.print_exc()

    return None, gr.update(visible=False)


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
    root = str(shared.user_data_dir / "presets") + "/"
    return [
        f"{preset}.yaml",
        root,
        root,
        gr.update(visible=True)
    ]


def handle_save_grammar_click(grammar_string):
    root = str(shared.user_data_dir / "grammars") + "/"
    return [
        grammar_string,
        "My Fancy Grammar.gbnf",
        root,
        root,
        gr.update(visible=True)
    ]


def handle_delete_grammar_click(grammar_file):
    root = str(shared.user_data_dir / "grammars") + "/"
    return [
        grammar_file,
        root,
        root,
        gr.update(visible=True)
    ]


def handle_save_user_confirm_click(name1, user_bio, your_picture, filename):
    try:
        chat.save_user(name1, user_bio, your_picture, filename)
        available_users = utils.get_available_users()
        output = gr.update(choices=available_users, value=filename)
    except Exception:
        output = gr.update()
        traceback.print_exc()

    return [
        output,
        gr.update(visible=False)
    ]


def handle_delete_user_confirm_click(user):
    try:
        index = str(utils.get_available_users().index(user))
        chat.delete_user(user)
        output = chat.update_user_menu_after_deletion(index)
    except Exception:
        output = gr.update()
        traceback.print_exc()

    return [
        output,
        gr.update(visible=False)
    ]
