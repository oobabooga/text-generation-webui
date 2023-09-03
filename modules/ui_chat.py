import json
from functools import partial
from pathlib import Path

import gradio as gr
from PIL import Image

from modules import chat, prompts, shared, ui, utils
from modules.html_generator import chat_html_wrapper
from modules.text_generation import stop_everything_event
from modules.utils import gradio

inputs = ('Chat input', 'interface_state')
reload_arr = ('history', 'name1', 'name2', 'mode', 'chat_style')
clear_arr = ('Clear history-confirm', 'Clear history', 'Clear history-cancel')


def create_ui():
    shared.gradio['Chat input'] = gr.State()
    shared.gradio['dummy'] = gr.State()
    shared.gradio['history'] = gr.State({'internal': [], 'visible': []})

    with gr.Tab('Chat', elem_id='chat-tab'):
        shared.gradio['display'] = gr.HTML(value=chat_html_wrapper({'internal': [], 'visible': []}, shared.settings['name1'], shared.settings['name2'], 'chat', 'cai-chat'))
        shared.gradio['textbox'] = gr.Textbox(label='', placeholder='Send a message', elem_id='chat-input')
        shared.gradio['show_controls'] = gr.Checkbox(value=shared.settings['show_controls'], label='Show controls (Ctrl+S)', elem_id='show-controls')
        shared.gradio['typing-dots'] = gr.HTML(value='<div class="typing"><span></span><span class="dot1"></span><span class="dot2"></span></div>', label='typing', elem_id='typing-container')

        with gr.Row():
            shared.gradio['Stop'] = gr.Button('Stop', elem_id='stop')
            shared.gradio['Generate'] = gr.Button('Generate', elem_id='Generate', variant='primary')
            shared.gradio['Continue'] = gr.Button('Continue')

        with gr.Row():
            shared.gradio['Impersonate'] = gr.Button('Impersonate')
            shared.gradio['Regenerate'] = gr.Button('Regenerate')
            shared.gradio['Remove last'] = gr.Button('Remove last', elem_classes=['button_nowrap'])

        with gr.Row():
            shared.gradio['Copy last reply'] = gr.Button('Copy last reply')
            shared.gradio['Replace last reply'] = gr.Button('Replace last reply')
            shared.gradio['Send dummy message'] = gr.Button('Send dummy message')
            shared.gradio['Send dummy reply'] = gr.Button('Send dummy reply')

        with gr.Row():
            shared.gradio['Clear history'] = gr.Button('Clear history')
            shared.gradio['Clear history-confirm'] = gr.Button('Confirm', variant='stop', visible=False)
            shared.gradio['Clear history-cancel'] = gr.Button('Cancel', visible=False)

        with gr.Row():
            shared.gradio['send-chat-to-default'] = gr.Button('Send to default')
            shared.gradio['send-chat-to-notebook'] = gr.Button('Send to notebook')

        with gr.Row():
            shared.gradio['start_with'] = gr.Textbox(label='Start reply with', placeholder='Sure thing!', value=shared.settings['start_with'])

        with gr.Row():
            shared.gradio['mode'] = gr.Radio(choices=['chat', 'chat-instruct', 'instruct'], value=shared.settings['mode'] if shared.settings['mode'] in ['chat', 'instruct', 'chat-instruct'] else 'chat', label='Mode', info='Defines how the chat prompt is generated. In instruct and chat-instruct modes, the instruction template selected under Parameters > Instruction template must match the current model.', elem_id='chat-mode')
            shared.gradio['chat_style'] = gr.Dropdown(choices=utils.get_available_chat_styles(), label='Chat style', value=shared.settings['chat_style'], visible=shared.settings['mode'] != 'instruct')


def create_chat_settings_ui():
    with gr.Tab('Character'):
        with gr.Row():
            with gr.Column(scale=8):
                with gr.Row():
                    shared.gradio['character_menu'] = gr.Dropdown(value='None', choices=utils.get_available_characters(), label='Character', elem_id='character-menu', info='Used in chat and chat-instruct modes.', elem_classes='slim-dropdown')
                    ui.create_refresh_button(shared.gradio['character_menu'], lambda: None, lambda: {'choices': utils.get_available_characters()}, 'refresh-button')
                    shared.gradio['save_character'] = gr.Button('💾', elem_classes='refresh-button')
                    shared.gradio['delete_character'] = gr.Button('🗑️', elem_classes='refresh-button')

                shared.gradio['name1'] = gr.Textbox(value=shared.settings['name1'], lines=1, label='Your name')
                shared.gradio['name2'] = gr.Textbox(value=shared.settings['name2'], lines=1, label='Character\'s name')
                shared.gradio['context'] = gr.Textbox(value=shared.settings['context'], lines=10, label='Context', elem_classes=['add_scrollbar'])
                shared.gradio['greeting'] = gr.Textbox(value=shared.settings['greeting'], lines=5, label='Greeting', elem_classes=['add_scrollbar'])

            with gr.Column(scale=1):
                shared.gradio['character_picture'] = gr.Image(label='Character picture', type='pil')
                shared.gradio['your_picture'] = gr.Image(label='Your picture', type='pil', value=Image.open(Path('cache/pfp_me.png')) if Path('cache/pfp_me.png').exists() else None)

    with gr.Tab('Instruction template'):
        with gr.Row():
            with gr.Row():
                shared.gradio['instruction_template'] = gr.Dropdown(choices=utils.get_available_instruction_templates(), label='Instruction template', value='None', info='Change this according to the model/LoRA that you are using. Used in instruct and chat-instruct modes.', elem_classes='slim-dropdown')
                ui.create_refresh_button(shared.gradio['instruction_template'], lambda: None, lambda: {'choices': utils.get_available_instruction_templates()}, 'refresh-button')
                shared.gradio['save_template'] = gr.Button('💾', elem_classes='refresh-button')
                shared.gradio['delete_template'] = gr.Button('🗑️ ', elem_classes='refresh-button')

        shared.gradio['name1_instruct'] = gr.Textbox(value='', lines=2, label='User string')
        shared.gradio['name2_instruct'] = gr.Textbox(value='', lines=1, label='Bot string')
        shared.gradio['context_instruct'] = gr.Textbox(value='', lines=4, label='Context')
        shared.gradio['turn_template'] = gr.Textbox(value='', lines=1, label='Turn template', info='Used to precisely define the placement of spaces and new line characters in instruction prompts.')
        with gr.Row():
            shared.gradio['send_instruction_to_default'] = gr.Button('Send to default', elem_classes=['small-button'])
            shared.gradio['send_instruction_to_notebook'] = gr.Button('Send to notebook', elem_classes=['small-button'])
            shared.gradio['send_instruction_to_negative_prompt'] = gr.Button('Send to negative prompt', elem_classes=['small-button'])

        with gr.Row():
            shared.gradio['chat-instruct_command'] = gr.Textbox(value=shared.settings['chat-instruct_command'], lines=4, label='Command for chat-instruct mode', info='<|character|> gets replaced by the bot name, and <|prompt|> gets replaced by the regular chat prompt.', elem_classes=['add_scrollbar'])

    with gr.Tab('Chat history'):
        with gr.Row():
            with gr.Column():
                shared.gradio['save_chat_history'] = gr.Button(value='Save history')

            with gr.Column():
                shared.gradio['load_chat_history'] = gr.File(type='binary', file_types=['.json', '.txt'], label='Upload History JSON')

    with gr.Tab('Upload character'):
        with gr.Tab('YAML or JSON'):
            with gr.Row():
                shared.gradio['upload_json'] = gr.File(type='binary', file_types=['.json', '.yaml'], label='JSON or YAML File')
                shared.gradio['upload_img_bot'] = gr.Image(type='pil', label='Profile Picture (optional)')

            shared.gradio['Submit character'] = gr.Button(value='Submit', interactive=False)

        with gr.Tab('TavernAI PNG'):
            with gr.Row():
                with gr.Column():
                    shared.gradio['upload_img_tavern'] = gr.Image(type='pil', label='TavernAI PNG File', elem_id='upload_img_tavern')
                    shared.gradio['tavern_json'] = gr.State()
                with gr.Column():
                    shared.gradio['tavern_name'] = gr.Textbox(value='', lines=1, label='Name', interactive=False)
                    shared.gradio['tavern_desc'] = gr.Textbox(value='', lines=4, max_lines=4, label='Description', interactive=False)

            shared.gradio['Submit tavern character'] = gr.Button(value='Submit', interactive=False)


def create_event_handlers():

    # Obsolete variables, kept for compatibility with old extensions
    shared.input_params = gradio(inputs)
    shared.reload_inputs = gradio(reload_arr)

    shared.gradio['Generate'].click(
        ui.gather_interface_values, gradio(shared.input_elements), gradio('interface_state')).then(
        lambda x: (x, ''), gradio('textbox'), gradio('Chat input', 'textbox'), show_progress=False).then(
        chat.generate_chat_reply_wrapper, gradio(inputs), gradio('display', 'history'), show_progress=False).then(
        ui.gather_interface_values, gradio(shared.input_elements), gradio('interface_state')).then(
        chat.save_persistent_history, gradio('history', 'character_menu', 'mode'), None).then(
        lambda: None, None, None, _js=f'() => {{{ui.audio_notification_js}}}')

    shared.gradio['textbox'].submit(
        ui.gather_interface_values, gradio(shared.input_elements), gradio('interface_state')).then(
        lambda x: (x, ''), gradio('textbox'), gradio('Chat input', 'textbox'), show_progress=False).then(
        chat.generate_chat_reply_wrapper, gradio(inputs), gradio('display', 'history'), show_progress=False).then(
        ui.gather_interface_values, gradio(shared.input_elements), gradio('interface_state')).then(
        chat.save_persistent_history, gradio('history', 'character_menu', 'mode'), None).then(
        lambda: None, None, None, _js=f'() => {{{ui.audio_notification_js}}}')

    shared.gradio['Regenerate'].click(
        ui.gather_interface_values, gradio(shared.input_elements), gradio('interface_state')).then(
        partial(chat.generate_chat_reply_wrapper, regenerate=True), gradio(inputs), gradio('display', 'history'), show_progress=False).then(
        ui.gather_interface_values, gradio(shared.input_elements), gradio('interface_state')).then(
        chat.save_persistent_history, gradio('history', 'character_menu', 'mode'), None).then(
        lambda: None, None, None, _js=f'() => {{{ui.audio_notification_js}}}')

    shared.gradio['Continue'].click(
        ui.gather_interface_values, gradio(shared.input_elements), gradio('interface_state')).then(
        partial(chat.generate_chat_reply_wrapper, _continue=True), gradio(inputs), gradio('display', 'history'), show_progress=False).then(
        ui.gather_interface_values, gradio(shared.input_elements), gradio('interface_state')).then(
        chat.save_persistent_history, gradio('history', 'character_menu', 'mode'), None).then(
        lambda: None, None, None, _js=f'() => {{{ui.audio_notification_js}}}')

    shared.gradio['Impersonate'].click(
        ui.gather_interface_values, gradio(shared.input_elements), gradio('interface_state')).then(
        lambda x: x, gradio('textbox'), gradio('Chat input'), show_progress=False).then(
        chat.impersonate_wrapper, gradio(inputs), gradio('textbox'), show_progress=False).then(
        ui.gather_interface_values, gradio(shared.input_elements), gradio('interface_state')).then(
        lambda: None, None, None, _js=f'() => {{{ui.audio_notification_js}}}')

    shared.gradio['Replace last reply'].click(
        ui.gather_interface_values, gradio(shared.input_elements), gradio('interface_state')).then(
        chat.replace_last_reply, gradio('textbox', 'interface_state'), gradio('history')).then(
        lambda: '', None, gradio('textbox'), show_progress=False).then(
        chat.redraw_html, gradio(reload_arr), gradio('display')).then(
        chat.save_persistent_history, gradio('history', 'character_menu', 'mode'), None)

    shared.gradio['Send dummy message'].click(
        ui.gather_interface_values, gradio(shared.input_elements), gradio('interface_state')).then(
        chat.send_dummy_message, gradio('textbox', 'interface_state'), gradio('history')).then(
        lambda: '', None, gradio('textbox'), show_progress=False).then(
        chat.redraw_html, gradio(reload_arr), gradio('display')).then(
        chat.save_persistent_history, gradio('history', 'character_menu', 'mode'), None)

    shared.gradio['Send dummy reply'].click(
        ui.gather_interface_values, gradio(shared.input_elements), gradio('interface_state')).then(
        chat.send_dummy_reply, gradio('textbox', 'interface_state'), gradio('history')).then(
        lambda: '', None, gradio('textbox'), show_progress=False).then(
        chat.redraw_html, gradio(reload_arr), gradio('display')).then(
        chat.save_persistent_history, gradio('history', 'character_menu', 'mode'), None)

    shared.gradio['Clear history'].click(lambda: [gr.update(visible=True), gr.update(visible=False), gr.update(visible=True)], None, gradio(clear_arr))
    shared.gradio['Clear history-cancel'].click(lambda: [gr.update(visible=False), gr.update(visible=True), gr.update(visible=False)], None, gradio(clear_arr))
    shared.gradio['Clear history-confirm'].click(
        ui.gather_interface_values, gradio(shared.input_elements), gradio('interface_state')).then(
        lambda: [gr.update(visible=False), gr.update(visible=True), gr.update(visible=False)], None, gradio(clear_arr)).then(
        chat.clear_chat_log, gradio('interface_state'), gradio('history')).then(
        chat.redraw_html, gradio(reload_arr), gradio('display')).then(
        chat.save_persistent_history, gradio('history', 'character_menu', 'mode'), None)

    shared.gradio['Remove last'].click(
        ui.gather_interface_values, gradio(shared.input_elements), gradio('interface_state')).then(
        chat.remove_last_message, gradio('history'), gradio('textbox', 'history'), show_progress=False).then(
        chat.redraw_html, gradio(reload_arr), gradio('display')).then(
        chat.save_persistent_history, gradio('history', 'character_menu', 'mode'), None)

    shared.gradio['character_menu'].change(
        partial(chat.load_character, instruct=False), gradio('character_menu', 'name1', 'name2'), gradio('name1', 'name2', 'character_picture', 'greeting', 'context', 'dummy')).then(
        ui.gather_interface_values, gradio(shared.input_elements), gradio('interface_state')).then(
        chat.load_persistent_history, gradio('interface_state'), gradio('history')).then(
        chat.redraw_html, gradio(reload_arr), gradio('display'))

    shared.gradio['Stop'].click(
        stop_everything_event, None, None, queue=False).then(
        chat.redraw_html, gradio(reload_arr), gradio('display'))

    shared.gradio['mode'].change(
        lambda x: gr.update(visible=x != 'instruct'), gradio('mode'), gradio('chat_style'), show_progress=False).then(
        chat.redraw_html, gradio(reload_arr), gradio('display'))

    shared.gradio['chat_style'].change(chat.redraw_html, gradio(reload_arr), gradio('display'))
    shared.gradio['instruction_template'].change(
        partial(chat.load_character, instruct=True), gradio('instruction_template', 'name1_instruct', 'name2_instruct'), gradio('name1_instruct', 'name2_instruct', 'dummy', 'dummy', 'context_instruct', 'turn_template'))

    shared.gradio['load_chat_history'].upload(
        chat.load_history, gradio('load_chat_history', 'history'), gradio('history')).then(
        chat.redraw_html, gradio(reload_arr), gradio('display')).then(
        lambda: None, None, None, _js=f'() => {{{ui.switch_tabs_js}; switch_to_chat()}}')

    shared.gradio['Copy last reply'].click(chat.send_last_reply_to_input, gradio('history'), gradio('textbox'), show_progress=False)

    # Save/delete a character
    shared.gradio['save_character'].click(
        lambda x: x, gradio('name2'), gradio('save_character_filename')).then(
        lambda: gr.update(visible=True), None, gradio('character_saver'))

    shared.gradio['delete_character'].click(lambda: gr.update(visible=True), None, gradio('character_deleter'))

    shared.gradio['save_template'].click(
        lambda: 'My Template.yaml', None, gradio('save_filename')).then(
        lambda: 'instruction-templates/', None, gradio('save_root')).then(
        chat.generate_instruction_template_yaml, gradio('name1_instruct', 'name2_instruct', 'context_instruct', 'turn_template'), gradio('save_contents')).then(
        lambda: gr.update(visible=True), None, gradio('file_saver'))

    shared.gradio['delete_template'].click(
        lambda x: f'{x}.yaml', gradio('instruction_template'), gradio('delete_filename')).then(
        lambda: 'instruction-templates/', None, gradio('delete_root')).then(
        lambda: gr.update(visible=True), None, gradio('file_deleter'))

    shared.gradio['save_chat_history'].click(
        lambda x: json.dumps(x, indent=4), gradio('history'), gradio('temporary_text')).then(
        None, gradio('temporary_text', 'character_menu', 'mode'), None, _js=f'(hist, char, mode) => {{{ui.save_files_js}; saveHistory(hist, char, mode)}}')

    shared.gradio['Submit character'].click(
        chat.upload_character, gradio('upload_json', 'upload_img_bot'), gradio('character_menu')).then(
        lambda: None, None, None, _js=f'() => {{{ui.switch_tabs_js}; switch_to_character()}}')

    shared.gradio['Submit tavern character'].click(
        chat.upload_tavern_character, gradio('upload_img_tavern', 'tavern_json'), gradio('character_menu')).then(
        lambda: None, None, None, _js=f'() => {{{ui.switch_tabs_js}; switch_to_character()}}')

    shared.gradio['upload_json'].upload(lambda: gr.update(interactive=True), None, gradio('Submit character'))
    shared.gradio['upload_json'].clear(lambda: gr.update(interactive=False), None, gradio('Submit character'))
    shared.gradio['upload_img_tavern'].upload(chat.check_tavern_character, gradio('upload_img_tavern'), gradio('tavern_name', 'tavern_desc', 'tavern_json', 'Submit tavern character'), show_progress=False)
    shared.gradio['upload_img_tavern'].clear(lambda: (None, None, None, gr.update(interactive=False)), None, gradio('tavern_name', 'tavern_desc', 'tavern_json', 'Submit tavern character'), show_progress=False)
    shared.gradio['your_picture'].change(
        chat.upload_your_profile_picture, gradio('your_picture'), None).then(
        partial(chat.redraw_html, reset_cache=True), gradio(reload_arr), gradio('display'))

    shared.gradio['send_instruction_to_default'].click(
        prompts.load_instruction_prompt_simple, gradio('instruction_template'), gradio('textbox-default')).then(
        lambda: None, None, None, _js=f'() => {{{ui.switch_tabs_js}; switch_to_default()}}')

    shared.gradio['send_instruction_to_notebook'].click(
        prompts.load_instruction_prompt_simple, gradio('instruction_template'), gradio('textbox-notebook')).then(
        lambda: None, None, None, _js=f'() => {{{ui.switch_tabs_js}; switch_to_notebook()}}')

    shared.gradio['send_instruction_to_negative_prompt'].click(
        prompts.load_instruction_prompt_simple, gradio('instruction_template'), gradio('negative_prompt')).then(
        lambda: None, None, None, _js=f'() => {{{ui.switch_tabs_js}; switch_to_generation_parameters()}}')

    shared.gradio['send-chat-to-default'].click(
        ui.gather_interface_values, gradio(shared.input_elements), gradio('interface_state')).then(
        partial(chat.generate_chat_prompt, '', _continue=True), gradio('interface_state'), gradio('textbox-default')).then(
        lambda: None, None, None, _js=f'() => {{{ui.switch_tabs_js}; switch_to_default()}}')

    shared.gradio['send-chat-to-notebook'].click(
        ui.gather_interface_values, gradio(shared.input_elements), gradio('interface_state')).then(
        partial(chat.generate_chat_prompt, '', _continue=True), gradio('interface_state'), gradio('textbox-notebook')).then(
        lambda: None, None, None, _js=f'() => {{{ui.switch_tabs_js}; switch_to_notebook()}}')

    shared.gradio['show_controls'].change(None, gradio('show_controls'), None, _js=f'(x) => {{{ui.show_controls_js}; toggle_controls(x)}}')
