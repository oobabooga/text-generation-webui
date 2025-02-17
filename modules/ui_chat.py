import json
from functools import partial
from pathlib import Path

import gradio as gr
from PIL import Image

from modules import chat, shared, ui, utils
from modules.html_generator import chat_html_wrapper
from modules.text_generation import stop_everything_event
from modules.utils import gradio

inputs = ('Chat input', 'interface_state')
reload_arr = ('history', 'name1', 'name2', 'mode', 'chat_style', 'character_menu')


def create_ui():
    mu = shared.args.multi_user

    shared.gradio['Chat input'] = gr.State()
    shared.gradio['history'] = gr.JSON(visible=False)
    shared.gradio['tools'] = gr.JSON(visible=False)

    with gr.Tab('Chat', id='Chat', elem_id='chat-tab'):
        with gr.Row(elem_id='past-chats-row', elem_classes=['pretty_scrollbar']):
            with gr.Column():
                with gr.Row(elem_id='past-chats-buttons'):
                    shared.gradio['branch_chat'] = gr.Button('Branch', elem_classes='refresh-button', interactive=not mu)
                    shared.gradio['rename_chat'] = gr.Button('Rename', elem_classes='refresh-button', interactive=not mu)
                    shared.gradio['delete_chat'] = gr.Button('🗑️', elem_classes='refresh-button', interactive=not mu)
                    shared.gradio['Start new chat'] = gr.Button('New chat', elem_classes=['refresh-button', 'focus-on-chat-input'])

                shared.gradio['search_chat'] = gr.Textbox(placeholder='Search chats...', max_lines=1, elem_id='search_chat')

                with gr.Row(elem_id='delete-chat-row', visible=False) as shared.gradio['delete-chat-row']:
                    shared.gradio['delete_chat-cancel'] = gr.Button('Cancel', elem_classes=['refresh-button', 'focus-on-chat-input'])
                    shared.gradio['delete_chat-confirm'] = gr.Button('Confirm', variant='stop', elem_classes=['refresh-button', 'focus-on-chat-input'])

                with gr.Row(elem_id='rename-row', visible=False) as shared.gradio['rename-row']:
                    shared.gradio['rename_to'] = gr.Textbox(label='Rename to:', placeholder='New name', elem_classes=['no-background'])
                    with gr.Row():
                        shared.gradio['rename_to-cancel'] = gr.Button('Cancel', elem_classes=['refresh-button', 'focus-on-chat-input'])
                        shared.gradio['rename_to-confirm'] = gr.Button('Confirm', elem_classes=['refresh-button', 'focus-on-chat-input'], variant='primary')

                with gr.Row():
                    shared.gradio['unique_id'] = gr.Radio(label="", elem_classes=['slim-dropdown', 'pretty_scrollbar'], interactive=not mu, elem_id='past-chats')

        with gr.Row():
            with gr.Column(elem_id='chat-col'):
                shared.gradio['html_display'] = gr.HTML(value=chat_html_wrapper({'internal': [], 'visible': []}, '', '', 'chat', 'cai-chat', ''), visible=True)
                shared.gradio['display'] = gr.Textbox(value="", visible=False)  # Hidden buffer
                with gr.Row(elem_id="chat-input-row"):
                    with gr.Column(scale=1, elem_id='gr-hover-container'):
                        gr.HTML(value='<div class="hover-element" onclick="void(0)"><span style="width: 100px; display: block" id="hover-element-button">&#9776;</span><div class="hover-menu" id="hover-menu"></div>', elem_id='gr-hover')

                    with gr.Column(scale=10, elem_id='chat-input-container'):
                        shared.gradio['textbox'] = gr.Textbox(label='', placeholder='Send a message', elem_id='chat-input', elem_classes=['add_scrollbar'])
                        shared.gradio['show_controls'] = gr.Checkbox(value=shared.settings['show_controls'], label='Show controls (Ctrl+S)', elem_id='show-controls')
                        shared.gradio['typing-dots'] = gr.HTML(value='<div class="typing"><span></span><span class="dot1"></span><span class="dot2"></span></div>', label='typing', elem_id='typing-container')

                    with gr.Column(scale=1, elem_id='generate-stop-container'):
                        with gr.Row():
                            shared.gradio['Stop'] = gr.Button('Stop', elem_id='stop', visible=False)
                            shared.gradio['Generate'] = gr.Button('Generate', elem_id='Generate', variant='primary')

        # Hover menu buttons
        with gr.Column(elem_id='chat-buttons'):
            with gr.Row():
                shared.gradio['Regenerate'] = gr.Button('Regenerate (Ctrl + Enter)', elem_id='Regenerate')
                shared.gradio['Continue'] = gr.Button('Continue (Alt + Enter)', elem_id='Continue')
                shared.gradio['Remove last'] = gr.Button('Remove last reply (Ctrl + Shift + Backspace)', elem_id='Remove-last')

            with gr.Row():
                shared.gradio['Replace last reply'] = gr.Button('Replace last reply (Ctrl + Shift + L)', elem_id='Replace-last')
                shared.gradio['Copy last reply'] = gr.Button('Copy last reply (Ctrl + Shift + K)', elem_id='Copy-last')
                shared.gradio['Impersonate'] = gr.Button('Impersonate (Ctrl + Shift + M)', elem_id='Impersonate')

            with gr.Row():
                shared.gradio['Send dummy message'] = gr.Button('Send dummy message')
                shared.gradio['Send dummy reply'] = gr.Button('Send dummy reply')

            with gr.Row():
                shared.gradio['send-chat-to-default'] = gr.Button('Send to default')
                shared.gradio['send-chat-to-notebook'] = gr.Button('Send to notebook')

        with gr.Row(elem_id='chat-controls', elem_classes=['pretty_scrollbar']):
            with gr.Column():
                with gr.Row():
                    shared.gradio['start_with'] = gr.Textbox(label='Start reply with', placeholder='Sure thing!', value=shared.settings['start_with'], elem_classes=['add_scrollbar'])

                with gr.Row():
                    shared.gradio['mode'] = gr.Radio(choices=['chat', 'chat-instruct', 'instruct'], value=shared.settings['mode'] if shared.settings['mode'] in ['chat', 'chat-instruct'] else None, label='Mode', info='Defines how the chat prompt is generated. In instruct and chat-instruct modes, the instruction template Parameters > Instruction template is used.', elem_id='chat-mode')

                with gr.Row():
                    shared.gradio['chat_style'] = gr.Dropdown(choices=utils.get_available_chat_styles(), label='Chat style', value=shared.settings['chat_style'], visible=shared.settings['mode'] != 'instruct')

                with gr.Row():
                    shared.gradio['chat-instruct_command'] = gr.Textbox(value=shared.settings['chat-instruct_command'], lines=12, label='Command for chat-instruct mode', info='<|character|> and <|prompt|> get replaced with the bot name and the regular chat prompt respectively.', visible=shared.settings['mode'] == 'chat-instruct', elem_classes=['add_scrollbar'])


def load_selected_tool_data():
    tool_name_list = utils.get_available_tools()
    if len(tool_name_list) > 0:
        tool_name, tool_type, tool_description, tool_parameters, tool_action = chat.load_tool(tool_name_list[0])
        shared.gradio['tool_type'].value = tool_type
        shared.gradio['tool_name'].value = tool_name
        shared.gradio['tool_description'].value = tool_description
        shared.gradio['tool_parameters'].value = json.dumps(tool_parameters, indent=4, sort_keys=True)
        shared.gradio['tool_action'].value = tool_action

def format_tool_parameters(tool_parameters):
    try:
        return json.dumps(json.loads(tool_parameters), indent=4, sort_keys=True)
    except json.decoder.JSONDecodeError:
        pass # Don't want to constantly notify while the user is editing
    return tool_parameters


def create_chat_settings_ui():
    mu = shared.args.multi_user
    with gr.Tab('Chat'):
        with gr.Row():
            with gr.Column(scale=8):
                with gr.Tab("Character"):
                    with gr.Row():
                        shared.gradio['character_menu'] = gr.Dropdown(value=None, choices=utils.get_available_characters(), label='Character', elem_id='character-menu', info='Used in chat and chat-instruct modes.', elem_classes='slim-dropdown')
                        ui.create_refresh_button(shared.gradio['character_menu'], lambda: None, lambda: {'choices': utils.get_available_characters()}, 'refresh-button', interactive=not mu)
                        shared.gradio['save_character'] = gr.Button('💾', elem_classes='refresh-button', elem_id="save-character", interactive=not mu)
                        shared.gradio['delete_character'] = gr.Button('🗑️', elem_classes='refresh-button', interactive=not mu)

                    shared.gradio['name2'] = gr.Textbox(value='', lines=1, label='Character\'s name')
                    shared.gradio['context'] = gr.Textbox(value='', lines=10, label='Context', elem_classes=['add_scrollbar'])
                    shared.gradio['greeting'] = gr.Textbox(value='', lines=5, label='Greeting', elem_classes=['add_scrollbar'])

                with gr.Tab("User"):
                    shared.gradio['name1'] = gr.Textbox(value=shared.settings['name1'], lines=1, label='Name')
                    shared.gradio['user_bio'] = gr.Textbox(value=shared.settings['user_bio'], lines=10, label='Description', info='Here you can optionally write a description of yourself.', placeholder='{{user}}\'s personality: ...', elem_classes=['add_scrollbar'])

                with gr.Tab('Chat history'):
                    with gr.Row():
                        with gr.Column():
                            shared.gradio['save_chat_history'] = gr.Button(value='Save history')

                        with gr.Column():
                            shared.gradio['load_chat_history'] = gr.File(type='binary', file_types=['.json', '.txt'], label='Upload History JSON')

                with gr.Tab('Upload character'):
                    with gr.Tab('YAML or JSON'):
                        with gr.Row():
                            shared.gradio['upload_json'] = gr.File(type='binary', file_types=['.json', '.yaml'], label='JSON or YAML File', interactive=not mu)
                            shared.gradio['upload_img_bot'] = gr.Image(type='pil', label='Profile Picture (optional)', interactive=not mu)

                        shared.gradio['Submit character'] = gr.Button(value='Submit', interactive=False)

                    with gr.Tab('TavernAI PNG'):
                        with gr.Row():
                            with gr.Column():
                                shared.gradio['upload_img_tavern'] = gr.Image(type='pil', label='TavernAI PNG File', elem_id='upload_img_tavern', interactive=not mu)
                                shared.gradio['tavern_json'] = gr.State()
                            with gr.Column():
                                shared.gradio['tavern_name'] = gr.Textbox(value='', lines=1, label='Name', interactive=False)
                                shared.gradio['tavern_desc'] = gr.Textbox(value='', lines=10, label='Description', interactive=False, elem_classes=['add_scrollbar'])

                        shared.gradio['Submit tavern character'] = gr.Button(value='Submit', interactive=False)

            with gr.Column(scale=1):
                shared.gradio['character_picture'] = gr.Image(label='Character picture', type='pil', interactive=not mu)
                shared.gradio['your_picture'] = gr.Image(label='Your picture', type='pil', value=Image.open(Path('cache/pfp_me.png')) if Path('cache/pfp_me.png').exists() else None, interactive=not mu)

    with gr.Tab('Instruction template'):
        with gr.Row():
            with gr.Column():
                with gr.Row():
                    shared.gradio['instruction_template'] = gr.Dropdown(choices=utils.get_available_instruction_templates(), label='Saved instruction templates', info="After selecting the template, click on \"Load\" to load and apply it.", value='None', elem_classes='slim-dropdown')
                    ui.create_refresh_button(shared.gradio['instruction_template'], lambda: None, lambda: {'choices': utils.get_available_instruction_templates()}, 'refresh-button', interactive=not mu)
                    shared.gradio['load_template'] = gr.Button("Load", elem_classes='refresh-button')
                    shared.gradio['save_template'] = gr.Button('💾', elem_classes='refresh-button', interactive=not mu)
                    shared.gradio['delete_template'] = gr.Button('🗑️ ', elem_classes='refresh-button', interactive=not mu)

            with gr.Column():
                pass

        with gr.Row():
            with gr.Column():
                shared.gradio['custom_system_message'] = gr.Textbox(value=shared.settings['custom_system_message'], lines=2, label='Custom system message', info='If not empty, will be used instead of the default one.', elem_classes=['add_scrollbar'])
                shared.gradio['instruction_template_str'] = gr.Textbox(value='', label='Instruction template', lines=24, info='This gets autodetected; you usually don\'t need to change it. Used in instruct and chat-instruct modes.', elem_classes=['add_scrollbar', 'monospace'])
                with gr.Row():
                    shared.gradio['send_instruction_to_default'] = gr.Button('Send to default', elem_classes=['small-button'])
                    shared.gradio['send_instruction_to_notebook'] = gr.Button('Send to notebook', elem_classes=['small-button'])
                    shared.gradio['send_instruction_to_negative_prompt'] = gr.Button('Send to negative prompt', elem_classes=['small-button'])

            with gr.Column():
                shared.gradio['chat_template_str'] = gr.Textbox(value=shared.settings['chat_template_str'], label='Chat template', lines=22, elem_classes=['add_scrollbar', 'monospace'])

    with gr.Tab('Tools'):
        with gr.Row():
            # Tool selection and creation options
            with gr.Column():
                # I added this as a first precaution for executing potentially malicious LLM-generated code, but it's ultimately up to the user to choose whether to run it.
                shared.gradio['confirm_tool_use'] = gr.Checkbox(value=True, label='Confirm tool use', elem_id='confirm-tool-use', info='When a tool is about to be called, pause execution until confirmed by user by pressing "continue"')
                shared.gradio['tools_in_user_message'] = gr.Checkbox(value=False, label='Tools in user message', elem_id='tools-in-user-message', info='Whether to include the tools in the user message as opposed to the system prompt')
                shared.gradio['max_consecutive_tool_uses'] = gr.Number(value=3, label='Max consecutive tool uses', elem_id='max-consecutive-tool-uses', info='Only applies when tool use is automatic (if "confirm tool use" is disabled)')
                # Tool Selection Presets
                with gr.Row():
                    shared.gradio['tool_preset_menu'] = gr.Dropdown(value='None', choices=utils.get_available_tool_presets(), label='Tool Selection Preset', elem_id='tool-preset-menu', info='Save or load a selection of tools', elem_classes='slim-dropdown')
                    ui.create_refresh_button(shared.gradio['tool_preset_menu'], lambda: None, lambda: {'choices': utils.get_available_tool_presets()}, 'refresh-button', interactive=not mu)
                    shared.gradio['save_tool_preset'] = gr.Button('💾', elem_classes='refresh-button', elem_id="save-tool-preset", interactive=not mu)
                    shared.gradio['delete_tool_preset'] = gr.Button('🗑️', elem_classes='refresh-button', interactive=not mu)
                # Tool menu (list of available tools, should look similar to past chats interface)
                with gr.Row():
                    shared.gradio['tool_menu'] = gr.CheckboxGroup(utils.get_available_tools(), elem_classes=['slim-dropdown', 'pretty_scrollbar'], elem_id='tool-menu', multiselect=True, info='Which tools the model can use. Note that only certain models are able to use tools, e.g. Llama 3. Currently only works in instruct mode.')

            # Tool Details / Add New Tool
            with gr.Column():
                with gr.Row():
                    shared.gradio['tool_name'] = gr.Textbox(value='', lines=1, label='Tool Name', interactive=not mu)
                    shared.gradio['save_tool'] = gr.Button('💾', elem_classes='refresh-button', elem_id="save-tool", interactive=not mu)
                    shared.gradio['delete_tool'] = gr.Button('🗑️', elem_classes='refresh-button', interactive=not mu)
                shared.gradio['tool_type'] = gr.Dropdown(value='function', choices=['function'], interactive=not mu, label='Tool Type', elem_id='tool-type-menu', info='Only function type available for now', elem_classes='slim-dropdown')
                shared.gradio['tool_description'] = gr.Textbox(value='', lines=3, interactive=not mu, label='Tool Description', info='This helps the model determine when to use the tool.')
                gr.HTML(value='<span class="tool-menu-label">Tool Parameters</span><div class="tool-menu-info">Specify the parameters for the tool in JSON. This helps the model determine what information to pass to the tool.</div>')
                shared.gradio['tool_parameters'] = gr.Code(label='Tool Parameters', language='json', lines=10, interactive=not mu, elem_id='tool-parameters-menu', info='Specify the parameters for the tool in JSON. This helps the model determine what information to pass to the tool.')
                # This is better than nothing, but using hljs would be better for syntax highlighting than what comes with Gradio
                # Also the "info" field doesn't work with Code elements, so we need a separate HTML element for it
                gr.HTML(value='<span class="tool-menu-label">Tool Action</span><div class="tool-menu-info">Specify the action of the tool as a Python code snippet. Define a function with a name matching the tool name above and either no parameters or one parameter called parameters. Important: When this tool is enabled (or the tool is saved while already enabled), this code will be evaluated in order to define the function. The code here will also be executed directly, so I highly recommend using a sandboxed environment if you are using a tool like code interpreter. Be very careful about the code you are executing!</div>')
                shared.gradio['tool_action'] = gr.Code(label='Tool Action', language='python', lines=20, interactive=not mu, elem_id='tool-action-menu', info='Specify the action of the tool as a Python code snippet. Define a function with a name matching the tool name above and either no parameters or one parameter called parameters. Important: When this tool is enabled (or the tool is saved while already enabled), this code will be evaluated in order to define the function. The code here will also be executed directly, so I highly recommend using a sandboxed environment if you are using a tool like code interpreter. Be very careful about the code you are executing!')
                # Ideally code interpreter would run in a separate container, but not sure how practical that is to set up here.
                # Currently, we're displaying a warning for use of eval/exec, but really this is up to the user to be careful with.
                # Other possible options: load code from a file, or call an API endpoint

                # Fill the previous fields, loading the first available tool if possible
                load_selected_tool_data()

                # Example tool fields based on OpenAI spec https://platform.openai.com/docs/guides/function-calling
                # The parameter format is not required to be exactly like this, but the validation I've added currently assumes it will be
                # Tool type (dropdown: function)
                # Tool name
                # Tool description
                # Tool parameters (list)
                    # For each parameter:
                    # Type (object, string, number, etc)
                    # Required properties
                    # Properties (if object?)
                        # For each property:
                        # Type
                        # Description
                        # Enum (optional)

def create_event_handlers():

    # Obsolete variables, kept for compatibility with old extensions
    shared.input_params = gradio(inputs)
    shared.reload_inputs = gradio(reload_arr)

    # Morph HTML updates instead of updating everything
    shared.gradio['display'].change(None, gradio('display'), None, js="(text) => handleMorphdomUpdate(text)")

    shared.gradio['Generate'].click(
        ui.gather_interface_values, gradio(shared.input_elements), gradio('interface_state')).then(
        lambda x: (x, ''), gradio('textbox'), gradio('Chat input', 'textbox'), show_progress=False).then(
        lambda: None, None, None, js='() => document.getElementById("chat").parentNode.parentNode.parentNode.classList.add("_generating")').then(
        chat.generate_chat_reply_wrapper, gradio(inputs), gradio('display', 'history'), show_progress=False).then(
        None, None, None, js='() => document.getElementById("chat").parentNode.parentNode.parentNode.classList.remove("_generating")').then(
        None, None, None, js=f'() => {{{ui.audio_notification_js}}}')

    shared.gradio['textbox'].submit(
        ui.gather_interface_values, gradio(shared.input_elements), gradio('interface_state')).then(
        lambda x: (x, ''), gradio('textbox'), gradio('Chat input', 'textbox'), show_progress=False).then(
        lambda: None, None, None, js='() => document.getElementById("chat").parentNode.parentNode.parentNode.classList.add("_generating")').then(
        chat.generate_chat_reply_wrapper, gradio(inputs), gradio('display', 'history'), show_progress=False).then(
        None, None, None, js='() => document.getElementById("chat").parentNode.parentNode.parentNode.classList.remove("_generating")').then(
        None, None, None, js=f'() => {{{ui.audio_notification_js}}}')

    shared.gradio['Regenerate'].click(
        ui.gather_interface_values, gradio(shared.input_elements), gradio('interface_state')).then(
        lambda: None, None, None, js='() => document.getElementById("chat").parentNode.parentNode.parentNode.classList.add("_generating")').then(
        partial(chat.generate_chat_reply_wrapper, regenerate=True), gradio(inputs), gradio('display', 'history'), show_progress=False).then(
        None, None, None, js='() => document.getElementById("chat").parentNode.parentNode.parentNode.classList.remove("_generating")').then(
        None, None, None, js=f'() => {{{ui.audio_notification_js}}}')

    shared.gradio['Continue'].click(
        ui.gather_interface_values, gradio(shared.input_elements), gradio('interface_state')).then(
        lambda: None, None, None, js='() => document.getElementById("chat").parentNode.parentNode.parentNode.classList.add("_generating")').then(
        partial(chat.generate_chat_reply_wrapper, _continue=True), gradio(inputs), gradio('display', 'history'), show_progress=False).then(
        None, None, None, js='() => document.getElementById("chat").parentNode.parentNode.parentNode.classList.remove("_generating")').then(
        None, None, None, js=f'() => {{{ui.audio_notification_js}}}')

    shared.gradio['Impersonate'].click(
        ui.gather_interface_values, gradio(shared.input_elements), gradio('interface_state')).then(
        lambda x: x, gradio('textbox'), gradio('Chat input'), show_progress=False).then(
        lambda: None, None, None, js='() => document.getElementById("chat").parentNode.parentNode.parentNode.classList.add("_generating")').then(
        chat.impersonate_wrapper, gradio(inputs), gradio('textbox', 'display'), show_progress=False).then(
        None, None, None, js='() => document.getElementById("chat").parentNode.parentNode.parentNode.classList.remove("_generating")').then(
        None, None, None, js=f'() => {{{ui.audio_notification_js}}}')

    shared.gradio['Replace last reply'].click(
        ui.gather_interface_values, gradio(shared.input_elements), gradio('interface_state')).then(
        chat.handle_replace_last_reply_click, gradio('textbox', 'interface_state'), gradio('history', 'display', 'textbox'), show_progress=False)

    shared.gradio['Send dummy message'].click(
        ui.gather_interface_values, gradio(shared.input_elements), gradio('interface_state')).then(
        chat.handle_send_dummy_message_click, gradio('textbox', 'interface_state'), gradio('history', 'display', 'textbox'), show_progress=False)

    shared.gradio['Send dummy reply'].click(
        ui.gather_interface_values, gradio(shared.input_elements), gradio('interface_state')).then(
        chat.handle_send_dummy_reply_click, gradio('textbox', 'interface_state'), gradio('history', 'display', 'textbox'), show_progress=False)

    shared.gradio['Remove last'].click(
        ui.gather_interface_values, gradio(shared.input_elements), gradio('interface_state')).then(
        chat.handle_remove_last_click, gradio('interface_state'), gradio('history', 'display', 'textbox'), show_progress=False)

    shared.gradio['Stop'].click(
        stop_everything_event, None, None, queue=False).then(
        chat.redraw_html, gradio(reload_arr), gradio('display'), show_progress=False)

    if not shared.args.multi_user:
        shared.gradio['unique_id'].select(
            ui.gather_interface_values, gradio(shared.input_elements), gradio('interface_state')).then(
            chat.handle_unique_id_select, gradio('interface_state'), gradio('history', 'display'), show_progress=False)

    shared.gradio['Start new chat'].click(
        ui.gather_interface_values, gradio(shared.input_elements), gradio('interface_state')).then(
        chat.handle_start_new_chat_click, gradio('interface_state'), gradio('history', 'display', 'unique_id'), show_progress=False)

    shared.gradio['delete_chat'].click(lambda: gr.update(visible=True), None, gradio('delete-chat-row'))
    shared.gradio['delete_chat-cancel'].click(lambda: gr.update(visible=False), None, gradio('delete-chat-row'))
    shared.gradio['delete_chat-confirm'].click(
        ui.gather_interface_values, gradio(shared.input_elements), gradio('interface_state')).then(
        chat.handle_delete_chat_confirm_click, gradio('interface_state'), gradio('history', 'display', 'unique_id', 'delete-chat-row'), show_progress=False)

    shared.gradio['branch_chat'].click(
        ui.gather_interface_values, gradio(shared.input_elements), gradio('interface_state')).then(
        chat.handle_branch_chat_click, gradio('interface_state'), gradio('history', 'display', 'unique_id'), show_progress=False)

    shared.gradio['rename_chat'].click(chat.handle_rename_chat_click, None, gradio('rename_to', 'rename-row'), show_progress=False)
    shared.gradio['rename_to-cancel'].click(lambda: gr.update(visible=False), None, gradio('rename-row'), show_progress=False)
    shared.gradio['rename_to-confirm'].click(
        ui.gather_interface_values, gradio(shared.input_elements), gradio('interface_state')).then(
        chat.handle_rename_chat_confirm, gradio('rename_to', 'interface_state'), gradio('unique_id', 'rename-row'))

    shared.gradio['rename_to'].submit(
        ui.gather_interface_values, gradio(shared.input_elements), gradio('interface_state')).then(
        chat.handle_rename_chat_confirm, gradio('rename_to', 'interface_state'), gradio('unique_id', 'rename-row'), show_progress=False)

    shared.gradio['search_chat'].change(
        ui.gather_interface_values, gradio(shared.input_elements), gradio('interface_state')).then(
        chat.handle_search_chat_change, gradio('interface_state'), gradio('unique_id'), show_progress=False)

    shared.gradio['load_chat_history'].upload(
        ui.gather_interface_values, gradio(shared.input_elements), gradio('interface_state')).then(
        chat.handle_upload_chat_history, gradio('load_chat_history', 'interface_state'), gradio('history', 'display', 'unique_id'), show_progress=False).then(
        None, None, None, js=f'() => {{{ui.switch_tabs_js}; switch_to_chat()}}')

    shared.gradio['character_menu'].change(
        ui.gather_interface_values, gradio(shared.input_elements), gradio('interface_state')).then(
        chat.handle_character_menu_change, gradio('interface_state'), gradio('history', 'display', 'name1', 'name2', 'character_picture', 'greeting', 'context', 'unique_id'), show_progress=False).then(
        None, None, None, js=f'() => {{{ui.update_big_picture_js}; updateBigPicture()}}')

    shared.gradio['mode'].change(
        ui.gather_interface_values, gradio(shared.input_elements), gradio('interface_state')).then(
        chat.handle_mode_change, gradio('interface_state'), gradio('history', 'display', 'chat_style', 'chat-instruct_command', 'unique_id'), show_progress=False).then(
        None, gradio('mode'), None, js="(mode) => {mode === 'instruct' ? document.getElementById('character-menu').parentNode.parentNode.style.display = 'none' : document.getElementById('character-menu').parentNode.parentNode.style.display = ''}")

    shared.gradio['chat_style'].change(chat.redraw_html, gradio(reload_arr), gradio('display'), show_progress=False)
    shared.gradio['Copy last reply'].click(chat.send_last_reply_to_input, gradio('history'), gradio('textbox'), show_progress=False)

    # Save/delete a character
    shared.gradio['save_character'].click(chat.handle_save_character_click, gradio('name2'), gradio('save_character_filename', 'character_saver'), show_progress=False)
    shared.gradio['delete_character'].click(lambda: gr.update(visible=True), None, gradio('character_deleter'), show_progress=False)
    shared.gradio['load_template'].click(chat.handle_load_template_click, gradio('instruction_template'), gradio('instruction_template_str', 'instruction_template'), show_progress=False)
    shared.gradio['save_template'].click(
        ui.gather_interface_values, gradio(shared.input_elements), gradio('interface_state')).then(
        chat.handle_save_template_click, gradio('instruction_template_str'), gradio('save_filename', 'save_root', 'save_contents', 'file_saver'), show_progress=False)

    shared.gradio['delete_template'].click(chat.handle_delete_template_click, gradio('instruction_template'), gradio('delete_filename', 'delete_root', 'file_deleter'), show_progress=False)
    shared.gradio['save_chat_history'].click(
        lambda x: json.dumps(x, indent=4), gradio('history'), gradio('temporary_text')).then(
        None, gradio('temporary_text', 'character_menu', 'mode'), None, js=f'(hist, char, mode) => {{{ui.save_files_js}; saveHistory(hist, char, mode)}}')

    shared.gradio['Submit character'].click(
        chat.upload_character, gradio('upload_json', 'upload_img_bot'), gradio('character_menu'), show_progress=False).then(
        None, None, None, js=f'() => {{{ui.switch_tabs_js}; switch_to_character()}}')

    shared.gradio['Submit tavern character'].click(
        chat.upload_tavern_character, gradio('upload_img_tavern', 'tavern_json'), gradio('character_menu'), show_progress=False).then(
        None, None, None, js=f'() => {{{ui.switch_tabs_js}; switch_to_character()}}')

    shared.gradio['upload_json'].upload(lambda: gr.update(interactive=True), None, gradio('Submit character'))
    shared.gradio['upload_json'].clear(lambda: gr.update(interactive=False), None, gradio('Submit character'))
    shared.gradio['upload_img_tavern'].upload(chat.check_tavern_character, gradio('upload_img_tavern'), gradio('tavern_name', 'tavern_desc', 'tavern_json', 'Submit tavern character'), show_progress=False)
    shared.gradio['upload_img_tavern'].clear(lambda: (None, None, None, gr.update(interactive=False)), None, gradio('tavern_name', 'tavern_desc', 'tavern_json', 'Submit tavern character'), show_progress=False)
    shared.gradio['your_picture'].change(
        ui.gather_interface_values, gradio(shared.input_elements), gradio('interface_state')).then(
        chat.handle_your_picture_change, gradio('your_picture', 'interface_state'), gradio('display'), show_progress=False)

    shared.gradio['send_instruction_to_default'].click(
        ui.gather_interface_values, gradio(shared.input_elements), gradio('interface_state')).then(
        chat.handle_send_instruction_click, gradio('interface_state'), gradio('textbox-default'), show_progress=False).then(
        None, None, None, js=f'() => {{{ui.switch_tabs_js}; switch_to_default()}}')

    shared.gradio['send_instruction_to_notebook'].click(
        ui.gather_interface_values, gradio(shared.input_elements), gradio('interface_state')).then(
        chat.handle_send_instruction_click, gradio('interface_state'), gradio('textbox-notebook'), show_progress=False).then(
        None, None, None, js=f'() => {{{ui.switch_tabs_js}; switch_to_notebook()}}')

    shared.gradio['send_instruction_to_negative_prompt'].click(
        ui.gather_interface_values, gradio(shared.input_elements), gradio('interface_state')).then(
        chat.handle_send_instruction_click, gradio('interface_state'), gradio('negative_prompt'), show_progress=False).then(
        None, None, None, js=f'() => {{{ui.switch_tabs_js}; switch_to_generation_parameters()}}')

    shared.gradio['send-chat-to-default'].click(
        ui.gather_interface_values, gradio(shared.input_elements), gradio('interface_state')).then(
        chat.handle_send_chat_click, gradio('interface_state'), gradio('textbox-default'), show_progress=False).then(
        None, None, None, js=f'() => {{{ui.switch_tabs_js}; switch_to_default()}}')

    shared.gradio['send-chat-to-notebook'].click(
        ui.gather_interface_values, gradio(shared.input_elements), gradio('interface_state')).then(
        chat.handle_send_chat_click, gradio('interface_state'), gradio('textbox-notebook'), show_progress=False).then(
        None, None, None, js=f'() => {{{ui.switch_tabs_js}; switch_to_notebook()}}')

    shared.gradio['show_controls'].change(None, gradio('show_controls'), None, js=f'(x) => {{{ui.show_controls_js}; toggle_controls(x)}}')

    shared.gradio['confirm_tool_use'].change(lambda x: gr.Warning("With this setting turned off, tool use will happen automatically, be careful, especially if you are running a code interpreter tool!") if not x else None, gradio('confirm_tool_use'), None, show_progress=False)

    shared.gradio['tool_preset_menu'].change(chat.handle_tool_preset_change, gradio('tool_preset_menu'), gradio('tool_menu'), show_progress=False)
    shared.gradio['save_tool_preset'].click(chat.handle_save_tool_preset_click, gradio('tool_preset_menu'), gradio('save_tool_preset_filename', 'tool_preset_saver'), show_progress=False)
    shared.gradio['delete_tool_preset'].click(lambda: gr.update(visible=True), None, gradio('tool_preset_deleter'), show_progress=False)

    shared.gradio['tool_menu'].change(chat.handle_tool_change, gradio('tool_menu', 'tools'), gradio('tool_name', 'tool_type', 'tool_description', 'tool_parameters', 'tool_action', 'tools'), show_progress=False)

    shared.gradio['save_tool'].click(chat.handle_save_tool_click, gradio('tool_name'), gradio('save_tool_filename', 'tool_saver'), show_progress=False)
    shared.gradio['delete_tool'].click(lambda: gr.update(visible=True), None, gradio('tool_deleter'), show_progress=False)

    # Re-format JSON automatically (I thought this was useful, but let me know what you think)
    shared.gradio['tool_parameters'].blur(format_tool_parameters, gradio('tool_parameters'), gradio('tool_parameters'), show_progress=False)
