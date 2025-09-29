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
    shared.gradio['history'] = gr.State({'internal': [], 'visible': [], 'metadata': {}})
    shared.gradio['display'] = gr.JSON(value={}, visible=False)  # Hidden buffer

    with gr.Tab('Чат', elem_id='chat-tab'):
        with gr.Row(elem_id='past-chats-row', elem_classes=['pretty_scrollbar']):
            with gr.Column():
                with gr.Row(elem_id='past-chats-buttons'):
                    shared.gradio['branch_chat'] = gr.Button('Ветка', elem_classes=['refresh-button', 'refresh-button-medium'], elem_id='Branch', interactive=not mu)
                    shared.gradio['rename_chat'] = gr.Button('Переименовать', elem_classes=['refresh-button', 'refresh-button-medium'], interactive=not mu)
                    shared.gradio['delete_chat'] = gr.Button('🗑️', visible=False, elem_classes='refresh-button', interactive=not mu, elem_id='delete_chat')
                    shared.gradio['Start new chat'] = gr.Button('Новый чат', elem_classes=['refresh-button', 'refresh-button-medium', 'focus-on-chat-input'])
                    shared.gradio['branch_index'] = gr.Number(value=-1, precision=0, visible=False, elem_id="Branch-index", interactive=True)

                shared.gradio['search_chat'] = gr.Textbox(placeholder='Поиск чатов...', max_lines=1, elem_id='search_chat')

                with gr.Row(elem_id='delete-chat-row', visible=False) as shared.gradio['delete-chat-row']:
                    shared.gradio['delete_chat-cancel'] = gr.Button('Отмена', elem_classes=['refresh-button', 'focus-on-chat-input'], elem_id='delete_chat-cancel')
                    shared.gradio['delete_chat-confirm'] = gr.Button('Подтвердить', variant='stop', elem_classes=['refresh-button', 'focus-on-chat-input'], elem_id='delete_chat-confirm')

                with gr.Row(elem_id='rename-row', visible=False) as shared.gradio['rename-row']:
                    shared.gradio['rename_to'] = gr.Textbox(label='Переименовать в:', placeholder='Новое название', elem_classes=['no-background'])
                    with gr.Row():
                        shared.gradio['rename_to-cancel'] = gr.Button('Отмена', elem_classes=['refresh-button', 'focus-on-chat-input'])
                        shared.gradio['rename_to-confirm'] = gr.Button('Подтвердить', elem_classes=['refresh-button', 'focus-on-chat-input'], variant='primary')

                with gr.Row():
                    shared.gradio['unique_id'] = gr.Radio(label="", elem_classes=['slim-dropdown', 'pretty_scrollbar'], interactive=not mu, elem_id='past-chats')

        with gr.Row():
            with gr.Column(elem_id='chat-col'):
                shared.gradio['html_display'] = gr.HTML(value=chat_html_wrapper({'internal': [], 'visible': [], 'metadata': {}}, '', '', 'chat', 'cai-chat', '')['html'], visible=True)
                with gr.Row(elem_id="chat-input-row"):
                    with gr.Column(scale=1, elem_id='gr-hover-container'):
                        gr.HTML(value='<div class="hover-element" onclick="void(0)"><span style="width: 100px; display: block" id="hover-element-button">&#9776;</span><div class="hover-menu" id="hover-menu"></div>', elem_id='gr-hover')

                    with gr.Column(scale=10, elem_id='chat-input-container'):
                        shared.gradio['textbox'] = gr.MultimodalTextbox(label='', placeholder='Напишите сообщение', file_types=['text', '.pdf', 'image'], file_count="multiple", elem_id='chat-input', elem_classes=['add_scrollbar'])
                        shared.gradio['typing-dots'] = gr.HTML(value='<div class="typing"><span></span><span class="dot1"></span><span class="dot2"></span></div>', label='typing', elem_id='typing-container')

                    with gr.Column(scale=1, elem_id='generate-stop-container'):
                        with gr.Row():
                            shared.gradio['Stop'] = gr.Button('Стоп', elem_id='stop', visible=False)
                            shared.gradio['Generate'] = gr.Button('Отправить', elem_id='Generate', variant='primary')

        # Hover menu buttons
        with gr.Column(elem_id='chat-buttons'):
            shared.gradio['Regenerate'] = gr.Button('Перегенерировать (Ctrl + Enter)', elem_id='Regenerate')
            shared.gradio['Continue'] = gr.Button('Продолжить (Alt + Enter)', elem_id='Continue')
            shared.gradio['Remove last'] = gr.Button('Удалить последний ответ (Ctrl + Shift + Backspace)', elem_id='Remove-last')
            shared.gradio['Impersonate'] = gr.Button('От лица персонажа (Ctrl + Shift + M)', elem_id='Impersonate')
            shared.gradio['Send dummy message'] = gr.Button('Отправить пустое сообщение')
            shared.gradio['Send dummy reply'] = gr.Button('Отправить пустой ответ')
            shared.gradio['send-chat-to-notebook'] = gr.Button('Отправить в блокнот')
            shared.gradio['show_controls'] = gr.Checkbox(value=shared.settings['show_controls'], label='Показать панель управления (Ctrl+S)', elem_id='show-controls')

        with gr.Row(elem_id='chat-controls', elem_classes=['pretty_scrollbar']):
            with gr.Column():
                with gr.Row():
                    shared.gradio['start_with'] = gr.Textbox(label='Начать ответ с', placeholder='Конечно!', value=shared.settings['start_with'], elem_classes=['add_scrollbar'])

                gr.HTML("<div class='sidebar-vertical-separator'></div>")

                shared.gradio['reasoning_effort'] = gr.Dropdown(value=shared.settings['reasoning_effort'], choices=['low', 'medium', 'high'], label='Уровень рассуждений', info='Используется GPT-OSS.')
                shared.gradio['enable_thinking'] = gr.Checkbox(value=shared.settings['enable_thinking'], label='Включить размышления', info='Используется Seed-OSS и pre-2507 Qwen3.')

                gr.HTML("<div class='sidebar-vertical-separator'></div>")

                shared.gradio['enable_web_search'] = gr.Checkbox(value=shared.settings.get('enable_web_search', False), label='Активировать веб-поиск', elem_id='web-search')
                with gr.Row(visible=shared.settings.get('enable_web_search', False)) as shared.gradio['web_search_row']:
                    shared.gradio['web_search_pages'] = gr.Number(value=shared.settings.get('web_search_pages', 3), precision=0, label='Количество страниц для загрузки', minimum=1, maximum=10)

                gr.HTML("<div class='sidebar-vertical-separator'></div>")

                with gr.Row():
                    shared.gradio['mode'] = gr.Radio(choices=['instruct', 'chat-instruct', 'chat'], value=None, label='Режим', info='В режимах instruct и chat-instruct используется шаблон из Параметры > Шаблон инструкций.', elem_id='chat-mode')

                with gr.Row():
                    shared.gradio['chat_style'] = gr.Dropdown(choices=utils.get_available_chat_styles(), label='Стиль чата', value=shared.settings['chat_style'], visible=shared.settings['mode'] != 'instruct')

                with gr.Row():
                    shared.gradio['chat-instruct_command'] = gr.Textbox(value=shared.settings['chat-instruct_command'], lines=12, label='Команда для режима chat-instruct', info='<|character|> и <|prompt|> заменяются на имя бота и обычный промпт чата соответственно.', visible=shared.settings['mode'] == 'chat-instruct', elem_classes=['add_scrollbar'])

                gr.HTML("<div class='sidebar-vertical-separator'></div>")

                with gr.Row():
                    shared.gradio['count_tokens'] = gr.Button('Подсчитать токены', size='sm')

                shared.gradio['token_display'] = gr.HTML(value='', elem_classes='token-display')

        # Hidden elements for version navigation and editing
        with gr.Row(visible=False):
            shared.gradio['navigate_message_index'] = gr.Number(value=-1, precision=0, elem_id="Navigate-message-index")
            shared.gradio['navigate_direction'] = gr.Textbox(value="", elem_id="Navigate-direction")
            shared.gradio['navigate_message_role'] = gr.Textbox(value="", elem_id="Navigate-message-role")
            shared.gradio['navigate_version'] = gr.Button(elem_id="Navigate-version")
            shared.gradio['edit_message_index'] = gr.Number(value=-1, precision=0, elem_id="Edit-message-index")
            shared.gradio['edit_message_text'] = gr.Textbox(value="", elem_id="Edit-message-text")
            shared.gradio['edit_message_role'] = gr.Textbox(value="", elem_id="Edit-message-role")
            shared.gradio['edit_message'] = gr.Button(elem_id="Edit-message")


def create_character_settings_ui():
    mu = shared.args.multi_user
    with gr.Tab('Персонаж', elem_id="character-tab"):
        with gr.Row():
            with gr.Column(scale=8):
                with gr.Tab("Персонаж"):
                    with gr.Row():
                        shared.gradio['character_menu'] = gr.Dropdown(value=shared.settings['character'], choices=utils.get_available_characters(), label='Персонаж', elem_id='character-menu', info='Используется в режимах чат и чат-инструкция.', elem_classes='slim-dropdown')
                        ui.create_refresh_button(shared.gradio['character_menu'], lambda: None, lambda: {'choices': utils.get_available_characters()}, 'refresh-button', interactive=not mu)
                        shared.gradio['save_character'] = gr.Button('💾', elem_classes='refresh-button', elem_id="save-character", interactive=not mu)
                        shared.gradio['delete_character'] = gr.Button('🗑️', elem_classes='refresh-button', interactive=not mu)
                        shared.gradio['restore_character'] = gr.Button('Восстановить персонажа', elem_classes='refresh-button', interactive=True, elem_id='restore-character')

                    shared.gradio['name2'] = gr.Textbox(value=shared.settings['name2'], lines=1, label='Имя персонажа')
                    shared.gradio['context'] = gr.Textbox(value=shared.settings['context'], lines=10, label='Контекст', elem_classes=['add_scrollbar'], elem_id="character-context")
                    shared.gradio['greeting'] = gr.Textbox(value=shared.settings['greeting'], lines=5, label='Приветствие', elem_classes=['add_scrollbar'], elem_id="character-greeting")

                with gr.Tab("Пользователь"):
                    shared.gradio['name1'] = gr.Textbox(value=shared.settings['name1'], lines=1, label='Имя')
                    shared.gradio['user_bio'] = gr.Textbox(value=shared.settings['user_bio'], lines=10, label='Описание', info='Здесь вы можете по желанию написать описание себя.', placeholder='Личность {{user}}: ...', elem_classes=['add_scrollbar'], elem_id="user-description")

                with gr.Tab('История чата'):
                    with gr.Row():
                        with gr.Column():
                            shared.gradio['save_chat_history'] = gr.Button(value='Сохранить историю')

                        with gr.Column():
                            shared.gradio['load_chat_history'] = gr.File(type='binary', file_types=['.json', '.txt'], label='Загрузить историю JSON')

                with gr.Tab('Загрузка персонажа'):
                    with gr.Tab('YAML или JSON'):
                        with gr.Row():
                            shared.gradio['upload_json'] = gr.File(type='binary', file_types=['.json', '.yaml'], label='Файл JSON или YAML', interactive=not mu)
                            shared.gradio['upload_img_bot'] = gr.Image(type='pil', label='Аватар (опционально)', interactive=not mu)

                        shared.gradio['Submit character'] = gr.Button(value='Отправить', interactive=False)

                    with gr.Tab('TavernAI PNG'):
                        with gr.Row():
                            with gr.Column():
                                shared.gradio['upload_img_tavern'] = gr.Image(type='pil', label='Файл TavernAI PNG', elem_id='upload_img_tavern', interactive=not mu)
                                shared.gradio['tavern_json'] = gr.State()
                            with gr.Column():
                                shared.gradio['tavern_name'] = gr.Textbox(value='', lines=1, label='Имя', interactive=False)
                                shared.gradio['tavern_desc'] = gr.Textbox(value='', lines=10, label='Описание', interactive=False, elem_classes=['add_scrollbar'])

                        shared.gradio['Submit tavern character'] = gr.Button(value='Отправить', interactive=False)

            with gr.Column(scale=1):
                shared.gradio['character_picture'] = gr.Image(label='Аватар персонажа', type='pil', interactive=not mu)
                shared.gradio['your_picture'] = gr.Image(label='Ваш аватар', type='pil', value=Image.open(Path('user_data/cache/pfp_me.png')) if Path('user_data/cache/pfp_me.png').exists() else None, interactive=not mu)


def create_chat_settings_ui():
    mu = shared.args.multi_user
    with gr.Tab('Шаблон инструкций'):
        with gr.Row():
            with gr.Column():
                with gr.Row():
                    shared.gradio['instruction_template'] = gr.Dropdown(choices=utils.get_available_instruction_templates(), label='Сохраненные шаблоны инструкций', info="После выбора шаблона нажмите 'Загрузить', чтобы загрузить и применить его.", value='None', elem_classes='slim-dropdown')
                    ui.create_refresh_button(shared.gradio['instruction_template'], lambda: None, lambda: {'choices': utils.get_available_instruction_templates()}, 'refresh-button', interactive=not mu)
                    shared.gradio['load_template'] = gr.Button("Загрузить", elem_classes='refresh-button')
                    shared.gradio['save_template'] = gr.Button('💾', elem_classes='refresh-button', interactive=not mu)
                    shared.gradio['delete_template'] = gr.Button('🗑️ ', elem_classes='refresh-button', interactive=not mu)

            with gr.Column():
                pass

        with gr.Row():
            with gr.Column():
                shared.gradio['instruction_template_str'] = gr.Textbox(value=shared.settings['instruction_template_str'], label='Шаблон инструкций', lines=24, info='Определяется автоматически; обычно вам не нужно его изменять. Используется в режимах instruct и chat-instruct.', elem_classes=['add_scrollbar', 'monospace'], elem_id='instruction-template-str')
                with gr.Row():
                    shared.gradio['send_instruction_to_notebook'] = gr.Button('Отправить в блокнот', elem_classes=['small-button'])

            with gr.Column():
                shared.gradio['chat_template_str'] = gr.Textbox(value=shared.settings['chat_template_str'], label='Шаблон чата', lines=22, elem_classes=['add_scrollbar', 'monospace'], info='Определяет, как генерируется промпт чата в режимах chat/chat-instruct.', elem_id='chat-template-str')


def create_event_handlers():

    # Obsolete variables, kept for compatibility with old extensions
    shared.input_params = gradio(inputs)
    shared.reload_inputs = gradio(reload_arr)

    # Morph HTML updates instead of updating everything
    shared.gradio['display'].change(None, gradio('display'), None, js="(data) => handleMorphdomUpdate(data)")

    shared.gradio['Generate'].click(
        ui.gather_interface_values, gradio(shared.input_elements), gradio('interface_state')).then(
        lambda x: (x, {"text": "", "files": []}), gradio('textbox'), gradio('Chat input', 'textbox'), show_progress=False).then(
        lambda: None, None, None, js='() => document.getElementById("chat").parentNode.parentNode.parentNode.classList.add("_generating")').then(
        chat.generate_chat_reply_wrapper, gradio(inputs), gradio('display', 'history'), show_progress=False).then(
        None, None, None, js='() => document.getElementById("chat").parentNode.parentNode.parentNode.classList.remove("_generating")').then(
        None, None, None, js=f'() => {{{ui.audio_notification_js}}}')

    shared.gradio['textbox'].submit(
        ui.gather_interface_values, gradio(shared.input_elements), gradio('interface_state')).then(
        lambda x: (x, {"text": "", "files": []}), gradio('textbox'), gradio('Chat input', 'textbox'), show_progress=False).then(
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

    shared.gradio['delete_chat-confirm'].click(
        ui.gather_interface_values, gradio(shared.input_elements), gradio('interface_state')).then(
        chat.handle_delete_chat_confirm_click, gradio('interface_state'), gradio('history', 'display', 'unique_id'), show_progress=False)

    shared.gradio['branch_chat'].click(
        ui.gather_interface_values, gradio(shared.input_elements), gradio('interface_state')).then(
        chat.handle_branch_chat_click, gradio('interface_state'), gradio('history', 'display', 'unique_id', 'branch_index'), show_progress=False)

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

    shared.gradio['character_picture'].change(chat.handle_character_picture_change, gradio('character_picture'), None, show_progress=False)

    shared.gradio['mode'].change(
        ui.gather_interface_values, gradio(shared.input_elements), gradio('interface_state')).then(
        chat.handle_mode_change, gradio('interface_state'), gradio('history', 'display', 'chat_style', 'chat-instruct_command', 'unique_id'), show_progress=False).then(
        None, gradio('mode'), None, js="(mode) => {const characterContainer = document.getElementById('character-menu').parentNode.parentNode; const isInChatTab = document.querySelector('#chat-controls').contains(characterContainer); if (isInChatTab) { characterContainer.style.display = mode === 'instruct' ? 'none' : ''; } if (mode === 'instruct') document.querySelectorAll('.bigProfilePicture').forEach(el => el.remove());}")

    shared.gradio['chat_style'].change(chat.redraw_html, gradio(reload_arr), gradio('display'), show_progress=False)

    shared.gradio['navigate_version'].click(
        ui.gather_interface_values, gradio(shared.input_elements), gradio('interface_state')).then(
        chat.handle_navigate_version_click, gradio('interface_state'), gradio('history', 'display'), show_progress=False)

    shared.gradio['edit_message'].click(
        ui.gather_interface_values, gradio(shared.input_elements), gradio('interface_state')).then(
        chat.handle_edit_message_click, gradio('interface_state'), gradio('history', 'display'), show_progress=False)

    # Save/delete a character
    shared.gradio['save_character'].click(chat.handle_save_character_click, gradio('name2'), gradio('save_character_filename', 'character_saver'), show_progress=False)
    shared.gradio['delete_character'].click(lambda: gr.update(visible=True), None, gradio('character_deleter'), show_progress=False)
    shared.gradio['load_template'].click(chat.handle_load_template_click, gradio('instruction_template'), gradio('instruction_template_str', 'instruction_template'), show_progress=False)
    shared.gradio['save_template'].click(
        ui.gather_interface_values, gradio(shared.input_elements), gradio('interface_state')).then(
        chat.handle_save_template_click, gradio('instruction_template_str'), gradio('save_filename', 'save_root', 'save_contents', 'file_saver'), show_progress=False)

    shared.gradio['restore_character'].click(
        ui.gather_interface_values, gradio(shared.input_elements), gradio('interface_state')).then(
        chat.restore_character_for_ui, gradio('interface_state'), gradio('interface_state', 'name2', 'context', 'greeting', 'character_picture'), show_progress=False)

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

    shared.gradio['send_instruction_to_notebook'].click(
        ui.gather_interface_values, gradio(shared.input_elements), gradio('interface_state')).then(
        chat.handle_send_instruction_click, gradio('interface_state'), gradio('textbox-notebook', 'textbox-default', 'output_textbox'), show_progress=False).then(
        None, None, None, js=f'() => {{{ui.switch_tabs_js}; switch_to_notebook()}}')

    shared.gradio['send-chat-to-notebook'].click(
        ui.gather_interface_values, gradio(shared.input_elements), gradio('interface_state')).then(
        chat.handle_send_chat_click, gradio('interface_state'), gradio('textbox-notebook', 'textbox-default', 'output_textbox'), show_progress=False).then(
        None, None, None, js=f'() => {{{ui.switch_tabs_js}; switch_to_notebook()}}')

    shared.gradio['show_controls'].change(None, gradio('show_controls'), None, js=f'(x) => {{{ui.show_controls_js}; toggle_controls(x)}}')

    shared.gradio['count_tokens'].click(
        ui.gather_interface_values, gradio(shared.input_elements), gradio('interface_state')).then(
        chat.count_prompt_tokens, gradio('textbox', 'interface_state'), gradio('token_display'), show_progress=False)

    shared.gradio['enable_web_search'].change(
        lambda x: gr.update(visible=x),
        gradio('enable_web_search'),
        gradio('web_search_row')
    )
