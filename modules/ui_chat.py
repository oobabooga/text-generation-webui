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
clear_arr = ('delete_chat-confirm', 'delete_chat', 'delete_chat-cancel')


def create_ui():
    mu = shared.args.multi_user

    shared.gradio['Chat input'] = gr.State()
    shared.gradio['history'] = gr.State({'internal': [], 'visible': []})

    with gr.Tab('聊天', elem_id='chat-tab', elem_classes=("old-ui" if shared.args.chat_buttons else None)):
        with gr.Row():
            with gr.Column(elem_id='chat-col'):
                shared.gradio['display'] = gr.HTML(value=chat_html_wrapper({'internal': [], 'visible': []}, '', '', 'chat', 'cai-chat', ''))

                with gr.Row(elem_id="chat-input-row"):
                    with gr.Column(scale=1, elem_id='gr-hover-container'):
                        gr.HTML(value='<div class="hover-element" onclick="void(0)"><span style="width: 100px; display: block" id="hover-element-button">&#9776;</span><div class="hover-menu" id="hover-menu"></div>', elem_id='gr-hover')

                    with gr.Column(scale=10, elem_id='chat-input-container'):
                        shared.gradio['textbox'] = gr.Textbox(label='', placeholder='发送消息', elem_id='chat-input', elem_classes=['add_scrollbar'])
                        shared.gradio['show_controls'] = gr.Checkbox(value=shared.settings['show_controls'], label='显示控件 (Ctrl+S)', elem_id='show-controls')
                        shared.gradio['typing-dots'] = gr.HTML(value='<div class="typing"><span></span><span class="dot1"></span><span class="dot2"></span></div>', label='打字中', elem_id='typing-container')

                    with gr.Column(scale=1, elem_id='generate-stop-container'):
                        with gr.Row():
                            shared.gradio['Stop'] = gr.Button('停止', elem_id='stop', visible=False)
                            shared.gradio['Generate'] = gr.Button('生成', elem_id='Generate', variant='primary')

        # Hover menu buttons
        with gr.Column(elem_id='chat-buttons'):
            with gr.Row():
                shared.gradio['Regenerate'] = gr.Button('重新生成 (Ctrl + Enter)', elem_id='Regenerate')
                shared.gradio['Continue'] = gr.Button('继续 (Alt + Enter)', elem_id='Continue')
                shared.gradio['Remove last'] = gr.Button('删除上一条 (Ctrl + Shift + Backspace)', elem_id='Remove-last')

            with gr.Row():
                shared.gradio['Replace last reply'] = gr.Button('替换上一条回复 (Ctrl + Shift + L)', elem_id='Replace-last')
                shared.gradio['Copy last reply'] = gr.Button('复制上一条回复 (Ctrl + Shift + K)', elem_id='Copy-last')
                shared.gradio['Impersonate'] = gr.Button('AI帮答 (Ctrl + Shift + M)', elem_id='Impersonate')

            with gr.Row():
                shared.gradio['Send dummy message'] = gr.Button('发送空消息')
                shared.gradio['Send dummy reply'] = gr.Button('触发空回复')

            with gr.Row():
                shared.gradio['send-chat-to-default'] = gr.Button('发送至默认')
                shared.gradio['send-chat-to-notebook'] = gr.Button('发送至笔记本')

        with gr.Row(elem_id='past-chats-row', elem_classes=['pretty_scrollbar']):
            with gr.Column():
                with gr.Row():
                    shared.gradio['unique_id'] = gr.Dropdown(label='过往聊天', elem_classes=['slim-dropdown'], interactive=not mu)

                with gr.Row():
                    shared.gradio['rename_chat'] = gr.Button('重命名', elem_classes='refresh-button', interactive=not mu)
                    shared.gradio['delete_chat'] = gr.Button('🗑️', elem_classes='refresh-button', interactive=not mu)
                    shared.gradio['delete_chat-confirm'] = gr.Button('确认', variant='stop', visible=False, elem_classes='refresh-button')
                    shared.gradio['delete_chat-cancel'] = gr.Button('取消', visible=False, elem_classes='refresh-button')
                    shared.gradio['Start new chat'] = gr.Button('新建聊天', elem_classes='refresh-button')

                with gr.Row(elem_id='rename-row'):
                    shared.gradio['rename_to'] = gr.Textbox(label='重命名为：', placeholder='新名称', visible=False, elem_classes=['no-background'])
                    shared.gradio['rename_to-confirm'] = gr.Button('确认', visible=False, elem_classes='refresh-button')
                    shared.gradio['rename_to-cancel'] = gr.Button('取消', visible=False, elem_classes='refresh-button')

        with gr.Row(elem_id='chat-controls', elem_classes=['pretty_scrollbar']):
            with gr.Column():
                with gr.Row():
                    shared.gradio['start_with'] = gr.Textbox(label='回复开头', placeholder='当然可以！', value=shared.settings['start_with'], elem_classes=['add_scrollbar'])

                with gr.Row():
                    shared.gradio['mode'] = gr.Radio(choices=['chat', 'chat-instruct', 'instruct'], value='chat', label='模式', info='定义如何生成聊天提示。在 instruct 和 chat-instruct 模式下，参数 > 指令模板下选择的指令模板必须与当前模型匹配。', elem_id='chat-mode')

                with gr.Row():
                    shared.gradio['chat_style'] = gr.Dropdown(choices=utils.get_available_chat_styles(), label='聊天界面风格', value=shared.settings['chat_style'], visible=shared.settings['mode'] != 'instruct')


def create_chat_settings_ui():
    mu = shared.args.multi_user
    with gr.Tab('角色'):
        with gr.Row():
            with gr.Column(scale=8):
                with gr.Row():
                    shared.gradio['character_menu'] = gr.Dropdown(value=None, choices=utils.get_available_characters(), label='角色', elem_id='character-menu', info='用于聊天和聊天指令模式。', elem_classes='slim-dropdown')
                    ui.create_refresh_button(shared.gradio['character_menu'], lambda: None, lambda: {'choices': utils.get_available_characters()}, 'refresh-button', interactive=not mu)
                    shared.gradio['save_character'] = gr.Button('💾', elem_classes='refresh-button', interactive=not mu)
                    shared.gradio['delete_character'] = gr.Button('🗑️', elem_classes='refresh-button', interactive=not mu)

                shared.gradio['name1'] = gr.Textbox(value=shared.settings['name1'], lines=1, label='你的名字')
                shared.gradio['name2'] = gr.Textbox(value='', lines=1, label='角色的名字')
                shared.gradio['context'] = gr.Textbox(value='', lines=10, label='背景', elem_classes=['add_scrollbar'])
                shared.gradio['greeting'] = gr.Textbox(value='', lines=5, label='问候', elem_classes=['add_scrollbar'])

            with gr.Column(scale=1):
                shared.gradio['character_picture'] = gr.Image(label='角色图片', type='pil', interactive=not mu)
                shared.gradio['your_picture'] = gr.Image(label='你的图片', type='pil', value=Image.open(Path('cache/pfp_me.png')) if Path('cache/pfp_me.png').exists() else None, interactive=not mu)

    with gr.Tab('指令模板'):
        with gr.Row():
            with gr.Column():
                with gr.Row():
                    shared.gradio['instruction_template'] = gr.Dropdown(choices=utils.get_available_instruction_templates(), label='已保存的指令模板', info="选择模板后，点击“加载”来加载并应用它。", value='None', elem_classes='slim-dropdown')
                    ui.create_refresh_button(shared.gradio['instruction_template'], lambda: None, lambda: {'choices': utils.get_available_instruction_templates()}, 'refresh-button', interactive=not mu)
                    shared.gradio['load_template'] = gr.Button("加载", elem_classes='refresh-button')
                    shared.gradio['save_template'] = gr.Button('💾', elem_classes='refresh-button', interactive=not mu)
                    shared.gradio['delete_template'] = gr.Button('🗑️ ', elem_classes='refresh-button', interactive=not mu)

            with gr.Row():
                with gr.Column():
                    shared.gradio['custom_system_message'] = gr.Textbox(value=shared.settings['custom_system_message'], lines=2, label='自定义系统消息', info='如果不为空，将代替默认消息使用。', elem_classes=['add_scrollbar'])
                    shared.gradio['instruction_template_str'] = gr.Textbox(value='', label='指令模板', lines=24, info='根据你正在使用的模型/LoRA进行更改。在指令和聊天指令模式下使用。', elem_classes=['add_scrollbar', 'monospace'])
                    with gr.Row():
                        shared.gradio['send_instruction_to_default'] = gr.Button('发送至默认', elem_classes=['small-button'])
                        shared.gradio['send_instruction_to_notebook'] = gr.Button('发送至笔记本', elem_classes=['small-button'])
                        shared.gradio['send_instruction_to_negative_prompt'] = gr.Button('发送至负面提示', elem_classes=['small-button'])

            with gr.Column():
                shared.gradio['chat_template_str'] = gr.Textbox(value=shared.settings['chat_template_str'], label='聊天模板', lines=22, elem_classes=['add_scrollbar', 'monospace'])
                shared.gradio['chat-instruct_command'] = gr.Textbox(value=shared.settings['chat-instruct_command'], lines=4, label='聊天指令模式命令', info='<|character|> 将被角色名称替换，<|prompt|> 将被常规聊天提示替换。', elem_classes=['add_scrollbar'])

    with gr.Tab('聊天记录'):
        with gr.Row():
            with gr.Column():
                shared.gradio['save_chat_history'] = gr.Button(value='保存历史记录')

            with gr.Column():
                shared.gradio['load_chat_history'] = gr.File(type='binary', file_types=['.json', '.txt'], label='上传历史记录 JSON')

    with gr.Tab('上传角色'):
        with gr.Tab('YAML 或 JSON'):
            with gr.Row():
                shared.gradio['upload_json'] = gr.File(type='binary', file_types=['.json', '.yaml'], label='JSON 或 YAML 文件', interactive=not mu)
                shared.gradio['upload_img_bot'] = gr.Image(type='pil', label='头像图片（可选）', interactive=not mu)

            shared.gradio['Submit character'] = gr.Button(value='提交', interactive=False)

        with gr.Tab('TavernAI PNG'):
            with gr.Row():
                with gr.Column():
                    shared.gradio['upload_img_tavern'] = gr.Image(type='pil', label='TavernAI PNG 文件', elem_id='upload_img_tavern', interactive=not mu)
                    shared.gradio['tavern_json'] = gr.State()
                with gr.Column():
                    shared.gradio['tavern_name'] = gr.Textbox(value='', lines=1, label='名称', interactive=False)
                    shared.gradio['tavern_desc'] = gr.Textbox(value='', lines=4, max_lines=4, label='描述', interactive=False)

            shared.gradio['Submit tavern character'] = gr.Button(value='提交', interactive=False)


def create_event_handlers():

    # Obsolete variables, kept for compatibility with old extensions
    shared.input_params = gradio(inputs)
    shared.reload_inputs = gradio(reload_arr)

    shared.gradio['Generate'].click(
        ui.gather_interface_values, gradio(shared.input_elements), gradio('interface_state')).then(
        lambda x: (x, ''), gradio('textbox'), gradio('Chat input', 'textbox'), show_progress=False).then(
        chat.generate_chat_reply_wrapper, gradio(inputs), gradio('display', 'history'), show_progress=False).then(
        ui.gather_interface_values, gradio(shared.input_elements), gradio('interface_state')).then(
        chat.save_history, gradio('history', 'unique_id', 'character_menu', 'mode'), None).then(
        lambda: None, None, None, _js=f'() => {{{ui.audio_notification_js}}}')

    shared.gradio['textbox'].submit(
        ui.gather_interface_values, gradio(shared.input_elements), gradio('interface_state')).then(
        lambda x: (x, ''), gradio('textbox'), gradio('Chat input', 'textbox'), show_progress=False).then(
        chat.generate_chat_reply_wrapper, gradio(inputs), gradio('display', 'history'), show_progress=False).then(
        ui.gather_interface_values, gradio(shared.input_elements), gradio('interface_state')).then(
        chat.save_history, gradio('history', 'unique_id', 'character_menu', 'mode'), None).then(
        lambda: None, None, None, _js=f'() => {{{ui.audio_notification_js}}}')

    shared.gradio['Regenerate'].click(
        ui.gather_interface_values, gradio(shared.input_elements), gradio('interface_state')).then(
        partial(chat.generate_chat_reply_wrapper, regenerate=True), gradio(inputs), gradio('display', 'history'), show_progress=False).then(
        ui.gather_interface_values, gradio(shared.input_elements), gradio('interface_state')).then(
        chat.save_history, gradio('history', 'unique_id', 'character_menu', 'mode'), None).then(
        lambda: None, None, None, _js=f'() => {{{ui.audio_notification_js}}}')

    shared.gradio['Continue'].click(
        ui.gather_interface_values, gradio(shared.input_elements), gradio('interface_state')).then(
        partial(chat.generate_chat_reply_wrapper, _continue=True), gradio(inputs), gradio('display', 'history'), show_progress=False).then(
        ui.gather_interface_values, gradio(shared.input_elements), gradio('interface_state')).then(
        chat.save_history, gradio('history', 'unique_id', 'character_menu', 'mode'), None).then(
        lambda: None, None, None, _js=f'() => {{{ui.audio_notification_js}}}')

    shared.gradio['Impersonate'].click(
        ui.gather_interface_values, gradio(shared.input_elements), gradio('interface_state')).then(
        lambda x: x, gradio('textbox'), gradio('Chat input'), show_progress=False).then(
        chat.impersonate_wrapper, gradio(inputs), gradio('textbox', 'display'), show_progress=False).then(
        ui.gather_interface_values, gradio(shared.input_elements), gradio('interface_state')).then(
        lambda: None, None, None, _js=f'() => {{{ui.audio_notification_js}}}')

    shared.gradio['Replace last reply'].click(
        ui.gather_interface_values, gradio(shared.input_elements), gradio('interface_state')).then(
        chat.replace_last_reply, gradio('textbox', 'interface_state'), gradio('history')).then(
        lambda: '', None, gradio('textbox'), show_progress=False).then(
        chat.redraw_html, gradio(reload_arr), gradio('display')).then(
        chat.save_history, gradio('history', 'unique_id', 'character_menu', 'mode'), None)

    shared.gradio['Send dummy message'].click(
        ui.gather_interface_values, gradio(shared.input_elements), gradio('interface_state')).then(
        chat.send_dummy_message, gradio('textbox', 'interface_state'), gradio('history')).then(
        lambda: '', None, gradio('textbox'), show_progress=False).then(
        chat.redraw_html, gradio(reload_arr), gradio('display')).then(
        chat.save_history, gradio('history', 'unique_id', 'character_menu', 'mode'), None)

    shared.gradio['Send dummy reply'].click(
        ui.gather_interface_values, gradio(shared.input_elements), gradio('interface_state')).then(
        chat.send_dummy_reply, gradio('textbox', 'interface_state'), gradio('history')).then(
        lambda: '', None, gradio('textbox'), show_progress=False).then(
        chat.redraw_html, gradio(reload_arr), gradio('display')).then(
        chat.save_history, gradio('history', 'unique_id', 'character_menu', 'mode'), None)

    shared.gradio['Remove last'].click(
        ui.gather_interface_values, gradio(shared.input_elements), gradio('interface_state')).then(
        chat.remove_last_message, gradio('history'), gradio('textbox', 'history'), show_progress=False).then(
        chat.redraw_html, gradio(reload_arr), gradio('display')).then(
        chat.save_history, gradio('history', 'unique_id', 'character_menu', 'mode'), None)

    shared.gradio['Stop'].click(
        stop_everything_event, None, None, queue=False).then(
        chat.redraw_html, gradio(reload_arr), gradio('display'))

    if not shared.args.multi_user:
        shared.gradio['unique_id'].select(
            chat.load_history, gradio('unique_id', 'character_menu', 'mode'), gradio('history')).then(
            chat.redraw_html, gradio(reload_arr), gradio('display'))

    shared.gradio['Start new chat'].click(
        ui.gather_interface_values, gradio(shared.input_elements), gradio('interface_state')).then(
        chat.start_new_chat, gradio('interface_state'), gradio('history')).then(
        chat.redraw_html, gradio(reload_arr), gradio('display')).then(
        lambda x: gr.update(choices=(histories := chat.find_all_histories(x)), value=histories[0]), gradio('interface_state'), gradio('unique_id'))

    shared.gradio['delete_chat'].click(lambda: [gr.update(visible=True), gr.update(visible=False), gr.update(visible=True)], None, gradio(clear_arr))
    shared.gradio['delete_chat-cancel'].click(lambda: [gr.update(visible=False), gr.update(visible=True), gr.update(visible=False)], None, gradio(clear_arr))
    shared.gradio['delete_chat-confirm'].click(
        ui.gather_interface_values, gradio(shared.input_elements), gradio('interface_state')).then(
        lambda x, y: str(chat.find_all_histories(x).index(y)), gradio('interface_state', 'unique_id'), gradio('temporary_text')).then(
        chat.delete_history, gradio('unique_id', 'character_menu', 'mode'), None).then(
        chat.load_history_after_deletion, gradio('interface_state', 'temporary_text'), gradio('history', 'unique_id')).then(
        chat.redraw_html, gradio(reload_arr), gradio('display')).then(
        lambda: [gr.update(visible=False), gr.update(visible=True), gr.update(visible=False)], None, gradio(clear_arr))

    shared.gradio['rename_chat'].click(
        lambda x: x, gradio('unique_id'), gradio('rename_to')).then(
        lambda: [gr.update(visible=True)] * 3, None, gradio('rename_to', 'rename_to-confirm', 'rename_to-cancel'), show_progress=False)

    shared.gradio['rename_to-cancel'].click(
        lambda: [gr.update(visible=False)] * 3, None, gradio('rename_to', 'rename_to-confirm', 'rename_to-cancel'), show_progress=False)

    shared.gradio['rename_to-confirm'].click(
        chat.rename_history, gradio('unique_id', 'rename_to', 'character_menu', 'mode'), None).then(
        lambda: [gr.update(visible=False)] * 3, None, gradio('rename_to', 'rename_to-confirm', 'rename_to-cancel'), show_progress=False).then(
        lambda x, y: gr.update(choices=chat.find_all_histories(x), value=y), gradio('interface_state', 'rename_to'), gradio('unique_id'))

    shared.gradio['rename_to'].submit(
        chat.rename_history, gradio('unique_id', 'rename_to', 'character_menu', 'mode'), None).then(
        lambda: [gr.update(visible=False)] * 3, None, gradio('rename_to', 'rename_to-confirm', 'rename_to-cancel'), show_progress=False).then(
        lambda x, y: gr.update(choices=chat.find_all_histories(x), value=y), gradio('interface_state', 'rename_to'), gradio('unique_id'))

    shared.gradio['load_chat_history'].upload(
        ui.gather_interface_values, gradio(shared.input_elements), gradio('interface_state')).then(
        chat.start_new_chat, gradio('interface_state'), gradio('history')).then(
        chat.load_history_json, gradio('load_chat_history', 'history'), gradio('history')).then(
        chat.redraw_html, gradio(reload_arr), gradio('display')).then(
        lambda x: gr.update(choices=(histories := chat.find_all_histories(x)), value=histories[0]), gradio('interface_state'), gradio('unique_id')).then(
        chat.save_history, gradio('history', 'unique_id', 'character_menu', 'mode'), None).then(
        lambda: None, None, None, _js=f'() => {{{ui.switch_tabs_js}; switch_to_chat()}}')

    shared.gradio['character_menu'].change(
        chat.load_character, gradio('character_menu', 'name1', 'name2'), gradio('name1', 'name2', 'character_picture', 'greeting', 'context')).success(
        ui.gather_interface_values, gradio(shared.input_elements), gradio('interface_state')).then(
        chat.load_latest_history, gradio('interface_state'), gradio('history')).then(
        chat.redraw_html, gradio(reload_arr), gradio('display')).then(
        lambda x: gr.update(choices=(histories := chat.find_all_histories(x)), value=histories[0]), gradio('interface_state'), gradio('unique_id')).then(
        lambda: None, None, None, _js=f'() => {{{ui.update_big_picture_js}; updateBigPicture()}}')

    shared.gradio['mode'].change(
        lambda x: gr.update(visible=x != 'instruct'), gradio('mode'), gradio('chat_style'), show_progress=False).then(
        ui.gather_interface_values, gradio(shared.input_elements), gradio('interface_state')).then(
        chat.load_latest_history, gradio('interface_state'), gradio('history')).then(
        chat.redraw_html, gradio(reload_arr), gradio('display')).then(
        lambda x: gr.update(choices=(histories := chat.find_all_histories(x)), value=histories[0]), gradio('interface_state'), gradio('unique_id'))

    shared.gradio['chat_style'].change(chat.redraw_html, gradio(reload_arr), gradio('display'))
    shared.gradio['Copy last reply'].click(chat.send_last_reply_to_input, gradio('history'), gradio('textbox'), show_progress=False)

    # Save/delete a character
    shared.gradio['save_character'].click(
        lambda x: x, gradio('name2'), gradio('save_character_filename')).then(
        lambda: gr.update(visible=True), None, gradio('character_saver'))

    shared.gradio['delete_character'].click(lambda: gr.update(visible=True), None, gradio('character_deleter'))

    shared.gradio['load_template'].click(
        chat.load_instruction_template, gradio('instruction_template'), gradio('instruction_template_str')).then(
        lambda: "Select template to load...", None, gradio('instruction_template'))

    shared.gradio['save_template'].click(
        lambda: 'My Template.yaml', None, gradio('save_filename')).then(
        lambda: 'instruction-templates/', None, gradio('save_root')).then(
        chat.generate_instruction_template_yaml, gradio('instruction_template_str'), gradio('save_contents')).then(
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
        ui.gather_interface_values, gradio(shared.input_elements), gradio('interface_state')).then(
        lambda x: x.update({'mode': 'instruct', 'history': {'internal': [], 'visible': []}}), gradio('interface_state'), None).then(
        partial(chat.generate_chat_prompt, 'Input'), gradio('interface_state'), gradio('textbox-default')).then(
        lambda: None, None, None, _js=f'() => {{{ui.switch_tabs_js}; switch_to_default()}}')

    shared.gradio['send_instruction_to_notebook'].click(
        ui.gather_interface_values, gradio(shared.input_elements), gradio('interface_state')).then(
        lambda x: x.update({'mode': 'instruct', 'history': {'internal': [], 'visible': []}}), gradio('interface_state'), None).then(
        partial(chat.generate_chat_prompt, 'Input'), gradio('interface_state'), gradio('textbox-notebook')).then(
        lambda: None, None, None, _js=f'() => {{{ui.switch_tabs_js}; switch_to_notebook()}}')

    shared.gradio['send_instruction_to_negative_prompt'].click(
        ui.gather_interface_values, gradio(shared.input_elements), gradio('interface_state')).then(
        lambda x: x.update({'mode': 'instruct', 'history': {'internal': [], 'visible': []}}), gradio('interface_state'), None).then(
        partial(chat.generate_chat_prompt, 'Input'), gradio('interface_state'), gradio('negative_prompt')).then(
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
