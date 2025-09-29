import gradio as gr

from modules import shared, ui, utils
from modules.utils import gradio


def create_ui():
    mu = shared.args.multi_user
    with gr.Tab("Сессия", elem_id="session-tab"):
        with gr.Row():
            with gr.Column():
                gr.Markdown("## Настройки")
                shared.gradio['toggle_dark_mode'] = gr.Button('Переключить светлую/тёмную тему 💡', elem_classes='refresh-button')
                shared.gradio['show_two_notebook_columns'] = gr.Checkbox(label='Показать два столбца в вкладке Блокнот', value=shared.settings['show_two_notebook_columns'])
                shared.gradio['paste_to_attachment'] = gr.Checkbox(label='Превращать длинные вставленные тексты в вложения в вкладке Чат', value=shared.settings['paste_to_attachment'], elem_id='paste_to_attachment')
                shared.gradio['include_past_attachments'] = gr.Checkbox(label='Включать вложения/результаты поиска из предыдущих сообщений в промпт чата', value=shared.settings['include_past_attachments'])

            with gr.Column():
                gr.Markdown("## Расширения и флаги")
                shared.gradio['save_settings'] = gr.Button('Сохранить настройки расширений в user_data/settings.yaml', elem_classes='refresh-button', interactive=not mu)
                shared.gradio['reset_interface'] = gr.Button("Применить флаги/расширения и перезапустить", interactive=not mu)
                with gr.Row():
                    with gr.Column():
                        shared.gradio['extensions_menu'] = gr.CheckboxGroup(choices=utils.get_available_extensions(), value=shared.args.extensions, label="Доступные расширения", info='Обратите внимание, что некоторые из этих расширений могут потребовать ручной установки зависимостей Python: pip install -r extensions/extension_name/requirements.txt', elem_classes='checkboxgroup-table')

                    with gr.Column():
                        shared.gradio['bool_menu'] = gr.CheckboxGroup(choices=get_boolean_arguments(), value=get_boolean_arguments(active=True), label="Логические флаги командной строки", elem_classes='checkboxgroup-table')

        shared.gradio['theme_state'] = gr.Textbox(visible=False, value='dark' if shared.settings['dark_theme'] else 'light')
        if not mu:
            shared.gradio['save_settings'].click(
                ui.gather_interface_values, gradio(shared.input_elements), gradio('interface_state')).then(
                handle_save_settings, gradio('interface_state', 'preset_menu', 'extensions_menu', 'show_controls', 'theme_state'), gradio('save_contents', 'save_filename', 'save_root', 'file_saver'), show_progress=False)

        shared.gradio['toggle_dark_mode'].click(
            lambda x: 'dark' if x == 'light' else 'light', gradio('theme_state'), gradio('theme_state')).then(
            None, None, None, js=f'() => {{{ui.dark_theme_js}; toggleDarkMode(); localStorage.setItem("theme", document.body.classList.contains("dark") ? "dark" : "light")}}')

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
    return [
        contents,
        "settings.yaml",
        "user_data/",
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


def set_interface_arguments(extensions, bool_active):
    shared.args.extensions = extensions

    bool_list = get_boolean_arguments()

    for k in bool_list:
        setattr(shared.args, k, False)
    for k in bool_active:
        setattr(shared.args, k, True)
        if k == 'api':
            shared.add_extension('openai', last=True)

    shared.need_restart = True


def get_boolean_arguments(active=False):
    cmd_list = vars(shared.args)
    bool_list = sorted([k for k in cmd_list if type(cmd_list[k]) is bool and k not in ui.list_model_elements()])
    bool_active = [k for k in bool_list if vars(shared.args)[k]]

    if active:
        return bool_active
    else:
        return bool_list
