import threading
import time
from pathlib import Path

import gradio as gr

from modules import logits, shared, ui, utils
from modules.prompts import count_tokens, load_prompt
from modules.text_generation import (
    generate_reply_wrapper,
    get_token_ids,
    stop_everything_event
)
from modules.utils import gradio

_notebook_file_lock = threading.Lock()
_notebook_auto_save_timer = None
_last_notebook_text = None
_last_notebook_prompt = None

inputs = ('textbox-notebook', 'interface_state')
outputs = ('textbox-notebook', 'html-notebook')


def create_ui():
    mu = shared.args.multi_user
    with gr.Row(visible=not shared.settings['show_two_notebook_columns']) as shared.gradio['notebook-tab']:
        shared.gradio['last_input-notebook'] = gr.State('')
        with gr.Row():
            with gr.Column(scale=4):
                with gr.Tab('Raw'):
                    with gr.Row():
                        initial_text = load_prompt(shared.settings['prompt-notebook'])
                        shared.gradio['textbox-notebook'] = gr.Textbox(label="", value=initial_text, lines=27, elem_id='textbox-notebook', elem_classes=['textbox', 'add_scrollbar'])
                        shared.gradio['token-counter-notebook'] = gr.HTML(value="<span>0</span>", elem_id="notebook-token-counter")

                with gr.Tab('Markdown'):
                    shared.gradio['markdown_render-notebook'] = gr.Button('Render')
                    shared.gradio['markdown-notebook'] = gr.Markdown()

                with gr.Tab('HTML'):
                    shared.gradio['html-notebook'] = gr.HTML()

                with gr.Tab('Logits'):
                    with gr.Row():
                        with gr.Column(scale=10):
                            shared.gradio['get_logits-notebook'] = gr.Button('Get next token probabilities')
                        with gr.Column(scale=1):
                            shared.gradio['use_samplers-notebook'] = gr.Checkbox(label='Use samplers', value=True, elem_classes=['no-background'])

                    with gr.Row():
                        shared.gradio['logits-notebook'] = gr.Textbox(lines=23, label='Output', elem_classes=['textbox_logits_notebook', 'add_scrollbar'])
                        shared.gradio['logits-notebook-previous'] = gr.Textbox(lines=23, label='Previous output', elem_classes=['textbox_logits_notebook', 'add_scrollbar'])

                with gr.Tab('Tokens'):
                    shared.gradio['get_tokens-notebook'] = gr.Button('Get token IDs for the input')
                    shared.gradio['tokens-notebook'] = gr.Textbox(lines=23, label='Tokens', elem_classes=['textbox_logits_notebook', 'add_scrollbar', 'monospace'])

                with gr.Row():
                    shared.gradio['Undo'] = gr.Button('Undo', elem_classes='small-button')
                    shared.gradio['Regenerate-notebook'] = gr.Button('Regenerate', elem_classes='small-button')
                    shared.gradio['Stop-notebook'] = gr.Button('Stop', visible=False, elem_classes='small-button', elem_id='stop')
                    shared.gradio['Generate-notebook'] = gr.Button('Generate', variant='primary', elem_classes='small-button')

            with gr.Column(scale=1):
                gr.HTML('<div style="padding-bottom: 13px"></div>')
                with gr.Row():
                    shared.gradio['prompt_menu-notebook'] = gr.Dropdown(choices=utils.get_available_prompts(), value=shared.settings['prompt-notebook'], label='Prompt', elem_classes='slim-dropdown')

                with gr.Row():
                    ui.create_refresh_button(shared.gradio['prompt_menu-notebook'], lambda: None, lambda: {'choices': utils.get_available_prompts()}, ['refresh-button'], interactive=not mu)
                    shared.gradio['new_prompt-notebook'] = gr.Button('New', elem_classes=['refresh-button'], interactive=not mu)
                    shared.gradio['rename_prompt-notebook'] = gr.Button('Rename', elem_classes=['refresh-button'], interactive=not mu)
                    shared.gradio['delete_prompt-notebook'] = gr.Button('🗑️', elem_classes=['refresh-button'], interactive=not mu)
                    shared.gradio['delete_prompt-confirm-notebook'] = gr.Button('Confirm', variant='stop', elem_classes=['refresh-button'], visible=False)
                    shared.gradio['delete_prompt-cancel-notebook'] = gr.Button('Cancel', elem_classes=['refresh-button'], visible=False)

                with gr.Row(visible=False) as shared.gradio['rename-row-notebook']:
                    shared.gradio['rename_prompt_to-notebook'] = gr.Textbox(label="New name", elem_classes=['no-background'])
                    shared.gradio['rename_prompt-cancel-notebook'] = gr.Button('Cancel', elem_classes=['refresh-button'])
                    shared.gradio['rename_prompt-confirm-notebook'] = gr.Button('Confirm', elem_classes=['refresh-button'], variant='primary')


def create_event_handlers():
    shared.gradio['Generate-notebook'].click(
        lambda x: x, gradio('textbox-notebook'), gradio('last_input-notebook')).then(
        ui.gather_interface_values, gradio(shared.input_elements), gradio('interface_state')).then(
        lambda: [gr.update(visible=True), gr.update(visible=False)], None, gradio('Stop-notebook', 'Generate-notebook')).then(
        generate_and_save_wrapper_notebook, gradio('textbox-notebook', 'interface_state', 'prompt_menu-notebook'), gradio(outputs), show_progress=False).then(
        lambda state, text: state.update({'textbox-notebook': text}), gradio('interface_state', 'textbox-notebook'), None).then(
        lambda: [gr.update(visible=False), gr.update(visible=True)], None, gradio('Stop-notebook', 'Generate-notebook')).then(
        None, None, None, js=f'() => {{{ui.audio_notification_js}}}')

    shared.gradio['textbox-notebook'].submit(
        lambda x: x, gradio('textbox-notebook'), gradio('last_input-notebook')).then(
        ui.gather_interface_values, gradio(shared.input_elements), gradio('interface_state')).then(
        lambda: [gr.update(visible=True), gr.update(visible=False)], None, gradio('Stop-notebook', 'Generate-notebook')).then(
        generate_and_save_wrapper_notebook, gradio('textbox-notebook', 'interface_state', 'prompt_menu-notebook'), gradio(outputs), show_progress=False).then(
        lambda state, text: state.update({'textbox-notebook': text}), gradio('interface_state', 'textbox-notebook'), None).then(
        lambda: [gr.update(visible=False), gr.update(visible=True)], None, gradio('Stop-notebook', 'Generate-notebook')).then(
        None, None, None, js=f'() => {{{ui.audio_notification_js}}}')

    shared.gradio['Regenerate-notebook'].click(
        lambda x: x, gradio('last_input-notebook'), gradio('textbox-notebook'), show_progress=False).then(
        ui.gather_interface_values, gradio(shared.input_elements), gradio('interface_state')).then(
        lambda: [gr.update(visible=True), gr.update(visible=False)], None, gradio('Stop-notebook', 'Generate-notebook')).then(
        generate_and_save_wrapper_notebook, gradio('textbox-notebook', 'interface_state', 'prompt_menu-notebook'), gradio(outputs), show_progress=False).then(
        lambda state, text: state.update({'textbox-notebook': text}), gradio('interface_state', 'textbox-notebook'), None).then(
        lambda: [gr.update(visible=False), gr.update(visible=True)], None, gradio('Stop-notebook', 'Generate-notebook')).then(
        None, None, None, js=f'() => {{{ui.audio_notification_js}}}')

    shared.gradio['Undo'].click(
        lambda x: x, gradio('last_input-notebook'), gradio('textbox-notebook'), show_progress=False).then(
        lambda state, text: state.update({'textbox-notebook': text}), gradio('interface_state', 'textbox-notebook'), None)

    shared.gradio['markdown_render-notebook'].click(lambda x: x, gradio('textbox-notebook'), gradio('markdown-notebook'), queue=False)
    shared.gradio['Stop-notebook'].click(stop_everything_event, None, None, queue=False)
    shared.gradio['prompt_menu-notebook'].change(load_prompt, gradio('prompt_menu-notebook'), gradio('textbox-notebook'), show_progress=False)
    shared.gradio['new_prompt-notebook'].click(handle_new_prompt, None, gradio('prompt_menu-notebook'), show_progress=False)

    shared.gradio['delete_prompt-notebook'].click(
        lambda: [gr.update(visible=False), gr.update(visible=True), gr.update(visible=True)],
        None,
        gradio('delete_prompt-notebook', 'delete_prompt-cancel-notebook', 'delete_prompt-confirm-notebook'),
        show_progress=False)

    shared.gradio['delete_prompt-cancel-notebook'].click(
        lambda: [gr.update(visible=True), gr.update(visible=False), gr.update(visible=False)],
        None,
        gradio('delete_prompt-notebook', 'delete_prompt-cancel-notebook', 'delete_prompt-confirm-notebook'),
        show_progress=False)

    shared.gradio['delete_prompt-confirm-notebook'].click(
        handle_delete_prompt_confirm_notebook,
        gradio('prompt_menu-notebook'),
        gradio('prompt_menu-notebook', 'delete_prompt-notebook', 'delete_prompt-cancel-notebook', 'delete_prompt-confirm-notebook'),
        show_progress=False)

    shared.gradio['rename_prompt-notebook'].click(
        handle_rename_prompt_click_notebook,
        gradio('prompt_menu-notebook'),
        gradio('rename_prompt_to-notebook', 'rename_prompt-notebook', 'rename-row-notebook'),
        show_progress=False)

    shared.gradio['rename_prompt-cancel-notebook'].click(
        lambda: [gr.update(visible=True), gr.update(visible=False)],
        None,
        gradio('rename_prompt-notebook', 'rename-row-notebook'),
        show_progress=False)

    shared.gradio['rename_prompt-confirm-notebook'].click(
        handle_rename_prompt_confirm_notebook,
        gradio('rename_prompt_to-notebook', 'prompt_menu-notebook'),
        gradio('prompt_menu-notebook', 'rename_prompt-notebook', 'rename-row-notebook'),
        show_progress=False)

    shared.gradio['textbox-notebook'].input(lambda x: f"<span>{count_tokens(x)}</span>", gradio('textbox-notebook'), gradio('token-counter-notebook'), show_progress=False)
    shared.gradio['textbox-notebook'].change(
        store_notebook_state_and_debounce,
        gradio('textbox-notebook', 'prompt_menu-notebook'),
        None,
        show_progress=False
    )

    shared.gradio['get_logits-notebook'].click(
        ui.gather_interface_values, gradio(shared.input_elements), gradio('interface_state')).then(
        logits.get_next_logits, gradio('textbox-notebook', 'interface_state', 'use_samplers-notebook', 'logits-notebook'), gradio('logits-notebook', 'logits-notebook-previous'), show_progress=False)

    shared.gradio['get_tokens-notebook'].click(get_token_ids, gradio('textbox-notebook'), gradio('tokens-notebook'), show_progress=False)


def generate_and_save_wrapper_notebook(textbox_content, interface_state, prompt_name):
    """Generate reply and automatically save the result for notebook mode with periodic saves"""
    last_save_time = time.monotonic()
    save_interval = 8
    output = textbox_content

    # Initial autosave
    safe_autosave_prompt(output, prompt_name)

    for i, (output, html_output) in enumerate(generate_reply_wrapper(textbox_content, interface_state)):
        yield output, html_output

        current_time = time.monotonic()
        # Save on first iteration or if save_interval seconds have passed
        if i == 0 or (current_time - last_save_time) >= save_interval:
            safe_autosave_prompt(output, prompt_name)
            last_save_time = current_time

    # Final autosave
    safe_autosave_prompt(output, prompt_name)


def handle_new_prompt():
    new_name = utils.current_time()

    # Create the new prompt file
    prompt_path = Path("user_data/logs/notebook") / f"{new_name}.txt"
    prompt_path.parent.mkdir(parents=True, exist_ok=True)
    prompt_path.write_text("In this story,", encoding='utf-8')

    return gr.update(choices=utils.get_available_prompts(), value=new_name)


def handle_delete_prompt_confirm_notebook(prompt_name):
    available_prompts = utils.get_available_prompts()
    current_index = available_prompts.index(prompt_name) if prompt_name in available_prompts else 0

    (Path("user_data/logs/notebook") / f"{prompt_name}.txt").unlink(missing_ok=True)
    available_prompts = utils.get_available_prompts()

    if available_prompts:
        new_value = available_prompts[min(current_index, len(available_prompts) - 1)]
    else:
        new_value = utils.current_time()
        Path("user_data/logs/notebook").mkdir(parents=True, exist_ok=True)
        (Path("user_data/logs/notebook") / f"{new_value}.txt").write_text("In this story,")
        available_prompts = [new_value]

    return [
        gr.update(choices=available_prompts, value=new_value),
        gr.update(visible=True),
        gr.update(visible=False),
        gr.update(visible=False)
    ]


def handle_rename_prompt_click_notebook(current_name):
    return [
        gr.update(value=current_name),
        gr.update(visible=False),
        gr.update(visible=True)
    ]


def handle_rename_prompt_confirm_notebook(new_name, current_name):
    old_path = Path("user_data/logs/notebook") / f"{current_name}.txt"
    new_path = Path("user_data/logs/notebook") / f"{new_name}.txt"

    if old_path.exists() and not new_path.exists():
        old_path.rename(new_path)

    available_prompts = utils.get_available_prompts()
    return [
        gr.update(choices=available_prompts, value=new_name),
        gr.update(visible=True),
        gr.update(visible=False)
    ]


def autosave_prompt(text, prompt_name):
    """Automatically save the text to the selected prompt file"""
    if prompt_name and text.strip():
        prompt_path = Path("user_data/logs/notebook") / f"{prompt_name}.txt"
        prompt_path.parent.mkdir(parents=True, exist_ok=True)
        prompt_path.write_text(text, encoding='utf-8')


def safe_autosave_prompt(content, prompt_name):
    """Thread-safe wrapper for autosave_prompt to prevent file corruption"""
    with _notebook_file_lock:
        autosave_prompt(content, prompt_name)


def store_notebook_state_and_debounce(text, prompt_name):
    """Store current notebook state and trigger debounced save"""
    global _notebook_auto_save_timer, _last_notebook_text, _last_notebook_prompt

    if shared.args.multi_user:
        return

    _last_notebook_text = text
    _last_notebook_prompt = prompt_name

    if _notebook_auto_save_timer is not None:
        _notebook_auto_save_timer.cancel()

    _notebook_auto_save_timer = threading.Timer(1.0, _perform_notebook_debounced_save)
    _notebook_auto_save_timer.start()


def _perform_notebook_debounced_save():
    """Actually perform the notebook save using the stored state"""
    try:
        if _last_notebook_text is not None and _last_notebook_prompt is not None:
            safe_autosave_prompt(_last_notebook_text, _last_notebook_prompt)
    except Exception as e:
        print(f"Notebook auto-save failed: {e}")
