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
from modules.ui_default import autosave_prompt, handle_delete_prompt
from modules.utils import gradio

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
                        shared.gradio['textbox-notebook'] = gr.Textbox(value=initial_text, lines=27, elem_id='textbox-notebook', elem_classes=['textbox', 'add_scrollbar'])
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
                    ui.create_refresh_button(shared.gradio['prompt_menu-notebook'], lambda: None, lambda: {'choices': utils.get_available_prompts()}, ['refresh-button', 'refresh-button-small'], interactive=not mu)
                    shared.gradio['new_prompt-notebook'] = gr.Button('New', elem_classes=['refresh-button', 'refresh-button-small'], interactive=not mu)
                    shared.gradio['delete_prompt-notebook'] = gr.Button('🗑️', elem_classes=['refresh-button', 'refresh-button-small'], interactive=not mu)


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
    shared.gradio['new_prompt-notebook'].click(handle_new_prompt, None, gradio('textbox-notebook', 'prompt_menu-notebook'), show_progress=False)
    shared.gradio['delete_prompt-notebook'].click(handle_delete_prompt, gradio('prompt_menu-notebook'), gradio('delete_filename', 'delete_root', 'file_deleter'), show_progress=False)
    shared.gradio['textbox-notebook'].input(lambda x: f"<span>{count_tokens(x)}</span>", gradio('textbox-notebook'), gradio('token-counter-notebook'), show_progress=False)
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
    autosave_prompt(output, prompt_name)

    for i, (output, html_output) in enumerate(generate_reply_wrapper(textbox_content, interface_state)):
        yield output, html_output

        current_time = time.monotonic()
        # Save on first iteration or if save_interval seconds have passed
        if i == 0 or (current_time - last_save_time) >= save_interval:
            autosave_prompt(output, prompt_name)
            last_save_time = current_time

    # Final autosave
    autosave_prompt(output, prompt_name)


def handle_new_prompt():
    new_name = utils.current_time()

    # Create the new prompt file
    prompt_path = Path("user_data/logs/notebook") / f"{new_name}.txt"
    prompt_path.parent.mkdir(parents=True, exist_ok=True)
    prompt_path.write_text("", encoding='utf-8')

    return [
        "In this story,",
        gr.update(choices=utils.get_available_prompts(), value=new_name)
    ]
