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

inputs = ('textbox-default', 'interface_state')
outputs = ('output_textbox', 'html-default')


def create_ui():
    mu = shared.args.multi_user
    with gr.Row(visible=shared.settings['show_two_notebook_columns']) as shared.gradio['default-tab']:
        with gr.Row():
            with gr.Column():
                with gr.Row():
                    initial_text = load_prompt(shared.settings['prompt-notebook'])
                    shared.gradio['textbox-default'] = gr.Textbox(value=initial_text, lines=27, label='Input', elem_classes=['textbox_default', 'add_scrollbar'])
                    shared.gradio['token-counter-default'] = gr.HTML(value="<span>0</span>", elem_id="default-token-counter")

                with gr.Row():
                    shared.gradio['Continue-default'] = gr.Button('Continue')
                    shared.gradio['Stop-default'] = gr.Button('Stop', elem_id='stop', visible=False)
                    shared.gradio['Generate-default'] = gr.Button('Generate', variant='primary')

                with gr.Row():
                    shared.gradio['prompt_menu-default'] = gr.Dropdown(choices=utils.get_available_prompts(), value=shared.settings['prompt-notebook'], label='Prompt', elem_classes='slim-dropdown')
                    ui.create_refresh_button(shared.gradio['prompt_menu-default'], lambda: None, lambda: {'choices': utils.get_available_prompts()}, 'refresh-button', interactive=not mu)
                    shared.gradio['new_prompt-default'] = gr.Button('New', elem_classes='refresh-button', interactive=not mu)
                    shared.gradio['delete_prompt-default'] = gr.Button('🗑️', elem_classes='refresh-button', interactive=not mu)

            with gr.Column():
                with gr.Tab('Raw'):
                    shared.gradio['output_textbox'] = gr.Textbox(lines=27, label='Output', elem_id='textbox-default', elem_classes=['textbox_default_output', 'add_scrollbar'])

                with gr.Tab('Markdown'):
                    shared.gradio['markdown_render-default'] = gr.Button('Render')
                    shared.gradio['markdown-default'] = gr.Markdown()

                with gr.Tab('HTML'):
                    shared.gradio['html-default'] = gr.HTML()

                with gr.Tab('Logits'):
                    with gr.Row():
                        with gr.Column(scale=10):
                            shared.gradio['get_logits-default'] = gr.Button('Get next token probabilities')
                        with gr.Column(scale=1):
                            shared.gradio['use_samplers-default'] = gr.Checkbox(label='Use samplers', value=True, elem_classes=['no-background'])

                    with gr.Row():
                        shared.gradio['logits-default'] = gr.Textbox(lines=23, label='Output', elem_classes=['textbox_logits', 'add_scrollbar'])
                        shared.gradio['logits-default-previous'] = gr.Textbox(lines=23, label='Previous output', elem_classes=['textbox_logits', 'add_scrollbar'])

                with gr.Tab('Tokens'):
                    shared.gradio['get_tokens-default'] = gr.Button('Get token IDs for the input')
                    shared.gradio['tokens-default'] = gr.Textbox(lines=23, label='Tokens', elem_classes=['textbox_logits', 'add_scrollbar', 'monospace'])


def create_event_handlers():
    shared.gradio['Generate-default'].click(
        ui.gather_interface_values, gradio(shared.input_elements), gradio('interface_state')).then(
        lambda: [gr.update(visible=True), gr.update(visible=False)], None, gradio('Stop-default', 'Generate-default')).then(
        generate_and_save_wrapper, gradio('textbox-default', 'interface_state', 'prompt_menu-default'), gradio(outputs), show_progress=False).then(
        lambda state, left, right: state.update({'textbox-default': left, 'output_textbox': right}), gradio('interface_state', 'textbox-default', 'output_textbox'), None).then(
        lambda: [gr.update(visible=False), gr.update(visible=True)], None, gradio('Stop-default', 'Generate-default')).then(
        None, None, None, js=f'() => {{{ui.audio_notification_js}}}')

    shared.gradio['textbox-default'].submit(
        ui.gather_interface_values, gradio(shared.input_elements), gradio('interface_state')).then(
        lambda: [gr.update(visible=True), gr.update(visible=False)], None, gradio('Stop-default', 'Generate-default')).then(
        generate_and_save_wrapper, gradio('textbox-default', 'interface_state', 'prompt_menu-default'), gradio(outputs), show_progress=False).then(
        lambda state, left, right: state.update({'textbox-default': left, 'output_textbox': right}), gradio('interface_state', 'textbox-default', 'output_textbox'), None).then(
        lambda: [gr.update(visible=False), gr.update(visible=True)], None, gradio('Stop-default', 'Generate-default')).then(
        None, None, None, js=f'() => {{{ui.audio_notification_js}}}')

    shared.gradio['Continue-default'].click(
        ui.gather_interface_values, gradio(shared.input_elements), gradio('interface_state')).then(
        lambda: [gr.update(visible=True), gr.update(visible=False)], None, gradio('Stop-default', 'Generate-default')).then(
        continue_and_save_wrapper, gradio('output_textbox', 'textbox-default', 'interface_state', 'prompt_menu-default'), gradio(outputs), show_progress=False).then(
        lambda state, left, right: state.update({'textbox-default': left, 'output_textbox': right}), gradio('interface_state', 'textbox-default', 'output_textbox'), None).then(
        lambda: [gr.update(visible=False), gr.update(visible=True)], None, gradio('Stop-default', 'Generate-default')).then(
        None, None, None, js=f'() => {{{ui.audio_notification_js}}}')

    shared.gradio['Stop-default'].click(stop_everything_event, None, None, queue=False)
    shared.gradio['markdown_render-default'].click(lambda x: x, gradio('output_textbox'), gradio('markdown-default'), queue=False)
    shared.gradio['prompt_menu-default'].change(load_prompt, gradio('prompt_menu-default'), gradio('textbox-default'), show_progress=False)
    shared.gradio['new_prompt-default'].click(handle_new_prompt, None, gradio('textbox-default', 'prompt_menu-default'), show_progress=False)
    shared.gradio['delete_prompt-default'].click(handle_delete_prompt, gradio('prompt_menu-default'), gradio('delete_filename', 'delete_root', 'file_deleter'), show_progress=False)
    shared.gradio['textbox-default'].change(lambda x: f"<span>{count_tokens(x)}</span>", gradio('textbox-default'), gradio('token-counter-default'), show_progress=False)
    shared.gradio['get_logits-default'].click(
        ui.gather_interface_values, gradio(shared.input_elements), gradio('interface_state')).then(
        logits.get_next_logits, gradio('textbox-default', 'interface_state', 'use_samplers-default', 'logits-default'), gradio('logits-default', 'logits-default-previous'), show_progress=False)

    shared.gradio['get_tokens-default'].click(get_token_ids, gradio('textbox-default'), gradio('tokens-default'), show_progress=False)


def autosave_prompt(text, prompt_name):
    """Automatically save the text to the selected prompt file"""
    if prompt_name and text.strip():
        prompt_path = Path("user_data/logs/notebook") / f"{prompt_name}.txt"
        prompt_path.parent.mkdir(parents=True, exist_ok=True)
        prompt_path.write_text(text, encoding='utf-8')


def generate_and_save_wrapper(textbox_content, interface_state, prompt_name):
    """Generate reply and automatically save the result with periodic saves"""
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


def continue_and_save_wrapper(output_textbox, textbox_content, interface_state, prompt_name):
    """Continue generation and automatically save the result with periodic saves"""
    last_save_time = time.monotonic()
    save_interval = 8
    output = output_textbox

    # Initial autosave
    autosave_prompt(output, prompt_name)

    for i, (output, html_output) in enumerate(generate_reply_wrapper(output_textbox, interface_state)):
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


def handle_delete_prompt(prompt):
    return [
        prompt + ".txt",
        "user_data/prompts/",
        gr.update(visible=True)
    ]
