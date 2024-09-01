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
    with gr.Tab('Default', elem_id='default-tab'):
        with gr.Row():
            with gr.Column():
                with gr.Row():
                    shared.gradio['textbox-default'] = gr.Textbox(value='', lines=27, label='Input', elem_classes=['textbox_default', 'add_scrollbar'])
                    shared.gradio['token-counter-default'] = gr.HTML(value="<span>0</span>", elem_classes=["token-counter", "default-token-counter"])

                with gr.Row():
                    shared.gradio['Generate-default'] = gr.Button('Generate', variant='primary')
                    shared.gradio['Stop-default'] = gr.Button('Stop', elem_id='stop')
                    shared.gradio['Continue-default'] = gr.Button('Continue')

                with gr.Row():
                    shared.gradio['prompt_menu-default'] = gr.Dropdown(choices=utils.get_available_prompts(), value='None', label='Prompt', elem_classes='slim-dropdown')
                    ui.create_refresh_button(shared.gradio['prompt_menu-default'], lambda: None, lambda: {'choices': utils.get_available_prompts()}, 'refresh-button', interactive=not mu)
                    shared.gradio['save_prompt-default'] = gr.Button('💾', elem_classes='refresh-button', interactive=not mu)
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
        generate_reply_wrapper, gradio(inputs), gradio(outputs), show_progress=False).then(
        lambda state, left, right: state.update({'textbox-default': left, 'output_textbox': right}), gradio('interface_state', 'textbox-default', 'output_textbox'), None).then(
        None, None, None, js=f'() => {{{ui.audio_notification_js}}}')

    shared.gradio['textbox-default'].submit(
        ui.gather_interface_values, gradio(shared.input_elements), gradio('interface_state')).then(
        generate_reply_wrapper, gradio(inputs), gradio(outputs), show_progress=False).then(
        lambda state, left, right: state.update({'textbox-default': left, 'output_textbox': right}), gradio('interface_state', 'textbox-default', 'output_textbox'), None).then(
        None, None, None, js=f'() => {{{ui.audio_notification_js}}}')

    shared.gradio['Continue-default'].click(
        ui.gather_interface_values, gradio(shared.input_elements), gradio('interface_state')).then(
        generate_reply_wrapper, [shared.gradio['output_textbox']] + gradio(inputs)[1:], gradio(outputs), show_progress=False).then(
        lambda state, left, right: state.update({'textbox-default': left, 'output_textbox': right}), gradio('interface_state', 'textbox-default', 'output_textbox'), None).then(
        None, None, None, js=f'() => {{{ui.audio_notification_js}}}')

    shared.gradio['Stop-default'].click(stop_everything_event, None, None, queue=False)
    shared.gradio['markdown_render-default'].click(lambda x: x, gradio('output_textbox'), gradio('markdown-default'), queue=False)
    shared.gradio['prompt_menu-default'].change(load_prompt, gradio('prompt_menu-default'), gradio('textbox-default'), show_progress=False)
    shared.gradio['save_prompt-default'].click(handle_save_prompt, gradio('textbox-default'), gradio('save_contents', 'save_filename', 'save_root', 'file_saver'), show_progress=False)
    shared.gradio['delete_prompt-default'].click(handle_delete_prompt, gradio('prompt_menu-default'), gradio('delete_filename', 'delete_root', 'file_deleter'), show_progress=False)
    shared.gradio['textbox-default'].change(lambda x: f"<span>{count_tokens(x)}</span>", gradio('textbox-default'), gradio('token-counter-default'), show_progress=False)
    shared.gradio['get_logits-default'].click(
        ui.gather_interface_values, gradio(shared.input_elements), gradio('interface_state')).then(
        logits.get_next_logits, gradio('textbox-default', 'interface_state', 'use_samplers-default', 'logits-default'), gradio('logits-default', 'logits-default-previous'), show_progress=False)

    shared.gradio['get_tokens-default'].click(get_token_ids, gradio('textbox-default'), gradio('tokens-default'), show_progress=False)


def handle_save_prompt(text):
    return [
        text,
        utils.current_time() + ".txt",
        "prompts/",
        gr.update(visible=True)
    ]


def handle_delete_prompt(prompt):
    return [
        prompt + ".txt",
        "prompts/",
        gr.update(visible=True)
    ]
