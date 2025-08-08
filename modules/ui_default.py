from pathlib import Path

import gradio as gr

from modules import logits, shared, ui, utils
from modules.prompts import count_tokens, load_prompt
from modules.text_generation import (
    generate_reply_wrapper,
    get_token_ids,
    stop_everything_event
)
from modules.ui_notebook import store_notebook_state_and_debounce
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
                    shared.gradio['rename_prompt-default'] = gr.Button('Rename', elem_classes='refresh-button', interactive=not mu)
                    shared.gradio['delete_prompt-default'] = gr.Button('🗑️', elem_classes='refresh-button', interactive=not mu)

                    # Rename elements (initially hidden)
                    shared.gradio['rename_prompt_to-default'] = gr.Textbox(label="New name", elem_classes=['no-background'], visible=False)
                    shared.gradio['rename_prompt-cancel-default'] = gr.Button('Cancel', elem_classes=['refresh-button'], visible=False)
                    shared.gradio['rename_prompt-confirm-default'] = gr.Button('Confirm', elem_classes=['refresh-button'], variant='primary', visible=False)

                    # Delete confirmation elements (initially hidden)
                    shared.gradio['delete_prompt-cancel-default'] = gr.Button('Cancel', elem_classes=['refresh-button'], visible=False)
                    shared.gradio['delete_prompt-confirm-default'] = gr.Button('Confirm', variant='stop', elem_classes=['refresh-button'], visible=False)

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
        generate_reply_wrapper, gradio('textbox-default', 'interface_state'), gradio(outputs), show_progress=False).then(
        lambda state, left, right: state.update({'textbox-default': left, 'output_textbox': right}), gradio('interface_state', 'textbox-default', 'output_textbox'), None).then(
        lambda: [gr.update(visible=False), gr.update(visible=True)], None, gradio('Stop-default', 'Generate-default')).then(
        None, None, None, js=f'() => {{{ui.audio_notification_js}}}')

    shared.gradio['textbox-default'].submit(
        ui.gather_interface_values, gradio(shared.input_elements), gradio('interface_state')).then(
        lambda: [gr.update(visible=True), gr.update(visible=False)], None, gradio('Stop-default', 'Generate-default')).then(
        generate_reply_wrapper, gradio('textbox-default', 'interface_state'), gradio(outputs), show_progress=False).then(
        lambda state, left, right: state.update({'textbox-default': left, 'output_textbox': right}), gradio('interface_state', 'textbox-default', 'output_textbox'), None).then(
        lambda: [gr.update(visible=False), gr.update(visible=True)], None, gradio('Stop-default', 'Generate-default')).then(
        None, None, None, js=f'() => {{{ui.audio_notification_js}}}')

    shared.gradio['Continue-default'].click(
        ui.gather_interface_values, gradio(shared.input_elements), gradio('interface_state')).then(
        lambda: [gr.update(visible=True), gr.update(visible=False)], None, gradio('Stop-default', 'Generate-default')).then(
        generate_reply_wrapper, gradio('output_textbox', 'interface_state'), gradio(outputs), show_progress=False).then(
        lambda state, left, right: state.update({'textbox-default': left, 'output_textbox': right}), gradio('interface_state', 'textbox-default', 'output_textbox'), None).then(
        lambda: [gr.update(visible=False), gr.update(visible=True)], None, gradio('Stop-default', 'Generate-default')).then(
        None, None, None, js=f'() => {{{ui.audio_notification_js}}}')

    shared.gradio['Stop-default'].click(stop_everything_event, None, None, queue=False)
    shared.gradio['markdown_render-default'].click(lambda x: x, gradio('output_textbox'), gradio('markdown-default'), queue=False)
    shared.gradio['prompt_menu-default'].change(lambda x: (load_prompt(x), ""), gradio('prompt_menu-default'), gradio('textbox-default', 'output_textbox'), show_progress=False)
    shared.gradio['new_prompt-default'].click(handle_new_prompt, None, gradio('prompt_menu-default'), show_progress=False)

    # Input change handler to save input (reusing notebook's debounced saving)
    shared.gradio['textbox-default'].change(
        store_notebook_state_and_debounce,
        gradio('textbox-default', 'prompt_menu-default'),
        None,
        show_progress=False
    )

    shared.gradio['delete_prompt-default'].click(
        lambda: [gr.update(visible=False), gr.update(visible=True), gr.update(visible=True)],
        None,
        gradio('delete_prompt-default', 'delete_prompt-cancel-default', 'delete_prompt-confirm-default'),
        show_progress=False)

    shared.gradio['delete_prompt-cancel-default'].click(
        lambda: [gr.update(visible=True), gr.update(visible=False), gr.update(visible=False)],
        None,
        gradio('delete_prompt-default', 'delete_prompt-cancel-default', 'delete_prompt-confirm-default'),
        show_progress=False)

    shared.gradio['delete_prompt-confirm-default'].click(
        handle_delete_prompt_confirm_default,
        gradio('prompt_menu-default'),
        gradio('prompt_menu-default', 'delete_prompt-default', 'delete_prompt-cancel-default', 'delete_prompt-confirm-default'),
        show_progress=False)

    shared.gradio['rename_prompt-default'].click(
        handle_rename_prompt_click_default,
        gradio('prompt_menu-default'),
        gradio('rename_prompt_to-default', 'rename_prompt-default', 'rename_prompt-cancel-default', 'rename_prompt-confirm-default'),
        show_progress=False)

    shared.gradio['rename_prompt-cancel-default'].click(
        lambda: [gr.update(visible=False), gr.update(visible=True), gr.update(visible=False), gr.update(visible=False)],
        None,
        gradio('rename_prompt_to-default', 'rename_prompt-default', 'rename_prompt-cancel-default', 'rename_prompt-confirm-default'),
        show_progress=False)

    shared.gradio['rename_prompt-confirm-default'].click(
        handle_rename_prompt_confirm_default,
        gradio('rename_prompt_to-default', 'prompt_menu-default'),
        gradio('prompt_menu-default', 'rename_prompt_to-default', 'rename_prompt-default', 'rename_prompt-cancel-default', 'rename_prompt-confirm-default'),
        show_progress=False)

    shared.gradio['textbox-default'].change(lambda x: f"<span>{count_tokens(x)}</span>", gradio('textbox-default'), gradio('token-counter-default'), show_progress=False)
    shared.gradio['get_logits-default'].click(
        ui.gather_interface_values, gradio(shared.input_elements), gradio('interface_state')).then(
        logits.get_next_logits, gradio('textbox-default', 'interface_state', 'use_samplers-default', 'logits-default'), gradio('logits-default', 'logits-default-previous'), show_progress=False)

    shared.gradio['get_tokens-default'].click(get_token_ids, gradio('textbox-default'), gradio('tokens-default'), show_progress=False)


def handle_new_prompt():
    new_name = utils.current_time()

    # Create the new prompt file
    prompt_path = Path("user_data/logs/notebook") / f"{new_name}.txt"
    prompt_path.parent.mkdir(parents=True, exist_ok=True)
    prompt_path.write_text("In this story,", encoding='utf-8')

    return gr.update(choices=utils.get_available_prompts(), value=new_name)


def handle_delete_prompt_confirm_default(prompt_name):
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


def handle_rename_prompt_click_default(current_name):
    return [
        gr.update(value=current_name, visible=True),
        gr.update(visible=False),
        gr.update(visible=True),
        gr.update(visible=True)
    ]


def handle_rename_prompt_confirm_default(new_name, current_name):
    old_path = Path("user_data/logs/notebook") / f"{current_name}.txt"
    new_path = Path("user_data/logs/notebook") / f"{new_name}.txt"

    if old_path.exists() and not new_path.exists():
        old_path.rename(new_path)

    available_prompts = utils.get_available_prompts()
    return [
        gr.update(choices=available_prompts, value=new_name),
        gr.update(visible=False),
        gr.update(visible=True),
        gr.update(visible=False),
        gr.update(visible=False)
    ]
