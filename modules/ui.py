from pathlib import Path

import gradio as gr
import torch

from modules import shared

with open(Path(__file__).resolve().parent / '../css/main.css', 'r') as f:
    css = f.read()
with open(Path(__file__).resolve().parent / '../css/chat.css', 'r') as f:
    chat_css = f.read()
with open(Path(__file__).resolve().parent / '../css/main.js', 'r') as f:
    main_js = f.read()
with open(Path(__file__).resolve().parent / '../css/chat.js', 'r') as f:
    chat_js = f.read()

refresh_symbol = '\U0001f504'  # 🔄
theme = gr.themes.Default(
    font=['Helvetica', 'ui-sans-serif', 'system-ui', 'sans-serif'],
    font_mono=['IBM Plex Mono', 'ui-monospace', 'Consolas', 'monospace'],
).set(
    border_color_primary='#c5c5d2',
    button_large_padding='6px 12px'
)

def list_model_elements():
    elements = ['cpu_memory', 'auto_devices', 'disk', 'cpu', 'bf16', 'load_in_8bit', 'wbits', 'groupsize', 'model_type', 'pre_layer']
    for i in range(torch.cuda.device_count()):
        elements.append(f'gpu_memory_{i}')
    return elements


def list_interface_input_elements(chat=False):
    elements = ['max_new_tokens', 'seed', 'temperature', 'top_p', 'top_k', 'typical_p', 'repetition_penalty', 'encoder_repetition_penalty', 'no_repeat_ngram_size', 'min_length', 'do_sample', 'penalty_alpha', 'num_beams', 'length_penalty', 'early_stopping', 'add_bos_token', 'ban_eos_token', 'truncation_length', 'custom_stopping_strings', 'skip_special_tokens']
    if chat:
        elements += ['name1', 'name2', 'greeting', 'context', 'end_of_turn', 'chat_prompt_size', 'chat_generation_attempts', 'stop_at_newline', 'mode', 'instruction_template']
    elements += list_model_elements()
    return elements


def gather_interface_values(*args):
    output = {}
    for i, element in enumerate(shared.input_elements):
        output[element] = args[i]
    return output


def apply_interface_values(state):
    return [state[i] for i in list_interface_input_elements(chat=shared.is_chat())]


class ToolButton(gr.Button, gr.components.FormComponent):
    """Small button with single emoji as text, fits inside gradio forms"""

    def __init__(self, **kwargs):
        super().__init__(variant="tool", **kwargs)

    def get_block_name(self):
        return "button"


def create_refresh_button(refresh_component, refresh_method, refreshed_args, elem_id):
    def refresh():
        refresh_method()
        args = refreshed_args() if callable(refreshed_args) else refreshed_args

        for k, v in args.items():
            setattr(refresh_component, k, v)

        return gr.update(**(args or {}))

    refresh_button = ToolButton(value=refresh_symbol, elem_id=elem_id)
    refresh_button.click(
        fn=refresh,
        inputs=[],
        outputs=[refresh_component]
    )
    return refresh_button
