from pathlib import Path

import gradio as gr

refresh_symbol = '\U0001f504'  # 🔄

with open(Path(__file__).resolve().parent / '../css/main.css', 'r') as f:
    css = f.read()
with open(Path(__file__).resolve().parent / '../css/chat.css', 'r') as f:
    chat_css = f.read()
with open(Path(__file__).resolve().parent / '../css/main.js', 'r') as f:
    main_js = f.read()
with open(Path(__file__).resolve().parent / '../css/chat.js', 'r') as f:
    chat_js = f.read()


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
