import gradio as gr

from modules import shared, ui, utils, ui_chat, ui_default, ui_notebook
from modules.utils import gradio


def create_ui():

    with gr.Tab('Generate', elem_id='generate-tab'):
        ui_chat.create_ui()
        ui_default.create_ui()
        ui_notebook.create_ui()