import gradio as gr
import gc
import json
import os
import re
import time
import zipfile
from pathlib import Path

import numpy as np
import torch
import transformers

from modules import api, chat, shared, training, ui
import modules.shared as shared


def create_output_tab():
    with gr.Column(scale=4):
        shared.gradio['textbox'] = gr.Textbox(value="", elem_id="Outputbox", lines=27, label="Loaded Output File")
    with gr.Column(scale=1):
        create_output_display_menus()


def create_output_display_menus():
    with gr.Row():
        with gr.Column():
            with gr.Row():
                shared.gradio['output_menu'] = gr.Dropdown(choices=get_available_outputs(), value='None',
                                                           label='Output File')
                ui.create_refresh_button(shared.gradio['output_menu'], lambda: None,
                                         lambda: {'choices': get_available_outputs()}, 'refresh-button')
            with gr.Row():
                shared.gradio['OutputDir'] = gr.Button('Open Output Directory')

    shared.gradio['output_menu'].change(load_output, [shared.gradio['output_menu']], [shared.gradio['textbox']],
                                        show_progress=False)
    shared.gradio['OutputDir'].click(open_output_dir)


def create_output_sidebar_menus():
    with gr.Column():
        shared.gradio['OutputName'] = gr.Textbox(value="", elem_id="OutputName", lines=1,
                                                 label="Save Notebook Output Name")
    with gr.Row():
        shared.gradio['Save'] = gr.Button('Save')
        shared.gradio['OutputDir'] = gr.Button('Open Output Directory')
    with gr.Row(elem_id="OutputSaveRow"):
        save_status = gr.Markdown('Status: Ready')
        shared.gradio['SaveStatus'] = save_status


def load_output_sidebar():
    shared.gradio['Save'].click(save_file_event, [shared.gradio['textbox'], shared.gradio['OutputName']],
                                [shared.gradio['SaveStatus']], show_progress=False)
    shared.gradio['OutputDir'].click(open_output_dir)


def get_available_outputs():
    outputs = []
    outputs += sorted(set((k.stem for k in Path('OutputExport').glob('[0-9]*.txt'))), key=str.lower, reverse=True)
    outputs += sorted(set((k.stem for k in Path('OutputExport').glob('*.txt'))), key=str.lower)
    outputs += ['None']
    return outputs


def load_output(fname):
    if fname in ['None', '']:
        return ''
    else:
        with open(Path(f'OutputExport/{fname}.txt'), 'r', encoding='utf-8') as f:
            text = f.read()
            if text[-1] == '\n':
                text = text[:-1]
            return text


def open_output_dir():
    output_dir = os.path.join(os.getcwd(), "OutputExport")
    os.startfile(output_dir)


def save_file_event(textbox, output_name):
    if output_name:
        filename = f"{output_name}.txt"
    else:
        filename = f"{datetime.now().strftime('%Y-%m-%d-%H%M%S')}.txt"
    file_dir = os.path.join(os.getcwd(), "OutputExport")
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)
    file_path = os.path.join(file_dir, filename)
    with open(file_path, "w") as f:
        f.write(textbox)
        # save_status.value = f"**Save of {filename} successful.**"
    return f"Save of {filename} successful."
