import os
import json
import csv
import re

import gradio as gr
import modules.shared as shared
import pyparsing as pp
import modules.ui

from modules.chat import chatbot_wrapper
from pathlib import Path

custom_state = {}
custom_output = []

def load_preset_values(preset_menu, state, return_dict=False):
    generate_params = {
        'do_sample': True,
        'temperature': 1,
        'top_p': 1,
        'typical_p': 1,
        'repetition_penalty': 1,
        'encoder_repetition_penalty': 1,
        'top_k': 50,
        'num_beams': 1,
        'penalty_alpha': 0,
        'min_length': 0,
        'length_penalty': 1,
        'no_repeat_ngram_size': 0,
        'early_stopping': False,
    }
    with open(Path(f'presets/{preset_menu}.txt'), 'r') as infile:
        preset = infile.read()
    for i in preset.splitlines():
        i = i.rstrip(',').strip().split('=')
        if len(i) == 2 and i[0].strip() != 'tokens':
            generate_params[i[0].strip()] = eval(i[1].strip())
    generate_params['temperature'] = min(1.99, generate_params['temperature'])

    if return_dict:
        return generate_params
    else:
        state.update(generate_params)
        return state, *[generate_params[k] for k in ['do_sample', 'temperature', 'top_p', 'typical_p', 'repetition_penalty', 'encoder_repetition_penalty', 'top_k', 'min_length', 'no_repeat_ngram_size', 'num_beams', 'penalty_alpha', 'length_penalty', 'early_stopping']]


def get_presets():
    global custom_state
    presets = []
    filenames = os.listdir("presets/")
    for file in filenames:
        preset = file[:-4]
        presets.append(preset)
        custom_state = load_preset_values(preset, custom_state)[0]
    return ", ".join(presets)

def get_params(*args):
    global custom_state
    custom_state = modules.ui.gather_interface_values(*args)
    return json.dumps(custom_state)

def run(x="",y=""):
    global custom_state
    global custom_output

    output = "<style>table {border-collapse: collapse;border: 1px solid black;}th, td {border: 1px solid black;padding: 5px;}</style><table><thead><tr><th></th>"

    x_strings = pp.common.comma_separated_list.parseString(x).asList()
    y_strings = pp.common.comma_separated_list.parseString(y).asList()

    for i in y_strings:
        output = output + f"<th>{i.strip()}</th>"
    output = output + "</thead><tbody>"
    for i in x_strings:
        output = output + f"<tr><th>{i}</th>"
        for j in y_strings:
            custom_state = load_preset_values(j.strip(), custom_state)[0]
            for new in chatbot_wrapper(i.strip(), custom_state):
                custom_output = new
            output = output + f"<td>{custom_state['name1']}: {custom_output[-1][0]}<br><br>{custom_state['name2']}: {custom_output[-1][1]}</td>"
            custom_output.pop()
            shared.history['internal'].pop()
        output = output + "</tr>"
    output = output + "</tbody></table>"
    return output

def gradio_sucks(flubby):
    return flubby

def ui():
    butt="name1"
    prompt = gr.Textbox(value="name1", label='Input Prompt', interactive=True)
    with gr.Row():
        presets_box = gr.Textbox(placeholder="presets go here...", label='Presets', interactive=True)
        refresh_presets = modules.ui.ToolButton(value='\U0001f504', elem_id='refresh-button')
        refresh_presets.click(fn=get_presets, outputs=presets_box)
    with gr.Accordion("flippity floppity", open=True):
        make_state = gr.Button("make_state")
        test_output = gr.Button("test_output")
        tester = gr.HTML(value="what the fuck is happening?")
        state = gr.HTML(value="the state will go here")
        custom_chat = gr.HTML(value="for the love of God, is this actually going to work???")

    prompt.change(gradio_sucks, prompt, tester)
    make_state.click(get_params, [shared.gradio[k] for k in shared.input_elements], state)
    test_output.click(run, [prompt, presets_box], custom_chat)
