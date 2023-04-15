import os
import json
import datetime

import gradio as gr
import modules.shared as shared
import modules.ui
import pyparsing as pp

from modules.chat import chatbot_wrapper
from pathlib import Path

custom_state = {}
custom_output = []

# I had to steal this from server.py because the program freaks out if I try to `import server`
def load_preset_values(preset_menu, state):
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

    state.update(generate_params)
    return state, *[generate_params[k] for k in ['do_sample', 'temperature', 'top_p', 'typical_p', 'repetition_penalty', 'encoder_repetition_penalty', 'top_k', 'min_length', 'no_repeat_ngram_size', 'num_beams', 'penalty_alpha', 'length_penalty', 'early_stopping']]


# Get all of the presets from the presets folder
def get_presets():
    global custom_state
    presets = []
    filenames = os.listdir("presets/")
    for file in filenames:
        preset = file[:-4]
        presets.append(preset)
        custom_state = load_preset_values(preset, custom_state)[0]
    return ", ".join(presets)

# This is a workaround function because gradio has to access parameters if you want them to be current
def get_params(*args):
    global custom_state
    custom_state = modules.ui.gather_interface_values(*args)
    return json.dumps(custom_state)

# The main function that generates the output, formats the html table, and returns it to the interface
def run(x="", y=""):
    global custom_state
    global custom_output
    custom_state['seed'] = "420691337"
    

    output = "<style>table {border-collapse: collapse;border: 1px solid black;}th, td {border: 1px solid black;padding: 5px;}</style><table><thead><tr><th></th>"

    # Have to format the strings because gradio makes it difficult to pass lists around
    x_strings = pp.common.comma_separated_list.parseString(x).asList()
    y_strings = pp.common.comma_separated_list.parseString(y).asList()

    for i in y_strings:
        output = output + f"<th>{i.strip()}</th>"
    output = output + "</thead><tbody>"
    for i in x_strings:
        output = output + f"<tr><th>{i}</th>"
        if y_strings[0] != '':
            for j in y_strings:
                custom_state = load_preset_values(j.strip(), custom_state)[0]

                # This is the part that actually does the generating
                for new in chatbot_wrapper(i.strip(), custom_state):
                    custom_output = new

                output = output + f"<td><b>{custom_state['name1']}:</b> {custom_output[-1][0]}<br><b>{custom_state['name2']}:</b> {custom_output[-1][1]}</td>"
                custom_output.pop()
                shared.history['internal'].pop()

            output = output + "</tr>"
        else:
                for new in chatbot_wrapper(i.strip(), custom_state):
                    custom_output = new
                output = output + f"<td><b>{custom_state['name1']}:</b> {custom_output[-1][0]}<br><b>{custom_state['name2']}:</b> {custom_output[-1][1]}</td>"
                custom_output.pop()
                shared.history['internal'].pop()
        output = output + "</tr>"
    output = output + "</tbody></table>"

    # Save the output to a file
    # Useful for large grids that don't display well in gradio
    save_filename = f"{datetime.datetime.now().strftime('%Y_%m_%d_%f')}.html"
    with open(Path(f"extensions/xy_grid/outputs/{save_filename}"), 'w') as outfile:
        outfile.write(output)

    # Trying to include a link to easily open the html file in a new tab, but I think this is gonna be more confusing than I expected
    output = output + f"<br><br><a href=\"file/extensions/xy_grid/outputs/{save_filename}\" target=\"_blank\">open html file</a>"
    return output

# Create the interface for the extension (this runs first)
def ui():
    with gr.Accordion("XY Grid", open=True):
        prompt = gr.Textbox(placeholder="Comma separated prompts go here...", label='Input Prompts', interactive=True)
        with gr.Row():
            presets_box = gr.Textbox(placeholder="Presets go here. Click the buttton to the right...", label='Presets', interactive=True)
            refresh_presets = modules.ui.ToolButton(value='\U0001f504', elem_id='refresh-button')
            refresh_presets.click(fn=get_presets, outputs=presets_box)
        generate_grid = gr.Button("generate_grid")
        with gr.Accordion("Generation Parameters for testing", open=False):
            state = gr.HTML(value="the state will go here")
        custom_chat = gr.HTML(value="")

    generate_grid.click(get_params, [shared.gradio[k] for k in shared.input_elements], state).then(run, [prompt, presets_box], custom_chat)
