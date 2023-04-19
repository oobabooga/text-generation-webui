import os
import json
import datetime
import random

import gradio as gr
import modules.shared as shared
import pyparsing as pp

from modules.chat import chatbot_wrapper, load_character
from pathlib import Path

axis_type = {'x': "prompts", 'y': "presets"}
custom_state = {}
gen_output = []


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
    custom_state['preset_menu'] = preset_menu
    return state, *[generate_params[k] for k in ['do_sample', 'temperature', 'top_p', 'typical_p', 'repetition_penalty', 'encoder_repetition_penalty', 'top_k', 'min_length', 'no_repeat_ngram_size', 'num_beams', 'penalty_alpha', 'length_penalty', 'early_stopping']]


# Get all of the characters from the character folder
def get_characters():
    paths = (x for x in Path('characters').iterdir() if x.suffix in ('.json', '.yaml', '.yml'))
    return ", ".join(['None'] + sorted(set((k.stem for k in paths if k.stem != "instruction-following")), key=str.lower))


# Get all of the presets from the presets folder
def get_presets():
    presets = []
    filenames = os.listdir("presets/")
    for file in filenames:
        preset = file[:-4]
        presets.append(preset)
    return ", ".join(presets)


# Returns the correct results for the axis type chosen by the axis dropdown box
def fill_axis(option):
    global axis_get
    global custom_state
    if option == "prompts":
        return gr.update(label=option, value=custom_state['textbox'])
    else:
        return gr.update(label=option, value=axis_get.get(option))


# Sets the type of data each axis will use
def set_axis(x, y):
    global axis_type
    axis_type.update({'x': x})
    axis_type.update({'y': y})


# Parse the type of the X axis and alter custom_state accordingly
# If you want to add more axes, this is where you would do it. 
# Add logic here, add an entry to axis_type{}, and add it to the dropdown menus
def parse_axis(axis, value):
    global custom_state
    global axis_type

    # PRESETS
    if axis_type[axis] == "presets":
        if value.strip() != "":
            custom_state = load_preset_values(value.strip(), custom_state)[0]
        else:
            custom_state = load_preset_values(shared.gradio['preset_menu'].value, custom_state)[0]
    # CHARACTERS
    elif axis_type[axis] == "characters":
        if value.strip() != "":
            custom_state['character_menu'] = value.strip()
        else:
            custom_state['character_menu'] = shared.gradio["character_menu"].value
        custom_state.update({k: v for k, v in zip(['name1', 'name2', 'character_picture', 'greeting', 'context', 'end_of_turn', 'display'], load_character(custom_state['character_menu'], custom_state['name1'], custom_state['name2'], custom_state['mode']))})
    # SEEDS
    elif axis_type[axis] == "seeds":
        if value.strip() != "":
            custom_state['seed'] = value.strip()
        else:
            custom_state['seed'] = shared.gradio['seed'].value
#    # TEMPLATE
#    elif axis_type[axis] == "":
#        if value.strip() != "":
#            custom_state[''] = value.strip()
#        else:
#            custom_state[''] = shared.gradio[''].value
    return None


def run(constant_seed, seed_value, use_history, x="", y=""):

    global custom_state
    global gen_output
    global axis_type

    if constant_seed:
        if seed_value == "-1":
            custom_state['seed'] = random.randint(1, 2**31)
        else:
            custom_state['seed'] = seed_value

    temp_history = shared.history['internal']

    # Gather output json info, from before the X/Y parameters take effect
    output_json = {k: custom_state[k] for k in shared.input_elements}

    if custom_state['custom_stopping_strings'] is None:
        custom_state['custom_stopping_strings'] = ""

    # Have to format the strings because gradio makes it difficult to pass lists around
    if x == "":
        x_strings = ""
    else:
        x_strings = pp.common.comma_separated_list.parseString(x).asList()
    if y == "":
        y_strings = ""
    else:
        y_strings = pp.common.comma_separated_list.parseString(y).asList()

    output = "<style>table {border-collapse: collapse;border: 1px solid black;}th, td {border: 1px solid black;padding: 5px;}</style><table><thead><tr><th></th>"

    if axis_type['x'] == axis_type['y']:
        return "<h1><span style=\"color: red;\">ERROR: both axes cannot be the same setting</span>"
    
    # Run as if x axis is prompts
    elif axis_type['x'] == "prompts":
        for i in x_strings:
            output = output + f"<th>{i.strip()}</th>"
        output = output + "</thead><tbody>"
        if y_strings != '':
            for i in y_strings:
                output = output + f"<tr><th>{i.strip()}</th>"
                for j in x_strings:

                    # parse the type of the Y axis and alter custom_state accordingly
                    parse_axis("y", i)
                    
                    # This was at the top of the function, but for some reason it broke with a recent update
                    if not use_history:
                        shared.history['internal'] = shared.history['internal'][:1]

                    # This is the part that actually does the generating
                    for new in chatbot_wrapper(j.strip().strip('"'), custom_state):
                        gen_output = new

                    output = output + f"<td><h3><b>{custom_state['name1']}:</b></h3> {gen_output[-1][0]}<br><h3><b>{custom_state['name2']}:</b></h3> {gen_output[-1][1]}</td>"
                    gen_output.pop()
                    shared.history['internal'].pop()

                output = output + "</tr>"
        else:
            output = output + "<tr><th></th>"
            for i in x_strings:
                for new in chatbot_wrapper(i.strip().strip('"'), custom_state):
                    gen_output = new
                output = output + f"<td><h3><b>{custom_state['name1']}:</b></h3> {gen_output[-1][0]}<br><h3><b>{custom_state['name2']}:</b></h3> {gen_output[-1][1]}</td>"

                # Remove the last outputs, so they don't influence future generations
                gen_output.pop()
                shared.history['internal'].pop()

            output = output + "</tr>"

    # Run as if y axis is prompts
    elif axis_type['y'] == "prompts":
        for i in x_strings:
            output = output + f"<th>{i.strip()}</th>"
        output = output + "</thead><tbody>"
        if x_strings != '':
            for i in y_strings:
                output = output + f"<tr><th>{i.strip()}</th>"
                for j in x_strings:

                    # parse the type of the X axis and alter custom_state accordingly
                    parse_axis("x", j)

                    # This was at the top of the function, but for some reason it broke with a recent update
                    if not use_history:
                        shared.history['internal'] = shared.history['internal'][:1]

                    # This is the part that actually does the generating
                    for new in chatbot_wrapper(i.strip().strip('"'), custom_state):
                        gen_output = new

                    output = output + f"<td><h3><b>{custom_state['name1']}:</b></h3> {gen_output[-1][0]}<br><h3><b>{custom_state['name2']}:</b></h3> {gen_output[-1][1]}</td>"
                    gen_output.pop()
                    shared.history['internal'].pop()

                output = output + "</tr>"
        else:
            for i in y_strings:
                for new in chatbot_wrapper(i.strip().strip('"'), custom_state):
                    gen_output = new
                output = output + f"<tr><tr><th>{i.strip()}</th><td><h3><b>{custom_state['name1']}:</b></h3> {gen_output[-1][0]}<br><h3><b>{custom_state['name2']}:</b></h3> {gen_output[-1][1]}</td></tr>"

                # Remove the last outputs, so they don't influence future generations
                gen_output.pop()
                shared.history['internal'].pop()

    # Take the prompts from custom_state['textbox']
    else:
        for i in x_strings:
            output = output + f"<th>{i.strip()}</th>"
        output = output + "</thead><tbody>"
        if y_strings != '' and x_strings != '':
            for i in y_strings:
                output = output + f"<tr><th>{i.strip()}</th>"
                for j in x_strings:
                    # parse the types of the axes and alter custom_state accordingly
                    parse_axis("y", i)
                    parse_axis("x", j)

                    # This was at the top of the function, but for some reason it broke with a recent update
                    if not use_history:
                        shared.history['internal'] = shared.history['internal'][:1]

                    # This is the part that actually does the generating
                    for new in chatbot_wrapper(custom_state['textbox'].strip(), custom_state):
                        gen_output = new

                    output = output + f"<td><h3><b>{custom_state['name1']}:</b></h3> {gen_output[-1][0]}<br><h3><b>{custom_state['name2']}:</b></h3> {gen_output[-1][1]}</td>"
                    gen_output.pop()
                    shared.history['internal'].pop()

                output = output + "</tr>"

        elif x_strings != '':
            output = output + "<tr><th></th>"
            for j in x_strings:

                # parse the types of the axes and alter custom_state accordingly
                parse_axis("x", j)

                # This was at the top of the function, but for some reason it broke with a recent update
                if not use_history:
                    shared.history['internal'] = shared.history['internal'][:1]

                # Run the actual text generator
                for new in chatbot_wrapper(custom_state['textbox'].strip(), custom_state):
                    gen_output = new
                output = output + f"<td><h3><b>{custom_state['name1']}:</b></h3> {gen_output[-1][0]}<br><h3><b>{custom_state['name2']}:</b></h3> {gen_output[-1][1]}</td>"

                # Remove the last outputs, so they don't influence future generations
                gen_output.pop()
                shared.history['internal'].pop()

            output = output + "</tr>"
        
        elif y_strings != '':
            for i in y_strings:
                # parse the types of the axes and alter custom_state accordingly
                parse_axis("y", i)

                # This was at the top of the function, but for some reason it broke with a recent update
                if not use_history:
                    shared.history['internal'] = shared.history['internal'][:1]
                    
                # Run the actual text generator
                for new in chatbot_wrapper(custom_state['textbox'].strip(), custom_state):
                    gen_output = new
                output = output + f"<tr><th>{i.strip()}</th><td><h3><b>{custom_state['name1']}:</b></h3> {gen_output[-1][0]}<br><h3><b>{custom_state['name2']}:</b></h3> {gen_output[-1][1]}</td></tr>"

                # Remove the last outputs, so they don't influence future generations
                gen_output.pop()
                shared.history['internal'].pop()

        else:
            return "<h1><span style=\"color: red;\">ERROR: both fields are empty</span>"

    output = output + "</tbody></table>"

    # Save the output to a file
    output_folder = Path("extensions/xy_grid/outputs")
    if not Path(output_folder).exists():
        os.mkdir(output_folder)
    output_filename = Path(f"{datetime.datetime.now().strftime('%Y_%m_%d_%H%M%S')}")
    with open(Path(f"{output_folder}/{output_filename}.html"), 'w') as outfile:
        outfile.write(output)
    with open(Path(f"{output_folder}/{output_filename}.json"), 'w') as outparams:
        outparams.write(json.dumps(output_json))

    # Include a link to the generated HTML file
    output = output + f"<br><br><h2><a href=\"file/extensions/xy_grid/outputs/{output_filename}.html\" target=\"_blank\">[ <em>open html file 🔗</em> ]</a></h2>"

    # Clean up some of the changes that were made during this generation
    custom_state['seed'] = -1
    shared.history['internal'] = temp_history
    return output


# Necessary for some stuff because gradio
def swap_axes(x_menu, x_data, y_menu, y_data):
    return y_menu, y_data, gr.update(label=y_menu), x_menu, x_data, gr.update(label=x_menu)


def toggle_visible(var):
    if not var:
        custom_state['seed'] = -1
    return gr.update(visible=var)


axis_get = {
        'presets': get_presets(),
        'prompts': "",
        'characters': get_characters(),
        'seeds': "-1"
        }


# Create the interface for the extension (this runs first)
def ui():
    global custom_state
    global axis_type
    global axis_get

    # Grab all the variable from shared.gradio and put them in the custom_state dictionary
    custom_state.update({k: v for k, v in zip([key for key in shared.gradio if not isinstance(shared.gradio[key], (gr.Blocks, gr.Button, gr.State))], [shared.gradio[k].value for k in [key for key in shared.gradio] if not isinstance(shared.gradio[k], (gr.Blocks, gr.Button, gr.State))])})

    # Track changes to all variables in shared.gradio
    shared.gradio['add_bos_token'].change(lambda x: custom_state.update({'add_bos_token': x}), shared.gradio['add_bos_token'], [])
    shared.gradio['auto_devices'].change(lambda x: custom_state.update({'auto_devices': x}), shared.gradio['auto_devices'], [])
    shared.gradio['ban_eos_token'].change(lambda x: custom_state.update({'ban_eos_token': x}), shared.gradio['ban_eos_token'], [])
    shared.gradio['bf16'].change(lambda x: custom_state.update({'bf16': x}), shared.gradio['bf16'], [])
    shared.gradio['bool_menu'].change(lambda x: custom_state.update({'bool_menu': x}), shared.gradio['bool_menu'], [])
    shared.gradio['character_menu'].change(lambda x: custom_state.update({'character_menu': x}), shared.gradio['character_menu'], [])
    shared.gradio['character_picture'].change(lambda x: custom_state.update({'character_picture': x}), shared.gradio['character_picture'], [])
    shared.gradio['chat_generation_attempts'].change(lambda x: custom_state.update({'chat_generation_attempts': x}), shared.gradio['chat_generation_attempts'], [])
    shared.gradio['chat_prompt_size'].change(lambda x: custom_state.update({'chat_prompt_size': x}), shared.gradio['chat_prompt_size'], [])
    shared.gradio['context'].change(lambda x: custom_state.update({'context': x}), shared.gradio['context'], [])
    shared.gradio['cpu'].change(lambda x: custom_state.update({'cpu': x}), shared.gradio['cpu'], [])
    shared.gradio['cpu_memory'].change(lambda x: custom_state.update({'cpu_memory': x}), shared.gradio['cpu_memory'], [])
    shared.gradio['custom_model_menu'].change(lambda x: custom_state.update({'custom_model_menu': x}), shared.gradio['custom_model_menu'], [])
    shared.gradio['custom_stopping_strings'].change(lambda x: custom_state.update({'custom_stopping_strings': x}), shared.gradio['custom_stopping_strings'], [])
    shared.gradio['disk'].change(lambda x: custom_state.update({'disk': x}), shared.gradio['disk'], [])
    shared.gradio['display'].change(lambda x: custom_state.update({'display': x}), shared.gradio['display'], [])
    shared.gradio['do_sample'].change(lambda x: custom_state.update({'do_sample': x}), shared.gradio['do_sample'], [])
    shared.gradio['download'].change(lambda x: custom_state.update({'download': x}), shared.gradio['download'], [])
    shared.gradio['early_stopping'].change(lambda x: custom_state.update({'early_stopping': x}), shared.gradio['early_stopping'], [])
    shared.gradio['encoder_repetition_penalty'].change(lambda x: custom_state.update({'encoder_repetition_penalty': x}), shared.gradio['encoder_repetition_penalty'], [])
    shared.gradio['end_of_turn'].change(lambda x: custom_state.update({'end_of_turn': x}), shared.gradio['end_of_turn'], [])
    shared.gradio['extensions_menu'].change(lambda x: custom_state.update({'extensions_menu': x}), shared.gradio['extensions_menu'], [])
    shared.gradio['gpu_memory_0'].change(lambda x: custom_state.update({'gpu_memory_0': x}), shared.gradio['gpu_memory_0'], [])
    shared.gradio['greeting'].change(lambda x: custom_state.update({'greeting': x}), shared.gradio['greeting'], [])
    shared.gradio['groupsize'].change(lambda x: custom_state.update({'groupsize': x}), shared.gradio['groupsize'], [])
    shared.gradio['instruction_template'].change(lambda x: custom_state.update({'instruction_template': x}), shared.gradio['instruction_template'], [])
    shared.gradio['interface_modes_menu'].change(lambda x: custom_state.update({'interface_modes_menu': x}), shared.gradio['interface_modes_menu'], [])
    shared.gradio['length_penalty'].change(lambda x: custom_state.update({'length_penalty': x}), shared.gradio['length_penalty'], [])
    shared.gradio['load_in_8bit'].change(lambda x: custom_state.update({'load_in_8bit': x}), shared.gradio['load_in_8bit'], [])
    shared.gradio['lora_menu'].change(lambda x: custom_state.update({'lora_menu': x}), shared.gradio['lora_menu'], [])
    shared.gradio['max_new_tokens'].change(lambda x: custom_state.update({'max_new_tokens': x}), shared.gradio['max_new_tokens'], [])
    shared.gradio['min_length'].change(lambda x: custom_state.update({'min_length': x}), shared.gradio['min_length'], [])
    shared.gradio['mode'].change(lambda x: custom_state.update({'mode': x}), shared.gradio['mode'], [])
    shared.gradio['model_menu'].change(lambda x: custom_state.update({'model_menu': x}), shared.gradio['model_menu'], [])
    shared.gradio['model_status'].change(lambda x: custom_state.update({'model_status': x}), shared.gradio['model_status'], [])
    shared.gradio['model_type'].change(lambda x: custom_state.update({'model_type': x}), shared.gradio['model_type'], [])
    shared.gradio['name1'].change(lambda x: custom_state.update({'name1': x}), shared.gradio['name1'], [])
    shared.gradio['name2'].change(lambda x: custom_state.update({'name2': x}), shared.gradio['name2'], [])
    shared.gradio['no_repeat_ngram_size'].change(lambda x: custom_state.update({'no_repeat_ngram_size': x}), shared.gradio['no_repeat_ngram_size'], [])
    shared.gradio['num_beams'].change(lambda x: custom_state.update({'num_beams': x}), shared.gradio['num_beams'], [])
    shared.gradio['penalty_alpha'].change(lambda x: custom_state.update({'penalty_alpha': x}), shared.gradio['penalty_alpha'], [])
    shared.gradio['pre_layer'].change(lambda x: custom_state.update({'pre_layer': x}), shared.gradio['pre_layer'], [])
    shared.gradio['preset_menu'].change(lambda x: custom_state.update({'preset_menu': x}), shared.gradio['preset_menu'], [])
    shared.gradio['repetition_penalty'].change(lambda x: custom_state.update({'repetition_penalty': x}), shared.gradio['repetition_penalty'], [])
    shared.gradio['seed'].change(lambda x: custom_state.update({'seed': x}), shared.gradio['seed'], [])
    shared.gradio['skip_special_tokens'].change(lambda x: custom_state.update({'skip_special_tokens': x}), shared.gradio['skip_special_tokens'], [])
    shared.gradio['softprompts_menu'].change(lambda x: custom_state.update({'softprompts_menu': x}), shared.gradio['softprompts_menu'], [])
    shared.gradio['stop_at_newline'].change(lambda x: custom_state.update({'stop_at_newline': x}), shared.gradio['stop_at_newline'], [])
    shared.gradio['temperature'].change(lambda x: custom_state.update({'temperature': x}), shared.gradio['temperature'], [])
    shared.gradio['textbox'].change(lambda x: custom_state.update({'textbox': x}), shared.gradio['textbox'], [])
    shared.gradio['top_k'].change(lambda x: custom_state.update({'top_k': x}), shared.gradio['top_k'], [])
    shared.gradio['top_p'].change(lambda x: custom_state.update({'top_p': x}), shared.gradio['top_p'], [])
    shared.gradio['truncation_length'].change(lambda x: custom_state.update({'truncation_length': x}), shared.gradio['truncation_length'], [])
    shared.gradio['typical_p'].change(lambda x: custom_state.update({'typical_p': x}), shared.gradio['typical_p'], [])
    shared.gradio['upload_chat_history'].change(lambda x: custom_state.update({'upload_chat_history': x}), shared.gradio['upload_chat_history'], [])
    shared.gradio['upload_img_bot'].change(lambda x: custom_state.update({'upload_img_bot': x}), shared.gradio['upload_img_bot'], [])
    shared.gradio['upload_img_tavern'].change(lambda x: custom_state.update({'upload_img_tavern': x}), shared.gradio['upload_img_tavern'], [])
    shared.gradio['upload_json'].change(lambda x: custom_state.update({'upload_json': x}), shared.gradio['upload_json'], [])
    shared.gradio['upload_softprompt'].change(lambda x: custom_state.update({'upload_softprompt': x}), shared.gradio['upload_softprompt'], [])
    shared.gradio['wbits'].change(lambda x: custom_state.update({'wbits': x}), shared.gradio['wbits'], [])
    shared.gradio['your_picture'].change(lambda x: custom_state.update({'your_picture': x}), shared.gradio['your_picture'], [])

    with gr.Accordion("XY Grid", open=True):

        # Axis selections and inputs
        with gr.Row():
            x_type = gr.Dropdown(label='X Axis', choices=list(["prompts", "presets", "characters", "seeds"]), value="prompts", interactive=True)
            x_input = gr.Textbox(label=x_type.value, interactive=True)
        with gr.Row():
            y_type = gr.Dropdown(label='Y Axis', choices=["prompts", "presets", "characters", "seeds"], value="presets", interactive=True)
            y_input = gr.Textbox(label=y_type.value, value=axis_get[y_type.value], interactive=True)
        x_type.select(set_axis, [x_type, y_type], []).then(fill_axis, x_type, x_input)
        y_type.select(set_axis, [x_type, y_type], []).then(fill_axis, y_type, y_input)
        x_type.change(set_axis, [x_type, y_type], [])
        y_type.change(set_axis, [x_type, y_type], [])
        with gr.Row():
            swap_xy = gr.Button(value='Swap X/Y Axes 🔀')
        with gr.Row():
            seed_input = gr.Checkbox(label='Use a constant seed', value=False)
            use_history = gr.Checkbox(label='Use character\'s chat history', value=False)
        with gr.Row():
            seed_value = gr.Textbox(label='Seed', value="-1", visible=False, interactive=True)
        seed_input.change(toggle_visible, seed_input, seed_value)
        swap_xy.click(swap_axes, [x_type, x_input, y_type, y_input], [x_type, x_input, x_input, y_type, y_input, y_input])

        generate_grid = gr.Button("generate_grid")
        custom_chat = gr.HTML(value="")

        generate_grid.click(run, [seed_input, seed_value, use_history, x_input, y_input], custom_chat)
